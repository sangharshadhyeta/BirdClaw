from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Literal, Optional

# Max chars injected as file context per write call (~1500 tokens).
# Prevents context overflow on large growing documents.
_MAX_CTX_CHARS = 6_000

from .subtask_manifest import SubtaskItem, SubtaskManifest
from . import subtask_planner as _planner
from . import subtask_verifier as _verifier
from birdclaw.tools.write_guard import pre_write_check, post_write_check
from birdclaw.llm.schemas import WRITE_ITEM_SCHEMA as _WRITE_ITEM_SCHEMA

logger = logging.getLogger(__name__)

MAX_ITEM_RETRIES = 2
_FILE_TAIL_LINES  = 30   # lines of written file injected as context each call


# ---------------------------------------------------------------------------
# Result returned to loop.py
# ---------------------------------------------------------------------------

class StageResult:
    def __init__(self, manifest: SubtaskManifest, written_path: str, reflect_hint: str = ""):
        self.manifest     = manifest
        self.written_path = written_path
        self.reflect_hint = reflect_hint  # "done" | "deepen: <gap>" | ""

    @property
    def summary(self) -> str:
        complete = [it.title for it in self.manifest.items if it.status == "complete"]
        partial  = [it.title for it in self.manifest.items if it.status == "partial"]
        missing  = [it.title for it in self.manifest.items if it.status in ("missing", "regressed")]
        parts = [f"{len(complete)}/{self.manifest.total} items complete"]
        if partial:
            parts.append(f"partial: {', '.join(partial)}")
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _resolve_output_path(path: str) -> Path:
    """Resolve and validate a write path against workspace roots.

    Raises PermissionError if the resolved path is outside all workspace roots
    so callers can surface a clear error instead of silently writing elsewhere.
    """
    import os as _os
    from birdclaw.config import settings

    p = Path(path)
    if not p.is_absolute():
        p = Path(_os.getcwd()) / p
    p = p.resolve()

    for root in settings.workspace_roots:
        try:
            root_r = root.resolve()
        except OSError:
            continue
        if p == root_r or p.is_relative_to(root_r):
            return p

    raise PermissionError(
        f"path {p} is outside workspace roots {[str(r) for r in settings.workspace_roots]}"
    )


def _read_file(path: str) -> str:
    try:
        return _resolve_output_path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _write_file(path: str, content: str, append: bool) -> str:
    p = _resolve_output_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)
    return f"{'Appended' if append else 'Wrote'} {len(content)} chars to {str(p)}"


def _file_tail(path: str, n: int = _FILE_TAIL_LINES) -> str:
    """Return the last N lines — fallback when search finds no continuation point."""
    content = _read_file(path)
    if not content:
        return ""
    lines = content.splitlines()
    return "\n".join(lines[-n:] if len(lines) > n else lines)


def _read_for_context(path: str, item_title: str, file_type: str) -> str:
    """Progressive disclosure — 4-step hierarchy, most specific first.

    1. BIRDCLAW.md     — workspace index lines relevant to this item (big picture)
    2. find_section    — exact section/function in the target file matching this item
    3. search_relevant — goal-relevant lines if no exact section found
    4. find_continuation_point — last header → EOF (original fallback)

    Returns a pre-labeled context string ready for LLM injection.
    """
    import os as _os
    from pathlib import Path as _Path
    from birdclaw.tools.line_search import find_section, find_continuation_point, search_relevant

    parts: list[str] = []

    # Step 1: task log — what this task has done so far.
    # BIRDCLAW.md lives in the task subfolder (parent of the output file),
    # not at cwd root, so tasks never bleed into each other.
    task_dir    = _Path(path).parent
    birdclaw_md = task_dir / "BIRDCLAW.md"
    if not birdclaw_md.is_file():
        birdclaw_md = _Path(_os.getcwd()) / "BIRDCLAW.md"  # legacy fallback
    if birdclaw_md.is_file():
        ws_ctx = search_relevant(item_title, [birdclaw_md], context_lines=1, max_results=3)
        if ws_ctx:
            logger.debug("[ctx] BIRDCLAW.md hit  item=%r  chars=%d", item_title[:40], len(ws_ctx))
            parts.append(f"[BIRDCLAW.md]\n{ws_ctx}")
        else:
            logger.debug("[ctx] BIRDCLAW.md miss  item=%r", item_title[:40])
    else:
        logger.debug("[ctx] no BIRDCLAW.md  task_dir=%s", task_dir)

    # Step 2: exact section/function in the target file matching this item
    section = find_section(path, item_title, file_type)
    if section:
        logger.debug("[ctx] find_section hit  item=%r  path=%s  chars=%d", item_title[:40], _Path(path).name, len(section))
        parts.append(f"[{path} — {item_title}]\n{section}")
        return "\n\n".join(parts)

    # Step 3: goal-relevant lines scattered through the file
    rel = search_relevant(item_title, [path], context_lines=2)
    if rel:
        logger.debug("[ctx] search_relevant hit  item=%r  path=%s  chars=%d", item_title[:40], _Path(path).name, len(rel))
        parts.append(f"[{path}]\n{rel}")
        return "\n\n".join(parts)

    # Step 4: last section → EOF (natural continuation point)
    cont = find_continuation_point(path, file_type)
    if cont:
        logger.debug("[ctx] continuation fallback  item=%r  path=%s  chars=%d", item_title[:40], _Path(path).name, len(cont))
        parts.append(f"[{path}]\n{cont}")
        return "\n\n".join(parts)

    logger.debug("[ctx] no context found  item=%r  path=%s", item_title[:40], _Path(path).name)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Per-call message builder — file is the memory, not the message list
# ---------------------------------------------------------------------------

def _build_call_messages(
    item:       SubtaskItem,
    manifest:   SubtaskManifest,
    attempt:    int,
    step:       int,
    max_steps:  int,
    error_hint: str = "",
) -> list:
    """Build a fresh 3-message context for one LLM write call.

    Progressive disclosure: _read_for_context returns a pre-labeled context
    string (BIRDCLAW.md + matching section + relevant lines). No shared history.
    On retry (attempt > 0) the verifier diff replaces the normal instruction.
    """
    from birdclaw.llm.types import Message
    from birdclaw.agent.prompts import SYSTEM

    if attempt > 0:
        # Re-read file and show the verifier's gap analysis
        fc   = _read_file(manifest.file_path)
        diff = _verifier.run(manifest, fc)
        instruction = diff.resume_context
        if len(instruction) > _MAX_CTX_CHARS:
            instruction = instruction[:_MAX_CTX_CHARS] + "\n...[truncated]"
        if error_hint:
            instruction += f"\n\nNote from last attempt: {error_hint}"
    else:
        done = [it for it in manifest.items if it.status == "complete"]
        done_str = ", ".join(it.title for it in done) or "none"

        # Progressive disclosure: pre-labeled context (BIRDCLAW.md + file section)
        ctx = _read_for_context(manifest.file_path, item.title, manifest.file_type)
        if ctx and len(ctx) > _MAX_CTX_CHARS:
            ctx = ctx[:_MAX_CTX_CHARS] + "\n...[truncated for context budget]"
        file_state = ctx if ctx else f"[{manifest.file_path} — empty, start fresh]"

        _marker_hint = (
            f"Start the section with exactly: ## {item.title}\n"
            if manifest.file_type == "code"
            else f"Start the section with exactly: ## {item.title}\n"
        )
        instruction = (
            f"Goal: {manifest.stage_goal}\n"
            f"Done: {done_str}\n"
            f"{file_state}\n\n"
            f"Write next: {item.title} (min {item.expected_min_chars} chars)\n"
            f"{_marker_hint}"
            f"Append after the last line above. Do not repeat earlier content.\n"
            f"[{step}/{max_steps}]"
        )
        if error_hint:
            instruction += f"\n\nNote: {error_hint}"

    return [
        Message(role="system", content=SYSTEM),
        Message(role="user",   content=instruction),
    ]


# ---------------------------------------------------------------------------
# Single item write + verify (with retries)
# ---------------------------------------------------------------------------

def _write_item(
    item:       SubtaskItem,
    manifest:   SubtaskManifest,
    llm_client: Any,
    step:       int,
    max_steps:  int,
    on_write:   Callable[[str], None],
    force_append: bool = False,
) -> bool:
    """Write one item with up to MAX_ITEM_RETRIES retries. Returns True if complete."""
    return _write_item_with_messages(item, manifest, llm_client, step, max_steps, on_write, None, force_append=force_append)


def _write_item_with_messages(
    item:            SubtaskItem,
    manifest:        SubtaskManifest,
    llm_client:      Any,
    step:            int,
    max_steps:       int,
    on_write:        Callable[[str], None],
    first_messages:  list | None,
    force_append:    bool = False,
) -> bool:
    """Write one item; uses pre-built messages for attempt 0 if provided."""
    item.status = "in_progress"
    is_first_item = item.index == 0
    error_hint = ""

    for attempt in range(MAX_ITEM_RETRIES + 1):
        logger.info("[item] attempt=%d/%d  item=%r  file=%s", attempt, MAX_ITEM_RETRIES, item.title[:40], manifest.file_path)
        if attempt == 0 and first_messages is not None:
            call_messages = first_messages
        else:
            call_messages = _build_call_messages(item, manifest, attempt, step, max_steps, error_hint)
        error_hint = ""  # reset; will be set below if this attempt produces an error

        result = llm_client.generate(
            call_messages, format_schema=_WRITE_ITEM_SCHEMA, thinking=True,
        )
        raw = result.content or ""

        from birdclaw.agent.planner import parse_format_response as _pfr
        parsed = _pfr(raw) or {}

        if parsed and "content" not in parsed and "path" not in parsed:
            error_hint = (
                f"Your JSON is missing required fields. "
                f"Output: {{\"path\": \"{manifest.file_path}\", \"content\": \"<text>\"}}. "
                f"You used keys: {list(parsed.keys())[:4]}"
            )
            logger.warning("subtask_executor: wrong schema for item %r attempt %d", item.title, attempt)
            continue

        content = parsed.get("content", raw)
        section = parsed.get("section", "")

        # Validate the path: only accept absolute paths that match the manifest.
        # Relative paths (e.g. "3/500") from the model are always wrong — they
        # resolve relative to CWD and create stray files/dirs.
        raw_path = parsed.get("path", manifest.file_path)
        if raw_path and Path(raw_path).is_absolute() and raw_path == manifest.file_path:
            path = raw_path
        else:
            if raw_path and raw_path != manifest.file_path:
                logger.warning(
                    "subtask_executor: ignoring model path %r — using manifest path %s",
                    raw_path, manifest.file_path,
                )
            path = manifest.file_path

        write_text = f"\n\n## {section}\n\n{content}" if section else content

        existed = Path(path).exists()
        guard   = pre_write_check(
            path=path, content=write_text,
            file_type=manifest.file_type, existed_before=existed,
        )
        if not guard.ok:
            error_hint = (
                f"Output rejected: {guard.error}. "
                f"Write file content as plain text — not wrapped in JSON."
            )
            logger.warning(
                "subtask_executor: pre-write check failed (item=%r attempt=%d): %s",
                item.title, attempt, guard.error,
            )
            continue

        write_text = guard.content
        if not write_text.strip():
            logger.warning("subtask_executor: empty content for item %r attempt %d", item.title, attempt)
            error_hint = "Your output was empty. Write actual content."
            continue

        append = force_append or not is_first_item or attempt > 0

        # Snapshot before write so we can roll back if previously-complete items regress
        snapshot = _read_file(manifest.file_path)
        prev_complete = {it.anchor for it in manifest.items if it.status == "complete"}

        try:
            obs = _write_file(path, write_text, append=append)
        except PermissionError as _pe:
            error_hint = f"Path rejected: {_pe}. Use the exact path {manifest.file_path}"
            logger.warning("subtask_executor: write rejected (item=%r attempt=%d): %s", item.title, attempt, _pe)
            continue
        on_write(path)
        logger.info("subtask_executor: %s", obs)

        post = post_write_check(path, existed_before=existed)
        if post.errors:
            logger.warning("subtask_executor: post-write errors: %s", post.errors)

        file_content = _read_file(manifest.file_path)
        diff         = _verifier.run(manifest, file_content)

        # Roll back if previously-complete items regressed
        regressed_anchors = {it.anchor for it in diff.regressed} & prev_complete
        if regressed_anchors:
            logger.warning(
                "subtask_executor: regression on write (item=%r attempt=%d) — rolling back. regressed=%s",
                item.title[:40], attempt, sorted(regressed_anchors)[:3],
            )
            try:
                Path(path).write_text(snapshot, encoding="utf-8")
            except OSError as _re:
                logger.error("subtask_executor: rollback write failed: %s", _re)
            error_hint = (
                f"Your output caused {len(regressed_anchors)} previously-complete section(s) to disappear. "
                f"Append only — do not overwrite earlier content."
            )
            continue

        bad = {it.anchor for it in diff.partial + diff.missing + diff.regressed}
        if item.anchor not in bad:
            logger.info("[item] complete  item=%r  attempt=%d", item.title[:40], attempt)
            return True

        logger.info("[item] verify-fail  item=%r  attempt=%d  bad=%s", item.title[:40], attempt, sorted(bad)[:3])

    # Out of retries — mark partial, fail forward (forgiving)
    file_content = _read_file(manifest.file_path)
    if manifest.file_type == "doc":
        body = _verifier.parse_doc_sections(file_content).get(item.anchor, "")
    else:
        body = _verifier.parse_code_items(file_content).get(item.anchor, "")
    item.mark_partial(body)
    logger.warning("subtask_executor: item %r exhausted retries — marked partial", item.title)
    return False


# ---------------------------------------------------------------------------
# Stage runner — called from loop.py
# ---------------------------------------------------------------------------

def run_stage(
    llm_client:        Any,
    stage:             dict,
    file_path:         str,
    file_type:         Literal["doc", "code"],
    step:              int,
    max_steps:         int,
    store_manifest:    Callable[[SubtaskManifest], None],
    existing_manifest: Optional[SubtaskManifest] = None,
    on_item_start:     Optional[Callable[[str, str], None]] = None,
    on_item_done:      Optional[Callable[[str, bool, int], None]] = None,
    interrupt_event:   "threading.Event | None" = None,
) -> StageResult:
    """Drive write_doc / write_code items. Each item call reads from the file —
    the file is the memory, not the message history."""
    stage_goal = stage.get("goal", "")

    if existing_manifest is not None:
        manifest = existing_manifest
        force_append = True  # resuming mid-stage — file already has content, never overwrite
        logger.info("[stage] resume  file=%s  items=%d/%d  goal=%r", file_path, manifest.done_count, manifest.total, stage_goal[:50])
    else:
        existing_content = _read_file(file_path)
        force_append = False
        manifest = _planner.plan(llm_client, stage_goal, file_path, file_type, existing_content)
        store_manifest(manifest)
        logger.info("[stage] planned  file=%s  items=%d  type=%s  goal=%r", file_path, manifest.total, file_type, stage_goal[:50])

    last_written_path = file_path

    def _on_write(path: str) -> None:
        nonlocal last_written_path
        last_written_path = path
        store_manifest(manifest)

    import time as _t
    from birdclaw.agent.supervisor import StepSupervisor

    pending_items = [it for it in manifest.items if it.status != "complete"]
    sup = StepSupervisor(max_workers=2)

    try:
        # Submit first item immediately so LLM starts before the loop body runs
        if pending_items:
            sup.submit(
                _build_call_messages, pending_items[0], manifest, 0, step, max_steps, "",
                tag=pending_items[0].title,
            )

        for idx, item in enumerate(pending_items):
            if interrupt_event and interrupt_event.is_set():
                logger.info("subtask_executor: interrupted before item %r", item.title)
                break
            if on_item_start:
                on_item_start(item.title, file_path)
            t0 = _t.time()

            # Retrieve the pre-generated messages (or generate now if submit failed)
            call_messages = sup.collect()
            if call_messages is None:
                call_messages = _build_call_messages(item, manifest, 0, step, max_steps, "")

            # Submit NEXT item's context build to background slot immediately —
            # this runs while we write + verify the current item below.
            next_idx = idx + 1
            if next_idx < len(pending_items):
                next_item = pending_items[next_idx]
                sup.submit(
                    _build_call_messages, next_item, manifest, 0, step, max_steps, "",
                    tag=next_item.title,
                )

            # Write current item (uses pre-built messages on first attempt)
            ok = _write_item_with_messages(
                item, manifest, llm_client, step, max_steps, _on_write, call_messages,
                force_append=force_append,
            )

            if on_item_done:
                on_item_done(item.title, ok, int((_t.time() - t0) * 1000))
            store_manifest(manifest)
    finally:
        sup.shutdown()

    file_content  = _read_file(last_written_path)
    final_diff    = _verifier.run(manifest, file_content)
    store_manifest(manifest)

    reflect_hint = f"deepen: {final_diff.summary}" if final_diff.needs_resume else ""
    logger.info(
        "[stage] done  file=%s  complete=%d/%d  reflect=%r",
        last_written_path, manifest.done_count, manifest.total, reflect_hint[:60] or "none",
    )
    return StageResult(manifest, last_written_path, reflect_hint)
