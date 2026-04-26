from __future__ import annotations
import logging
import re
import time
from typing import Optional
from .subtask_manifest import SubtaskItem, SubtaskManifest, SubtaskDiff

logger = logging.getLogger(__name__)


_STUB_BODIES = {"pass", "...", "..", "raise NotImplementedError", "raise NotImplementedError()"}


# ---------------------------------------------------------------------------
# Doc parser
# ---------------------------------------------------------------------------

def parse_doc_sections(content: str) -> dict[str, str]:
    """Return {heading_text: body_text} splitting on ## or # headings."""
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []

    for line in content.splitlines(keepends=True):
        m = re.match(r"^#{1,2}\s+(.+)", line)
        if m:
            if current is not None:
                sections[current] = "".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            buf.append(line)

    if current is not None:
        sections[current] = "".join(buf).strip()

    return sections


# ---------------------------------------------------------------------------
# Code parser
# ---------------------------------------------------------------------------

def parse_code_items(content: str) -> dict[str, str]:
    """Return {name: body} for top-level def/class in Python source."""
    items: dict[str, str] = {}
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(def |class )(\w+)", lines[i])
        if m:
            name = m.group(2)
            body_lines: list[str] = [lines[i]]
            i += 1
            while i < len(lines):
                if lines[i] and not lines[i][0].isspace() and re.match(r"^(def |class |\S)", lines[i]):
                    break
                body_lines.append(lines[i])
                i += 1
            items[name] = "\n".join(body_lines).rstrip()
        else:
            i += 1
    return items


# ---------------------------------------------------------------------------
# Stub detection
# ---------------------------------------------------------------------------

def is_stub_body(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return True
    meaningful = [
        l.strip() for l in stripped.splitlines()
        if l.strip()
        and not l.strip().startswith("#")
        and not l.strip().startswith('"""')
        and not l.strip().startswith("'''")
        and not re.match(r'^(def |class )', l)
    ]
    if not meaningful:
        return True
    return all(l in _STUB_BODIES for l in meaningful)


# ---------------------------------------------------------------------------
# Key matching
# ---------------------------------------------------------------------------

def _match_key(anchor: str, parsed: dict[str, str]) -> Optional[str]:
    """Fuzzy match anchor against parsed keys."""
    anchor_clean = anchor.strip().lower()
    for key in parsed:
        if key.strip().lower() == anchor_clean:
            return key
        if anchor_clean in key.strip().lower() or key.strip().lower() in anchor_clean:
            return key
    return None


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

def run(manifest: SubtaskManifest, file_content: str) -> SubtaskDiff:
    """Compare manifest items against actual file content and return a SubtaskDiff."""
    manifest.last_verified_at = time.time()

    if manifest.file_type == "doc":
        parsed = parse_doc_sections(file_content)
    else:
        parsed = parse_code_items(file_content)

    logger.debug("[verifier] start  file=%s  type=%s  items=%d  parsed_keys=%d",
                 manifest.file_path, manifest.file_type, manifest.total, len(parsed))
    diff = SubtaskDiff()

    for item in manifest.items:
        key = _match_key(item.anchor, parsed)

        if key is None:
            item.status = "missing"
            diff.missing.append(item)
            continue

        body = parsed[key]
        current_chars = len(body)
        stub = manifest.file_type == "code" and is_stub_body(body)

        if stub or current_chars < item.expected_min_chars:
            prev_complete = item.status == "complete"
            prev_chars = item.actual_chars
            item.is_stub = stub
            item.mark_partial(body)
            if prev_complete and prev_chars > 0 and current_chars < prev_chars * 0.8:
                item.status = "regressed"
                diff.regressed.append(item)
            else:
                diff.partial.append(item)
            continue

        # Check regression: was complete before and now shorter
        prev_chars = item.actual_chars
        prev_hash = item.content_hash
        item.mark_complete(body)

        if prev_hash and prev_chars > 0 and current_chars < prev_chars * 0.8:
            item.status = "regressed"
            diff.regressed.append(item)
        else:
            diff.complete.append(item)

    # seam = first non-complete item
    diff.seam_index = next(
        (it.index for it in manifest.items if it.status != "complete"),
        len(manifest.items),
    )
    manifest.update_file_hash(file_content)
    diff.resume_context = _build_resume_context(manifest, diff, file_content)
    logger.info(
        "[verifier] done  file=%s  complete=%d  partial=%d  missing=%d  regressed=%d",
        manifest.file_path, len(diff.complete), len(diff.partial), len(diff.missing), len(diff.regressed),
    )
    return diff


# ---------------------------------------------------------------------------
# Resume context
# ---------------------------------------------------------------------------

def _build_resume_context(manifest: SubtaskManifest, diff: SubtaskDiff, file_content: str) -> str:
    bad_anchors = {it.anchor for it in diff.partial + diff.missing + diff.regressed}
    lines = [
        f"File: {manifest.file_path}",
    ]
    for it in manifest.items:
        if it.status == "complete":
            lines.append(f"  ✓ [{it.index}] {it.title}  ({it.actual_chars}c) — {it.summary[:60]}")
        elif it.status == "partial":
            lines.append(f"  ~ [{it.index}] {it.title}  — header only, {it.actual_chars}c (needs body, min {it.expected_min_chars}c)   ← seam")
        elif it.status in ("missing", "regressed"):
            marker = "⚠ REGRESSED" if it.status == "regressed" else "✗"
            lines.append(f"  {marker} [{it.index}] {it.title}  — {it.status}")
        else:
            lines.append(f"  · [{it.index}] {it.title}  — pending")

    lines.append("")
    if diff.seam_index < len(manifest.items):
        seam_item = manifest.items[diff.seam_index]
        lines.append(f"Resume at [{diff.seam_index}]. Anchor \"{seam_item.anchor}\" already in file (or must be created).")
        lines.append(f"Append body content only. Do NOT rewrite or repeat earlier sections.")
        lines.append(f"Min {seam_item.expected_min_chars} chars of substantive content.")

    if diff.regressed:
        for it in diff.regressed:
            lines.append(f"⚠ {it.title} regressed — the file may have been overwritten. Check before appending.")

    if file_content:
        tail = file_content[-800:].strip()
        lines += ["", "--- current file tail (seam) ---", tail, "--- end seam ---"]

    return "\n".join(lines)
