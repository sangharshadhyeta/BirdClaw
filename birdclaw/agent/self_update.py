"""Self-update loop — BirdClaw improves its own source code.

Pattern: Karpathy's autoresearch loop adapted for agent harness evolution.
  hypothesis → backup source → generate patch → run tests → evaluate → keep or revert

Activated only when BC_SELF_MODIFY=1 (settings.self_modify = True).

Safety model:
  - Before any modification, a full snapshot of birdclaw/ is copied to
    ~/.birdclaw/self_update/backup_<timestamp>/.
  - If tests fail or any exception occurs, the backup is restored atomically.
  - No git commits, no branches — offline only, completely reversible.
  - Only files under birdclaw/ are eligible for modification.
  - Each successful self-generated code block is stored as a GraphRAG skill
    node so future sessions can reference the technique.

Entry point:
    from birdclaw.agent.self_update import run_self_update_cycle
    run_self_update_cycle()   # blocking; call from a background thread

The soul loop calls trigger_self_update() when idle and self_modify is enabled.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from birdclaw.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BIRDCLAW_SRC   = Path(__file__).resolve().parent.parent   # …/birdclaw/
_REPO_ROOT      = _BIRDCLAW_SRC.parent
_INSTALL_SCRIPT = _REPO_ROOT / "install.sh"
_TEST_CMD       = [
    "python", "-m", "pytest", "tests/test.py",
    "-x", "-q", "--tb=short", "--no-header", "-m", "not long_running",
]
_BACKUP_ROOT    = settings.data_dir / "self_update"
# Isolated workspace for the agent loop — keeps repo root clean.
# write_code/write_doc stage outputs land here; actual source edits use
# absolute paths derived from reading birdclaw/ directly.
_WORK_DIR       = settings.data_dir / "self_update" / "workspace"

# ---------------------------------------------------------------------------
# Constitution — immutable rules for what self-update may and may not do
# ---------------------------------------------------------------------------

from birdclaw.agent.soul_constitution import check_protected, patch_prompt_rules  # noqa: E402

# ---------------------------------------------------------------------------
# Path guard — self-update MUST NOT touch anything outside birdclaw/
# Also enforces the protected-file list from soul_constitution.
# ---------------------------------------------------------------------------

def _validate_path_in_birdclaw(path: str | Path) -> bool:
    """Return True only if path is inside birdclaw/ AND not constitutionally protected."""
    allowed, reason = check_protected(path, _BIRDCLAW_SRC)
    if not allowed:
        logger.warning("self-update: BLOCKED write to %s — %s", path, reason)
    return allowed


def _safe_write_file(path: str, content: str) -> str:
    """Write a file only if allowed by constitution. Returns status message."""
    allowed, reason = check_protected(path, _BIRDCLAW_SRC)
    if not allowed:
        return f"BLOCKED: {reason}"
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {p} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------

def _backup_dir() -> Path:
    ts = int(time.time())
    d = _BACKUP_ROOT / f"backup_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot(backup: Path) -> None:
    """Copy birdclaw/ source tree to backup directory."""
    dest = backup / "birdclaw"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_BIRDCLAW_SRC, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    logger.info("self-update: snapshot saved to %s", backup)


def _restore(backup: Path) -> None:
    """Replace birdclaw/ with the backup snapshot.

    The soul_constitution.py is never overwritten during restore — it is the
    immutable ruleset and must always reflect the latest version on disk.
    """
    src = backup / "birdclaw"
    if not src.exists():
        logger.error("self-update: backup not found at %s", src)
        return

    # Preserve the live constitution — it must never regress
    constitution_live = _BIRDCLAW_SRC / "agent" / "soul_constitution.py"
    constitution_text: str | None = None
    if constitution_live.exists():
        try:
            constitution_text = constitution_live.read_text(encoding="utf-8")
        except OSError:
            pass

    # In-place file-by-file swap: copy each backup file over the live tree using
    # atomic tmp+rename so no module ever disappears between rmtree and copytree.
    # Other threads can keep importing safely throughout the restore.
    restored = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = _BIRDCLAW_SRC / rel
        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_file.with_suffix(dst_file.suffix + ".restoring")
            shutil.copy2(src_file, tmp)
            tmp.replace(dst_file)
            restored += 1
        except OSError as e:
            logger.warning("self-update: restore copy failed for %s: %s", rel, e)

    # Remove live files that don't exist in the backup (files added by the bad patch)
    for live_file in _BIRDCLAW_SRC.rglob("*"):
        if not live_file.is_file():
            continue
        rel = live_file.relative_to(_BIRDCLAW_SRC)
        if not (src / rel).exists():
            try:
                live_file.unlink(missing_ok=True)
            except OSError:
                pass

    logger.info("self-update: in-place restore complete — %d files written", restored)

    # Re-apply live constitution so a buggy patch can't revert our safety rules
    if constitution_text:
        try:
            constitution_live.write_text(constitution_text, encoding="utf-8")
            logger.debug("self-update: constitution preserved through restore")
        except OSError as e:
            logger.warning("self-update: could not re-apply constitution: %s", e)

    logger.info("self-update: restored from %s", backup)


def _hot_reload() -> None:
    """Sync patched source to the install directory and restart services.

    Calls `install.sh update --yes` which:
      - rsync's source → /opt/birdclaw/
      - runs pip install -e to pick up any new dependencies
      - restarts birdclaw-daemon.service (if running under systemd)

    Runs non-blocking in a background thread so the current process can
    write its completion record before the daemon restarts under it.
    """
    if not _INSTALL_SCRIPT.exists():
        logger.debug("self-update: install.sh not found at %s — skipping hot-reload", _INSTALL_SCRIPT)
        return

    import threading

    def _run() -> None:
        try:
            r = subprocess.run(
                ["bash", str(_INSTALL_SCRIPT), "update", "--yes"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                logger.info("self-update: hot-reload complete")
            else:
                logger.warning("self-update: hot-reload exited %d\n%s", r.returncode, r.stderr[-400:])
        except Exception as e:
            logger.warning("self-update: hot-reload failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="self-update-reload")
    t.start()
    logger.info("self-update: hot-reload started in background (install.sh update --yes)")


def _scrub_workspace() -> None:
    """Remove the isolated agent workspace and any __pycache__ dirs left by
    py_compile calls inside birdclaw/ after a self-update attempt."""
    # 1. Wipe the per-cycle write workspace
    if _WORK_DIR.exists():
        try:
            shutil.rmtree(_WORK_DIR)
            logger.debug("self-update: workspace scrubbed")
        except Exception as e:
            logger.debug("self-update: workspace scrub failed: %s", e)

    # 2. Remove __pycache__ dirs inside birdclaw/ source (py_compile artefacts)
    for cache_dir in _BIRDCLAW_SRC.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
        except Exception:
            pass


def _prune_old_backups(keep: int = 5) -> None:
    """Keep only the N most recent backups to avoid filling disk."""
    if not _BACKUP_ROOT.exists():
        return
    backups = sorted(
        [p for p in _BACKUP_ROOT.iterdir() if p.name.startswith("backup_")],
        key=lambda p: p.stat().st_mtime,
    )
    for old in backups[:-keep]:
        try:
            shutil.rmtree(old)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_tests(cwd: Path = _REPO_ROOT) -> tuple[bool, str]:
    """Run the regression suite. Returns (passed, summary_text)."""
    try:
        r = subprocess.run(
            _TEST_CMD, cwd=str(cwd), capture_output=True, text=True, timeout=300,
        )
        lines = (r.stdout + r.stderr).strip().splitlines()
        summary = "\n".join(lines[-10:]) if lines else "(no output)"
        return r.returncode == 0, summary
    except subprocess.TimeoutExpired:
        return False, "test suite timed out"
    except Exception as e:
        return False, str(e)


def _pass_rate(output: str) -> float:
    import re
    m = re.search(r"(\d+) passed", output)
    f = re.search(r"(\d+) failed", output)
    passed = int(m.group(1)) if m else 0
    failed = int(f.group(1)) if f else 0
    total  = passed + failed
    return passed / total if total else 0.0


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

def _score_pain_points() -> dict | None:
    """Derive the single highest-priority improvement target.

    Reads (in priority order):
      1. self_update_todo.jsonl  — explicit gaps logged by the agent during normal tasks
      2. stage_history.jsonl     — which stage types most often exhaust budget / get deepened
      3. birdclaw.log + failed tasks — concrete error signals

    Returns {"target": str, "evidence": str, "source": str} or None.
    """
    from birdclaw.config import settings as _s

    # 1. Explicit backlog — highest priority, agent observed these directly
    todo_path = _s.self_update_todo_path
    if todo_path.exists():
        try:
            import json as _json
            entries = []
            for line in todo_path.read_text(encoding="utf-8").splitlines():
                try:
                    e = _json.loads(line)
                    if not e.get("done"):
                        entries.append(e)
                except Exception:
                    pass
            if entries:
                # Sort by priority (high > normal > low), then recency
                _prank = {"high": 0, "normal": 1, "low": 2}
                entries.sort(key=lambda e: (_prank.get(e.get("priority", "normal"), 1), -e.get("ts", 0)))
                top = entries[0]
                logger.info("self-update: using todo item (priority=%s): %s", top.get("priority"), top["description"][:80])
                return {
                    "target": top["description"],
                    "evidence": f"Agent noted this during a task (priority={top.get('priority','normal')})",
                    "source": "todo",
                    "_todo_ts": top.get("ts"),
                }
        except Exception as e:
            logger.debug("self-update: todo read failed: %s", e)

    # 2. stage_history.jsonl — rank stage types by struggle (budget exhaustion + deepen)
    history_path = _s.data_dir / "stage_history.jsonl"
    if history_path.exists():
        try:
            import json as _json, collections
            struggle: dict[str, dict] = collections.defaultdict(lambda: {"exhausted": 0, "deepened": 0, "total": 0})
            for line in history_path.read_text(encoding="utf-8").splitlines()[-500:]:
                try:
                    r = _json.loads(line)
                    st = r.get("stage_type", "unknown")
                    struggle[st]["total"] += 1
                    if r.get("budget_exhausted"):
                        struggle[st]["exhausted"] += 1
                    if r.get("deepened"):
                        struggle[st]["deepened"] += 1
                except Exception:
                    pass
            if struggle:
                # Score = exhausted*2 + deepened, divided by total (rate)
                scored = sorted(
                    [(st, (v["exhausted"] * 2 + v["deepened"]) / max(v["total"], 1), v)
                     for st, v in struggle.items() if v["total"] >= 3],
                    key=lambda x: x[1], reverse=True,
                )
                if scored:
                    st, rate, counts = scored[0]
                    if rate > 0.2:  # >20% struggle rate — worth fixing
                        target = (
                            f"The '{st}' stage type has a high struggle rate "
                            f"({counts['exhausted']} budget exhaustions, {counts['deepened']} deepens "
                            f"out of {counts['total']} runs). "
                            f"Read birdclaw/agent/loop.py and birdclaw/agent/planner.py to understand "
                            f"why '{st}' stages struggle and improve the prompts, tool selection, "
                            f"or budget defaults for that stage type."
                        )
                        logger.info("self-update: stage '%s' struggle rate=%.2f — targeting", st, rate)
                        return {"target": target, "evidence": str(counts), "source": "stage_history"}
        except Exception as e:
            logger.debug("self-update: stage_history scan failed: %s", e)

    # 3. Error signals from logs / failed tasks (fallback)
    signals = _collect_error_signals(max_log_lines=30)
    if signals:
        return {
            "target": (
                "Recent errors were detected in the system logs and failed tasks. "
                "Read birdclaw/agent/loop.py, birdclaw/agent/planner.py, and birdclaw/agent/soul_loop.py "
                "to identify the root cause of these errors and fix them."
            ),
            "evidence": signals[:600],
            "source": "error_signals",
        }

    return None


def _mark_todo_done(ts: float) -> None:
    """Mark a todo item as done by timestamp so it isn't re-selected."""
    from birdclaw.config import settings as _s
    import json as _json
    todo_path = _s.self_update_todo_path
    if not todo_path.exists():
        return
    try:
        lines = todo_path.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            try:
                e = _json.loads(line)
                if abs(e.get("ts", 0) - ts) < 1.0:
                    e["done"] = True
                updated.append(_json.dumps(e))
            except Exception:
                updated.append(line)
        todo_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    except Exception:
        pass


def _collect_error_signals(max_log_lines: int = 60) -> str:
    """Collect recent error signals from logs, failed tasks, and session logs."""
    signals: list[str] = []

    # 1. birdclaw.log — grep for errors and key failure markers
    log_path = settings.data_dir / "birdclaw.log"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            error_lines = [
                l for l in lines
                if any(kw in l for kw in (
                    "ERROR", "exit_code:127", "force-completed", "stall guard",
                    "NameError", "AttributeError", "KeyError", "Traceback",
                    "failed:", "FAILED", "exception",
                ))
            ]
            if error_lines:
                signals.append("=== Recent log errors ===\n" + "\n".join(error_lines[-max_log_lines:]))
        except Exception:
            pass

    # 2. Failed tasks — titles and error fields
    try:
        from birdclaw.memory.task_registry import task_registry
        failed = task_registry.list(status="failed")
        if failed:
            lines = []
            for t in failed[-10:]:
                err = getattr(t, "error", "") or ""
                lines.append(f"- {t.prompt[:80]}  error={err[:120]}")
            signals.append("=== Recent failed tasks ===\n" + "\n".join(lines))
    except Exception:
        pass

    # 3. Session logs — look for lines containing user corrections and force-completions
    try:
        sessions_dir = settings.data_dir / "sessions"
        if sessions_dir.exists():
            session_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            correction_lines: list[str] = []
            for sf in session_files[:5]:
                try:
                    import json as _json
                    for raw in sf.read_text(encoding="utf-8", errors="replace").splitlines():
                        try:
                            ev = _json.loads(raw)
                            role = ev.get("role", "")
                            content = ev.get("content", "")
                            if role == "user" and any(kw in content.lower() for kw in (
                                "no,", "wrong", "don't", "stop", "that's not", "fix", "incorrect"
                            )):
                                correction_lines.append(f"[{sf.stem}] user: {content[:120]}")
                        except Exception:
                            pass
                except Exception:
                    pass
            if correction_lines:
                signals.append("=== Recent user corrections ===\n" + "\n".join(correction_lines[-20:]))
    except Exception:
        pass

    return "\n\n".join(signals)


def _generate_hypothesis() -> dict | None:
    """Ask the model to identify one improvement opportunity in birdclaw/.

    Pulls concrete error signals from: birdclaw.log, failed tasks, session logs.
    Returns {"hypothesis": str} or None.
    """
    from birdclaw.agent.loop import run_agent_loop

    error_signals = _collect_error_signals()
    error_section = (
        f"\n\nRecent error signals from the running system:\n{error_signals}\n"
        if error_signals else ""
    )

    prompt = (
        "You are improving BirdClaw's own source code. "
        "First run the regression tests with bash to see what fails. "
        "Then read the relevant source file(s) in birdclaw/. "
        f"{error_section}"
        "Identify ONE specific, small, safe improvement: a bug fix, better prompt wording, "
        "improved tool routing logic, or a missing guard. "
        "Prefer fixing issues that appear in the error signals above — they are real problems "
        "from the running system. "
        "Output: what file to change, what the specific problem is, and what the fix should be."
    )
    try:
        result = run_agent_loop(prompt)
        if result.answer and len(result.answer) > 30:
            return {"hypothesis": result.answer}
        return None
    except Exception as e:
        logger.warning("self-update: hypothesis generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Skill ingestion — store generated technique for future reference
# ---------------------------------------------------------------------------

def _ingest_as_skill(hypothesis: str, patch_description: str) -> None:
    """Store the self-generated improvement in GraphRAG as a skill node."""
    try:
        from birdclaw.memory.graph import knowledge_graph
        node_name = f"self_update:{int(time.time())}"
        knowledge_graph.upsert_node(
            name=node_name,
            node_type="skill",
            summary=(
                f"Self-update technique: {hypothesis[:120]} | "
                f"result: {patch_description[:120]}"
            ),
            sources=["self_update_loop"],
        )
        knowledge_graph.save()
        logger.info("self-update: technique ingested as skill node %r", node_name)
    except Exception as e:
        logger.debug("self-update: skill ingestion failed: %s", e)


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_self_update_cycle(max_attempts: int = 3) -> dict:
    """Run one self-update cycle. Returns a result dict:
        success:     bool
        improvement: float  (pass-rate delta, positive = better)
        summary:     str

    The cycle is:
      1. Python scores pain points (todo backlog > stage struggle > error signals)
      2. Agent runs the normal loop against _REPO_ROOT as workspace — reads source,
         uses edit_file / write_code / verify stages exactly as for any other task
      3. pytest gate — accept only if no regression; otherwise restore from backup
    """
    if not settings.self_modify:
        return {"success": False, "summary": "BC_SELF_MODIFY not enabled", "improvement": 0.0}

    logger.info("self-update: starting cycle (max %d attempts)", max_attempts)
    _prune_old_backups()

    # Measure baseline pass rate before any modification
    _, baseline_out = _run_tests()
    baseline_rate = _pass_rate(baseline_out)
    logger.info("self-update: baseline pass rate = %.2f", baseline_rate)

    for attempt in range(1, max_attempts + 1):
        logger.info("self-update: attempt %d/%d", attempt, max_attempts)

        # 1. Python: score pain points and derive the task prompt
        pain_point = _score_pain_points()
        if not pain_point:
            logger.warning("self-update: no pain point found — skipping attempt %d", attempt)
            continue

        target = pain_point["target"]
        evidence = pain_point.get("evidence", "")
        source = pain_point.get("source", "unknown")
        logger.info("self-update: target (source=%s) — %.120s", source, target)

        # 2. Snapshot before modification
        backup = _backup_dir()
        try:
            _snapshot(backup)
        except Exception as e:
            logger.error("self-update: snapshot failed: %s — aborting", e)
            continue

        try:
            # 3. Single agent loop call — reads source, makes changes, verifies syntax.
            #    write_dir=_WORK_DIR keeps any new files inside ~/.birdclaw/self_update/workspace/
            #    so the repo root and birdclaw/ tree stay clean.
            #    Edits to existing source files use absolute paths regardless of write_dir.
            #    BC_SELF_MODIFY=1 is already set (we checked above), so permission.py
            #    allows writes to birdclaw/ for this process.
            _WORK_DIR.mkdir(parents=True, exist_ok=True)

            task_prompt = (
                patch_prompt_rules(_BIRDCLAW_SRC)
                + f"\n\nImprovement target:\n{target}"
                + (f"\n\nEvidence:\n{evidence[:400]}" if evidence else "")
                + "\n\nInstructions:\n"
                f"- Source is at {_BIRDCLAW_SRC}/ — read files there first to understand the code.\n"
                "- Make the smallest change that fixes the problem — use edit_file for targeted changes, "
                "write_code stage for full rewrites (same as writing a skill file).\n"
                "- After each file change, verify syntax: bash python -m py_compile <file>\n"
            )

            from birdclaw.agent.loop import run_agent_loop
            patch_result = run_agent_loop(
                task_prompt,
                write_dir=str(_WORK_DIR),
            )
            patch_desc = (patch_result.answer or "no description")[:200]
            logger.info("self-update: agent done — %.80s", patch_desc)

            # 4. Run tests on the patched code
            passed, test_out = _run_tests()
            new_rate = _pass_rate(test_out)
            improvement = new_rate - baseline_rate
            logger.info(
                "self-update: tests %s  rate %.2f → %.2f (Δ %.2f)",
                "PASSED" if passed else "FAILED", baseline_rate, new_rate, improvement,
            )

            if not passed or improvement < 0:
                logger.info("self-update: reverting attempt %d (tests failed or regressed)", attempt)
                _restore(backup)
                continue

            # 5. Improvement confirmed — keep the patch, ingest as skill, mark todo done
            logger.info("self-update: improvement accepted (Δ %.2f)", improvement)
            _ingest_as_skill(target, patch_desc)

            if source == "todo" and pain_point.get("_todo_ts"):
                _mark_todo_done(pain_point["_todo_ts"])

            # Hot-reload: sync patched source to install dir and restart services
            _hot_reload()

            import json, time as _t
            record_path = _BACKUP_ROOT / "history.jsonl"
            with open(record_path, "a") as f:
                f.write(json.dumps({
                    "ts": _t.time(), "target": target[:200], "source": source,
                    "patch": patch_desc[:200], "baseline": baseline_rate,
                    "new_rate": new_rate, "improvement": improvement,
                    "backup": str(backup),
                }) + "\n")

            return {
                "success": True,
                "improvement": improvement,
                "summary": f"Applied ({source}): {target[:80]}",
            }

        except Exception as e:
            logger.exception("self-update: attempt %d failed: %s", attempt, e)
            try:
                _restore(backup)
            except Exception:
                pass
        finally:
            _scrub_workspace()

    return {
        "success": False,
        "summary": f"No improvement found after {max_attempts} attempts",
        "improvement": 0.0,
    }


# ---------------------------------------------------------------------------
# Test-failure triggered self-update
# ---------------------------------------------------------------------------

def run_self_update_for_failure(
    test_name: str,
    failure_reason: str,
    task_prompt: str = "",
) -> dict:
    """Trigger a targeted self-update cycle in response to a specific test failure.

    Called by test_long_running.py when a task fails process validation.
    Focuses the hypothesis generation on the specific failure rather than
    running the full regression suite first.

    Args:
        test_name:      Name of the failing test (e.g. "Financial Audit Report")
        failure_reason: What went wrong (e.g. "planned only 1 phase")
        task_prompt:    The prompt that failed (for context)

    Returns same dict as run_self_update_cycle().
    """
    if not settings.self_modify:
        return {"success": False, "summary": "BC_SELF_MODIFY not enabled", "improvement": 0.0}

    logger.info("self-update: triggered by test failure: %s — %s", test_name, failure_reason[:80])
    _prune_old_backups()

    backup = _backup_dir()
    try:
        _snapshot(backup)
    except Exception as e:
        return {"success": False, "summary": f"snapshot failed: {e}", "improvement": 0.0}

    # Focused hypothesis: fix the specific failure
    from birdclaw.agent.loop import run_agent_loop

    hypothesis_prompt = (
        f"A BirdClaw long-running task just failed validation.\n\n"
        f"Task: {test_name}\n"
        f"Failure reason: {failure_reason}\n"
        f"Task prompt: {task_prompt[:200]}\n\n"
        "Read the relevant birdclaw/ source files (agent/loop.py, agent/prompts.py, etc.) "
        "and identify what specifically caused this failure. "
        "Output: the exact file to change, what the problem is, and the fix."
    )

    try:
        hyp_result = run_agent_loop(hypothesis_prompt)
        hypothesis = hyp_result.answer or ""
        if len(hypothesis) < 20:
            logger.warning("self-update: hypothesis too short for %s — skipping", test_name)
            return {"success": False, "summary": "no useful hypothesis", "improvement": 0.0}

        patch_prompt = (
            patch_prompt_rules(_BIRDCLAW_SRC)
            + f"Fix this specific BirdClaw failure:\n\n"
            f"Task: {test_name}\n"
            f"Failure: {failure_reason}\n"
            f"Diagnosis: {hypothesis}\n\n"
            "Make the minimal change that fixes this specific failure. "
            "After writing, verify syntax with bash."
        )

        patch_result = run_agent_loop(patch_prompt)
        patch_desc = (patch_result.answer or "no description")[:200]
        logger.info("self-update: patch for %s — %.80s", test_name, patch_desc)

        # Run regression suite to check no regressions introduced
        passed, test_out = _run_tests()
        new_rate = _pass_rate(test_out)

        if not passed:
            logger.info("self-update: regression tests failed after patch for %s — reverting", test_name)
            _restore(backup)
            return {"success": False, "summary": f"patch introduced regressions: {test_out[-200:]}", "improvement": 0.0}

        # Accept the patch
        _ingest_as_skill(
            f"fix for {test_name}: {failure_reason[:80]}",
            patch_desc,
        )

        import json as _json
        record_path = _BACKUP_ROOT / "history.jsonl"
        import time as _t
        with open(record_path, "a") as f:
            f.write(_json.dumps({
                "ts": _t.time(), "trigger": "test_failure",
                "test_name": test_name, "failure": failure_reason,
                "patch": patch_desc[:200], "backup": str(backup),
            }) + "\n")

        logger.info("self-update: fix accepted for %s", test_name)
        return {
            "success": True,
            "improvement": new_rate,
            "summary": f"Fixed {test_name}: {patch_desc[:80]}",
        }

    except Exception as e:
        logger.exception("self-update: error fixing %s: %s", test_name, e)
        try:
            _restore(backup)
        except Exception:
            pass
        return {"success": False, "summary": str(e), "improvement": 0.0}


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

_running = False


def trigger_self_update() -> None:
    """Launch a self-update cycle in a background thread (non-blocking)."""
    global _running
    if _running:
        logger.debug("self-update: already running — skipping trigger")
        return
    if not settings.self_modify:
        return

    import threading

    def _bg() -> None:
        global _running
        _running = True
        try:
            result = run_self_update_cycle()
            logger.info("self-update cycle result: %s", result)
        finally:
            _running = False

    threading.Thread(target=_bg, daemon=True, name="self-update").start()
    logger.info("self-update: background cycle triggered")
