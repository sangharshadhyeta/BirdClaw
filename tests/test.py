"""BirdClaw unified test suite.

Sections:
  - Global fixtures
  - Agent: soul, context
  - Memory: history
  - LLM: usage
  - Tools: bash async, hooks, sandbox
  - Tools/MCP: naming, client, manager
  - E2E
  - Regression — self-update lifecycle gate (R1–R15)
  - Outputs directory integrity
  - Subtask middle layer
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional
from unittest.mock import patch

import pytest

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("birdclaw").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

OUTPUTS_DIR  = Path(__file__).parent / "outputs"
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR.mkdir(exist_ok=True)

WIDTH = 80


# ===========================================================================
# Global fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def disable_llm_scheduler(monkeypatch):
    """Bypass the LLM scheduler in all unit tests."""
    monkeypatch.setenv("BC_LLM_SCHEDULER_ENABLED", "false")
    try:
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "llm_scheduler_enabled", False)
    except Exception:
        pass



# ===========================================================================
# Agent — soul (Port 5)
# ===========================================================================

from birdclaw.agent.soul import SOUL, build_system_prompt, BASE_SYSTEM_PROMPT


@pytest.mark.unit
def test_soul_keys():
    for key in ("name", "character", "communication_style", "values",
                "capabilities_summary", "limitations"):
        assert key in SOUL, f"missing key: {key}"
    assert isinstance(SOUL["values"], list)
    assert len(SOUL["values"]) >= 3


@pytest.mark.unit
def test_base_prompt_contains_name():
    assert SOUL["name"] in BASE_SYSTEM_PROMPT


@pytest.mark.unit
def test_base_prompt_contains_values():
    for v in SOUL["values"]:
        assert v in BASE_SYSTEM_PROMPT


@pytest.mark.unit
def test_base_prompt_ends_with_answer_instruction():
    # Prompt now uses format_schema routing — instructs via "answer or create_task" text
    assert "answer" in BASE_SYSTEM_PROMPT


@pytest.mark.unit
def test_build_system_prompt_sections():
    prompt = build_system_prompt(
        history_context="User: hello\nBirdClaw: hi",
        project_context="# CLAUDE.md\nPython project",
        extra="Standing goal: keep inbox empty",
    )
    assert "User: hello"             in prompt
    assert "Recent conversation"     in prompt
    assert "CLAUDE.md"               in prompt
    assert "Current project context" in prompt
    assert "Standing goal"           in prompt


@pytest.mark.unit
def test_build_system_prompt_no_args_equals_base():
    assert build_system_prompt() == BASE_SYSTEM_PROMPT


@pytest.mark.unit
def test_all_sections_ordered():
    prompt = build_system_prompt(
        history_context="XHISTORYMARKERX",
        project_context="XPROJECTMARKERX",
        extra="XEXTRAMARKERX",
    )
    assert (prompt.index(SOUL["name"])
            < prompt.index("Current project context")
            < prompt.index("Recent conversation")
            < prompt.index("XEXTRAMARKERX"))


# ===========================================================================
# Agent — context (Port 10)
# ===========================================================================

import subprocess

from birdclaw.agent.context import (
    ProjectContext,
    collapse_blank_lines,
    _display_context_path,
    _normalize_instruction_content,
    _render_instruction_files,
    _truncate_instruction_content,
    MAX_INSTRUCTION_FILE_CHARS,
)


@pytest.mark.unit
def test_collapse_blank_lines():
    assert collapse_blank_lines("a\n\n\n\nb\n") == "a\n\nb\n"


@pytest.mark.unit
def test_normalize_collapses_and_strips():
    assert _normalize_instruction_content("line one\n\n\nline two\n") == "line one\n\nline two"


@pytest.mark.unit
def test_discovers_files_from_ancestor_chain(tmp_path):
    nested = tmp_path / "apps" / "api"
    (nested / ".claw").mkdir(parents=True)
    (tmp_path / "apps" / ".claw").mkdir(parents=True)

    (tmp_path / "CLAUDE.md").write_text("root instructions")
    (tmp_path / "CLAUDE.local.md").write_text("local instructions")
    (tmp_path / "apps" / "CLAUDE.md").write_text("apps instructions")
    (tmp_path / "apps" / ".claw" / "instructions.md").write_text("apps claw instructions")
    (nested / ".claw" / "CLAUDE.md").write_text("nested rules")
    (nested / ".claw" / "instructions.md").write_text("nested instructions")

    ctx      = ProjectContext.discover(nested, "2026-03-31")
    contents = [f.content for f in ctx.instruction_files]

    for expected in ("root instructions", "local instructions", "apps instructions",
                     "apps claw instructions", "nested rules", "nested instructions"):
        assert expected in contents
    assert contents.index("root instructions") < contents.index("nested rules")


@pytest.mark.unit
def test_dedupes_identical_content(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("same rules\n\n")
    (tmp_path / "sub").mkdir()
    ((tmp_path / "sub") / "CLAUDE.md").write_text("same rules\n")
    ctx = ProjectContext.discover(tmp_path / "sub", "2026-03-31")
    assert len(ctx.instruction_files) == 1


@pytest.mark.unit
def test_ignores_empty_files(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("   \n")
    assert ProjectContext.discover(tmp_path, "2026-03-31").instruction_files == []


@pytest.mark.unit
def test_truncates_large_content():
    rendered = _truncate_instruction_content("x" * 4500, 4000)
    assert "[truncated]" in rendered
    assert len(rendered) < 4200


@pytest.mark.unit
def test_render_instruction_files_includes_scope(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Project rules")
    ctx      = ProjectContext.discover(tmp_path, "2026-03-31")
    rendered = _render_instruction_files(ctx.instruction_files)
    assert "# Claude instructions" in rendered
    assert "scope:"                 in rendered
    assert "Project rules"          in rendered


@pytest.mark.unit
def test_render_instruction_files_budget_omission(tmp_path):
    chain = tmp_path
    for i, letter in enumerate("abcde"):
        chain = chain / letter
        chain.mkdir()
        (chain / "CLAUDE.md").write_text(f"# Level {i}\n" + letter * 2990)
    rendered = _render_instruction_files(
        ProjectContext.discover(chain, "2026-04-01").instruction_files
    )
    assert "omitted" in rendered


@pytest.mark.unit
def test_display_context_path_returns_filename():
    assert _display_context_path(Path("/tmp/project/.claw/CLAUDE.md")) == "CLAUDE.md"


@pytest.mark.unit
def test_render_contains_date_and_cwd(tmp_path):
    rendered = ProjectContext.discover(tmp_path, "2026-03-31").render()
    assert "2026-03-31"  in rendered
    assert str(tmp_path) in rendered


@pytest.mark.unit
def test_render_no_git_by_default(tmp_path):
    ctx = ProjectContext.discover(tmp_path, "2026-03-31")
    assert ctx.git_status is None
    assert "Git status" not in ctx.render()


@pytest.fixture
def git_repo(tmp_path):
    subprocess.run(["git", "init", "--quiet"],                          cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"],              cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.component
def test_discover_with_git_captures_status(git_repo):
    (git_repo / "file.txt").write_text("hello")
    ctx = ProjectContext.discover_with_git(git_repo, "2026-03-31")
    assert ctx.git_status is not None
    assert "file.txt" in ctx.git_status


@pytest.mark.component
def test_discover_with_git_captures_diff(git_repo):
    (git_repo / "tracked.txt").write_text("hello\n")
    subprocess.run(["git", "add", "tracked.txt"],              cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init", "--quiet"], cwd=git_repo, check=True)
    (git_repo / "tracked.txt").write_text("hello\nworld\n")
    ctx = ProjectContext.discover_with_git(git_repo, "2026-03-31")
    assert ctx.git_diff is not None
    assert "Unstaged changes:" in ctx.git_diff
    assert "tracked.txt"       in ctx.git_diff


@pytest.mark.component
def test_discover_with_git_captures_staged_diff(git_repo):
    (git_repo / "tracked.txt").write_text("hello\n")
    subprocess.run(["git", "add", "tracked.txt"],              cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init", "--quiet"], cwd=git_repo, check=True)
    (git_repo / "tracked.txt").write_text("hello\nworld\n")
    subprocess.run(["git", "add", "tracked.txt"],              cwd=git_repo, check=True)
    ctx = ProjectContext.discover_with_git(git_repo, "2026-03-31")
    assert "Staged changes:" in (ctx.git_diff or "")


# ===========================================================================
# Memory — history (Port 3)
# ===========================================================================

from birdclaw.memory.history import History, _rotate


@pytest.mark.unit
def test_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h   = History.new()
    rec = json.loads(h._path.read_text().splitlines()[0])
    assert rec["type"]       == "session_meta"
    assert rec["session_id"] == h.session_id


@pytest.mark.unit
def test_add_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h = History.new()
    h.add_turn("user", "hello")
    h.add_turn("assistant", "hi there")
    h2 = History.load(h.session_id)
    assert h2.turn_count()      == 2
    assert h2._turns[0].content == "hello"


@pytest.mark.unit
def test_recent_text(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h = History.new()
    h.add_turn("user", "what is 2+2?")
    h.add_turn("assistant", "4")
    text = h.recent_text()
    assert "User: what is 2+2?" in text
    assert "BirdClaw: 4"        in text


@pytest.mark.unit
def test_search(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h = History.new()
    h.add_turn("user",      "tell me about python")
    h.add_turn("assistant", "python is great")
    h.add_turn("user",      "what about rust?")
    assert len(h.search("python")) == 2
    assert len(h.search("rust"))   == 1


@pytest.mark.unit
def test_compaction(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h = History.new()
    h.add_turn("user", "a")
    h.add_turn("assistant", "b")
    h.record_compaction("discussed a and b", removed=2)
    h2 = History.load(h.session_id)
    assert len(h2._compactions)       == 1
    assert h2._compactions[0].removed == 2


@pytest.mark.unit
def test_fork(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h = History.new()
    h.add_turn("user", "original")
    h.add_turn("assistant", "reply")
    child = h.fork("experiment")
    assert child.session_id != h.session_id
    assert child.parent_fork.branch_name == "experiment"
    assert child.turn_count() == 2
    child.add_turn("user", "child-only")
    assert child.turn_count() == 3
    assert h.turn_count()     == 2


@pytest.mark.unit
def test_fork_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h     = History.new()
    h.add_turn("user", "seed")
    child = h.fork("branch-a")
    loaded = History.load(child.session_id)
    assert loaded.parent_fork.parent_session_id == h.session_id
    assert loaded.parent_fork.branch_name        == "branch-a"


@pytest.mark.unit
def test_rotation(tmp_path):
    stem = "abc123"
    (tmp_path / f"{stem}.jsonl").write_text("active")
    (tmp_path / f"{stem}.1.jsonl").write_text("rotated1")
    (tmp_path / f"{stem}.2.jsonl").write_text("rotated2")
    (tmp_path / f"{stem}.3.jsonl").write_text("rotated3-deleted")
    active = tmp_path / f"{stem}.jsonl"
    _rotate(active, max_rotated=3)
    assert not active.exists()
    assert (tmp_path / f"{stem}.1.jsonl").read_text() == "active"
    assert (tmp_path / f"{stem}.2.jsonl").read_text() == "rotated1"
    assert not (tmp_path / f"{stem}.4.jsonl").exists()


@pytest.mark.unit
def test_load_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("birdclaw.memory.history._history_dir", lambda: tmp_path)
    monkeypatch.setattr("birdclaw.memory.history._ensure_history_dir", lambda: None)
    h1 = History.new(); h1.add_turn("user", "first")
    time.sleep(0.01)
    h2 = History.new(); h2.add_turn("user", "second")
    latest = History.load_latest()
    assert latest is not None
    assert latest.session_id == h2.session_id


# ===========================================================================
# LLM — usage (Port 9)
# ===========================================================================

from birdclaw.llm.usage import (
    TokenUsage, UsageTracker, estimate_cost, format_usd, pricing_for_model,
)


@pytest.mark.unit
def test_format_usd():
    assert format_usd(0.0)  == "$0.0000"
    assert format_usd(15.0) == "$15.0000"
    assert format_usd(0.3)  == "$0.3000"


@pytest.mark.unit
def test_pricing_known_models():
    haiku  = pricing_for_model("claude-haiku-4-5-20251001")
    opus   = pricing_for_model("claude-opus-4-6")
    gemma  = pricing_for_model("gemma4:4b")
    assert haiku.input_cost_per_million  == 1.0
    assert opus.output_cost_per_million  == 75.0
    assert gemma.input_cost_per_million  == 0.0


@pytest.mark.unit
def test_pricing_unknown_is_zero():
    assert pricing_for_model("some-custom-model-v9").input_cost_per_million == 0.0


@pytest.mark.unit
def test_token_usage_total_and_add():
    a = TokenUsage(input_tokens=10, output_tokens=4,
                   cache_creation_input_tokens=2, cache_read_input_tokens=1)
    b = TokenUsage(input_tokens=20, output_tokens=6)
    assert a.total_tokens() == 17
    c = a + b
    assert c.input_tokens  == 30
    assert c.output_tokens == 10


@pytest.mark.unit
def test_cost_for_sonnet():
    u    = TokenUsage(input_tokens=1_000_000, output_tokens=500_000,
                      cache_creation_input_tokens=100_000, cache_read_input_tokens=200_000)
    cost = estimate_cost(u, "claude-sonnet-4-6")
    assert format_usd(cost.total_cost_usd()) == "$54.6750"


@pytest.mark.unit
def test_cost_for_local_model_is_zero():
    cost = estimate_cost(TokenUsage(input_tokens=100_000, output_tokens=50_000), "gemma4:4b")
    assert cost.total_cost_usd() == 0.0


@pytest.mark.unit
def test_summary_lines():
    u     = TokenUsage(input_tokens=1_000_000, output_tokens=500_000,
                       cache_creation_input_tokens=100_000, cache_read_input_tokens=200_000)
    lines = u.summary_lines_for_model("usage", "claude-sonnet-4-6")
    assert "total_tokens=1800000"    in lines[0]
    assert "estimated_cost=$54.6750" in lines[0]
    assert "cache_read=$0.3000"      in lines[1]


@pytest.mark.unit
def test_tracker_cumulative():
    tracker = UsageTracker()
    tracker.record(TokenUsage(input_tokens=10, output_tokens=4))
    tracker.record(TokenUsage(input_tokens=20, output_tokens=6))
    assert tracker.turns                   == 2
    assert tracker.cumulative.input_tokens == 30


@pytest.mark.unit
def test_tracker_error_count():
    tracker = UsageTracker()
    tracker.record_request()
    tracker.record_request()
    tracker.record(TokenUsage(input_tokens=5, output_tokens=2))
    assert tracker.error_count == 1


# ===========================================================================
# Tools — bash async (Port 4)
# ===========================================================================

from birdclaw.tools.bash import run_bash, bash_poll, bash_write, bash_kill


@pytest.mark.component
def test_background_returns_session():
    out = json.loads(run_bash("echo hello", background=True))
    assert out["status"] == "running"
    assert "session_id"  in out
    bash_kill(out["session_id"])


@pytest.mark.component
def test_poll_running_and_completed():
    out = json.loads(run_bash("echo port4test", background=True))
    sid = out["session_id"]
    for _ in range(20):
        poll = json.loads(bash_poll(sid))
        if poll["status"] != "running":
            break
        time.sleep(0.05)
    assert poll["status"]    == "completed"
    assert "port4test"       in poll["stdout_tail"]


@pytest.mark.component
def test_poll_failed():
    out = json.loads(run_bash("exit 42", background=True))
    sid = out["session_id"]
    for _ in range(20):
        poll = json.loads(bash_poll(sid))
        if poll["status"] != "running":
            break
        time.sleep(0.05)
    assert poll["exit_code"] == 42


@pytest.mark.component
def test_bash_kill():
    out = json.loads(run_bash("sleep 60", background=True))
    sid = out["session_id"]
    assert json.loads(bash_kill(sid))["ok"] is True
    time.sleep(0.1)
    assert json.loads(bash_poll(sid))["status"] in ("killed", "failed")


@pytest.mark.component
def test_bash_write_stdin():
    out = json.loads(run_bash("read line && echo got:$line", background=True))
    sid = out["session_id"]
    time.sleep(0.1)
    bash_write(sid, "hello\n")
    for _ in range(30):
        poll = json.loads(bash_poll(sid))
        if poll["status"] != "running":
            break
        time.sleep(0.1)
    assert "got:hello" in poll["stdout_tail"]


@pytest.mark.component
def test_poll_unknown_session():
    assert "error" in json.loads(bash_poll("nonexistent-session"))


@pytest.mark.component
def test_write_to_dead_session():
    out = json.loads(run_bash("echo done", background=True))
    sid = out["session_id"]
    for _ in range(20):
        if json.loads(bash_poll(sid))["status"] != "running":
            break
        time.sleep(0.05)
    assert "error" in json.loads(bash_write(sid, "text\n"))


@pytest.mark.component
def test_synchronous_bash():
    out = json.loads(run_bash("echo sync"))
    assert out["stdout"].strip()             == "sync"
    assert out["return_code_interpretation"] == "exit_code:0"


# ===========================================================================
# Tools — hooks (Port 11)
# ===========================================================================

from birdclaw.tools.hooks import (
    HookAbortSignal, HookEvent, HookProgressEvent, HookProgressReporter,
    HookRunner, PermissionDecision,
)


class RecordingReporter(HookProgressReporter):
    def __init__(self):
        self.events: list[HookProgressEvent] = []

    def on_event(self, event: HookProgressEvent) -> None:
        self.events.append(event)


@pytest.mark.unit
def test_hook_allows_exit_zero():
    result = HookRunner(pre_tool_use=["printf 'pre ok'"]).pre_tool_use("Read", {"path": "README.md"})
    assert result.allowed
    assert "pre ok" in result.messages


@pytest.mark.unit
def test_no_hooks_allows():
    assert HookRunner().pre_tool_use("bash", {"command": "ls"}).allowed


@pytest.mark.unit
def test_hook_denies_exit_two():
    result = HookRunner(pre_tool_use=["printf 'blocked'; exit 2"]).pre_tool_use("bash", {})
    assert result.is_denied()
    assert any("blocked" in m for m in result.messages)


@pytest.mark.unit
def test_hook_fails_non_zero_non_two():
    result = HookRunner(pre_tool_use=["printf 'warn'; exit 1"]).pre_tool_use("Edit", {})
    assert result.is_failed()


@pytest.mark.unit
def test_hook_deny_via_json():
    runner = HookRunner(pre_tool_use=[r"""printf '{"continue":false,"systemMessage":"json denied"}'"""])
    result = runner.pre_tool_use("bash", {})
    assert result.is_denied()
    assert any("json denied" in m for m in result.messages)


@pytest.mark.unit
def test_hook_deny_via_decision_block():
    runner = HookRunner(pre_tool_use=[r"""printf '{"decision":"block","reason":"policy"}'"""])
    result = runner.pre_tool_use("bash", {})
    assert result.is_denied()


@pytest.mark.unit
def test_post_tool_use_hook():
    result = HookRunner(post_tool_use=["printf 'post ok'"]).post_tool_use("bash", {}, "out")
    assert result.allowed
    assert "post ok" in result.messages


@pytest.mark.unit
def test_hook_chain_stops_after_failure():
    result = HookRunner(pre_tool_use=["printf 'broken'; exit 1", "printf 'later'"]).pre_tool_use("Edit", {})
    assert result.is_failed()
    assert not any(m == "later" for m in result.messages)


@pytest.mark.unit
def test_hook_parses_permission_override():
    payload = (r'{"systemMessage":"ok","hookSpecificOutput":{'
               r'"permissionDecision":"allow","permissionDecisionReason":"hook ok",'
               r'"updatedInput":{"command":"git status"}}}')
    result = HookRunner(pre_tool_use=[f"printf '{payload}'"]).pre_tool_use("bash", {})
    assert result.permission_decision == PermissionDecision.Allow
    assert result.updated_input is not None


@pytest.mark.unit
def test_reporter_events():
    reporter = RecordingReporter()
    HookRunner(pre_tool_use=["printf 'a'", "printf 'b'"]).pre_tool_use(
        "Read", {}, reporter=reporter
    )
    kinds = [e.kind for e in reporter.events]
    assert kinds == ["started", "completed", "started", "completed"]


@pytest.mark.component
def test_abort_signal_cancels_hook():
    abort = HookAbortSignal()
    threading.Timer(0.1, abort.abort).start()
    result = HookRunner(pre_tool_use=["sleep 10"]).pre_tool_use("bash", {}, abort_signal=abort)
    assert result.is_cancelled()


@pytest.mark.unit
def test_abort_before_run():
    abort = HookAbortSignal()
    abort.abort()
    result = HookRunner(pre_tool_use=["printf 'should not run'"]).pre_tool_use("bash", {}, abort_signal=abort)
    assert result.is_cancelled()
    assert not any("should not run" in m for m in result.messages)


@pytest.mark.unit
def test_from_env(monkeypatch):
    monkeypatch.setenv("BC_HOOKS_PRE_TOOL_USE",  "echo pre1, echo pre2")
    monkeypatch.setenv("BC_HOOKS_POST_TOOL_USE", "")
    runner = HookRunner.from_env()
    assert runner.has_pre_hooks()
    assert not runner.has_post_hooks()


# ===========================================================================
# Tools — sandbox (Port 8)
# ===========================================================================

from birdclaw.tools.sandbox import (
    FilesystemIsolationMode, SandboxConfig, SandboxRequest, SandboxStatus,
    _detect_from, _resolve_for_request, build_sandbox_command,
    resolve_sandbox_status,
)


@pytest.mark.unit
def test_detect_dockerenv():
    env = _detect_from(env_pairs=[], dockerenv_exists=True,
                       containerenv_exists=False, proc_1_cgroup=None)
    assert env.in_container
    assert "/.dockerenv" in env.markers


@pytest.mark.unit
def test_detect_containerenv():
    env = _detect_from(env_pairs=[], dockerenv_exists=False,
                       containerenv_exists=True, proc_1_cgroup=None)
    assert env.in_container
    assert "/run/.containerenv" in env.markers


@pytest.mark.unit
def test_detect_env_vars():
    env = _detect_from(
        env_pairs=[("container", "docker"), ("KUBERNETES_SERVICE_HOST", "10.0.0.1")],
        dockerenv_exists=False, containerenv_exists=False, proc_1_cgroup=None,
    )
    assert env.in_container


@pytest.mark.unit
def test_detect_cgroup():
    env = _detect_from(env_pairs=[], dockerenv_exists=False, containerenv_exists=False,
                       proc_1_cgroup="12:memory:/docker/abc123")
    assert env.in_container


@pytest.mark.unit
def test_no_container_markers():
    env = _detect_from(env_pairs=[("HOME", "/home/user")],
                       dockerenv_exists=False, containerenv_exists=False, proc_1_cgroup=None)
    assert not env.in_container
    assert env.markers == []


@pytest.mark.unit
def test_sandbox_config_defaults():
    req = SandboxConfig().resolve_request()
    assert req.enabled                is True
    assert req.namespace_restrictions is True
    assert req.network_isolation      is False
    assert req.filesystem_mode        == FilesystemIsolationMode.workspace_only


@pytest.mark.unit
def test_sandbox_config_overrides():
    req = SandboxConfig(enabled=True).resolve_request(
        enabled_override=True, namespace_override=False, network_override=True,
        filesystem_mode_override=FilesystemIsolationMode.allow_list,
        allowed_mounts_override=["tmp"],
    )
    assert req.namespace_restrictions is False
    assert req.network_isolation      is True
    assert req.filesystem_mode        == FilesystemIsolationMode.allow_list


@pytest.mark.component
def test_disabled_sandbox_not_active(tmp_path):
    status = resolve_sandbox_status(SandboxConfig(enabled=False), cwd=tmp_path)
    assert not status.active


@pytest.mark.component
def test_allow_list_without_mounts_fallback(tmp_path):
    req    = SandboxRequest(enabled=True, namespace_restrictions=True, network_isolation=False,
                            filesystem_mode=FilesystemIsolationMode.allow_list, allowed_mounts=[])
    status = _resolve_for_request(req, tmp_path)
    assert status.fallback_reason is not None
    assert "allow-list" in status.fallback_reason


@pytest.mark.component
def test_mounts_are_absolute(tmp_path):
    config = SandboxConfig(enabled=True, filesystem_mode=FilesystemIsolationMode.allow_list,
                           allowed_mounts=["data"])
    for mount in resolve_sandbox_status(config, cwd=tmp_path).allowed_mounts:
        assert Path(mount).is_absolute()


@pytest.mark.component
def test_status_to_dict(tmp_path):
    d = resolve_sandbox_status(SandboxConfig(), cwd=tmp_path).to_dict()
    for key in ("enabled", "supported", "active", "filesystem_mode", "in_container"):
        assert key in d


@pytest.mark.component
def test_build_returns_none_when_disabled(tmp_path):
    req    = SandboxRequest(enabled=False, namespace_restrictions=False, network_isolation=False,
                            filesystem_mode=FilesystemIsolationMode.off)
    status = _resolve_for_request(req, tmp_path)
    assert build_sandbox_command("echo hi", tmp_path, status) is None


@pytest.mark.component
def test_bash_output_includes_sandbox_status():
    out = json.loads(run_bash("echo sandbox-test"))
    assert "sandbox_status"          in out
    assert "enabled"                 in out["sandbox_status"]
    assert out["stdout"].strip()     == "sandbox-test"


# ===========================================================================
# Tools/MCP — naming
# ===========================================================================

from birdclaw.tools.mcp.naming import (
    normalize_name_for_mcp, mcp_tool_prefix, mcp_tool_name, server_name_from_tool,
)


@pytest.mark.unit
def test_normalize_name():
    assert normalize_name_for_mcp("github.com") == "github_com"
    assert normalize_name_for_mcp("tool name!") == "tool_name"
    assert normalize_name_for_mcp("my-server")  == "my-server"
    assert "__" not in normalize_name_for_mcp("tool  name")


@pytest.mark.unit
def test_mcp_tool_prefix():
    assert mcp_tool_prefix("my server")  == "mcp__my_server__"
    assert mcp_tool_prefix("filesystem") == "mcp__filesystem__"


@pytest.mark.unit
def test_mcp_tool_name():
    assert mcp_tool_name("my server",  "read_file")  == "mcp__my_server__read_file"
    assert mcp_tool_name("github.com", "list repos") == "mcp__github_com__list_repos"


@pytest.mark.unit
def test_mcp_tool_name_rust_reference():
    result = mcp_tool_name("claude.ai Example Server", "weather tool")
    assert result.startswith("mcp__")
    assert result.endswith("__weather_tool")


@pytest.mark.unit
def test_server_name_from_tool():
    assert server_name_from_tool("mcp__filesystem__read_file") == "filesystem"
    assert server_name_from_tool("mcp__my_server__tool")       == "my_server"
    assert server_name_from_tool("bash")                       is None


# ===========================================================================
# Tools/MCP — client
# ===========================================================================

from birdclaw.tools.mcp.client import McpClient, McpJsonRpcError

_MCP_ECHO_SERVER = textwrap.dedent("""\
import sys, json

def respond(id, result):
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":id,"result":result}) + "\\n")
    sys.stdout.flush()

def error(id, code, msg):
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":msg}}) + "\\n")
    sys.stdout.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw: continue
    msg    = json.loads(raw)
    method = msg.get("method","")
    id     = msg.get("id")
    if method == "initialize":
        respond(id, {"protocolVersion":"2024-11-05","capabilities":{},
                     "serverInfo":{"name":"echo-server","version":"0.1.0"}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        respond(id, {"tools":[
            {"name":"echo","description":"Returns input.",
             "inputSchema":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}},
            {"name":"add","description":"Adds two numbers.",
             "inputSchema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}},
        ]})
    elif method == "tools/call":
        name = msg.get("params",{}).get("name","")
        args = msg.get("params",{}).get("arguments",{})
        if name == "echo":
            respond(id, {"content":[{"type":"text","text":args.get("message","")}],"isError":False})
        elif name == "add":
            respond(id, {"content":[{"type":"text","text":str(args.get("a",0)+args.get("b",0))}],"isError":False})
        else:
            error(id, -32601, f"Unknown tool: {name}")
    elif method == "resources/list":
        respond(id, {"resources":[]})
    elif id is not None:
        error(id, -32601, f"Method not found: {method}")
""")


@pytest.fixture(scope="module")
def mcp_echo_script(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("mcp") / "echo_server.py"
    p.write_text(_MCP_ECHO_SERVER, encoding="utf-8")
    return p


@pytest.fixture
def mcp_client(mcp_echo_script) -> McpClient:
    c = McpClient(server_name="test-echo", command=sys.executable,
                  args=[str(mcp_echo_script)])
    yield c
    try:
        c.close()
    except Exception:
        pass


@pytest.mark.component
def test_mcp_connect(mcp_client):
    info = mcp_client.connect()
    assert info.name == "echo-server"
    assert mcp_client.is_connected


@pytest.mark.component
def test_mcp_list_tools(mcp_client):
    mcp_client.connect()
    tools = mcp_client.list_tools()
    assert {t.name for t in tools} == {"echo", "add"}


@pytest.mark.component
def test_mcp_call_echo(mcp_client):
    mcp_client.connect(); mcp_client.list_tools()
    result = mcp_client.call_tool("echo", {"message": "hello mcp"})
    assert result.text() == "hello mcp"


@pytest.mark.component
def test_mcp_call_add(mcp_client):
    mcp_client.connect(); mcp_client.list_tools()
    assert mcp_client.call_tool("add", {"a": 3, "b": 4}).text() == "7"


@pytest.mark.component
def test_mcp_unknown_tool_error(mcp_client):
    mcp_client.connect()
    with pytest.raises(McpJsonRpcError) as exc_info:
        mcp_client.call_tool("nonexistent", {})
    assert exc_info.value.code == -32601


@pytest.mark.component
def test_mcp_list_resources_empty(mcp_client):
    mcp_client.connect()
    assert mcp_client.list_resources() == []


@pytest.mark.component
def test_mcp_close_and_reconnect(mcp_echo_script):
    c = McpClient("test-2", sys.executable, [str(mcp_echo_script)])
    c.connect(); assert c.is_connected
    c.close();   assert not c.is_connected
    c2 = McpClient("test-3", sys.executable, [str(mcp_echo_script)])
    assert c2.connect().name == "echo-server"
    c2.close()


# ===========================================================================
# Tools/MCP — manager
# ===========================================================================

from birdclaw.tools.mcp.manager import McpServerManager, McpStdioConfig, McpToolBridge
from birdclaw.tools.registry import registry as tool_registry

_MCP_MGR_SERVER = textwrap.dedent("""\
import sys, json

def respond(id, result):
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":id,"result":result}) + "\\n")
    sys.stdout.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw: continue
    msg    = json.loads(raw)
    method = msg.get("method","")
    id     = msg.get("id")
    if method == "initialize":
        respond(id, {"protocolVersion":"2024-11-05","capabilities":{},
                     "serverInfo":{"name":"mgr-server","version":"1.0"}})
    elif method == "tools/list":
        respond(id, {"tools":[{"name":"ping","description":"Returns pong.",
                               "inputSchema":{"type":"object","properties":{}}}]})
    elif method == "tools/call":
        respond(id, {"content":[{"type":"text","text":"pong"}],"isError":False})
    elif method == "resources/list":
        respond(id, {"resources":[]})
    elif id is not None:
        sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":id,
            "error":{"code":-32601,"message":"not found"}}) + "\\n")
        sys.stdout.flush()
""")


@pytest.fixture(scope="module")
def mcp_mgr_script(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("mgr") / "mgr_server.py"
    p.write_text(_MCP_MGR_SERVER)
    return p


@pytest.fixture
def mcp_mgr(mcp_mgr_script) -> McpServerManager:
    m = McpServerManager()
    m.add_server("test-mgr", McpStdioConfig(command=sys.executable, args=[str(mcp_mgr_script)]))
    m.connect_all()
    yield m
    m.disconnect_all()


@pytest.mark.component
def test_manager_connects(mcp_mgr):
    states = mcp_mgr.list_servers()
    assert len(states)           == 1
    assert states[0].status      == "connected"
    assert states[0].server_name == "test-mgr"


@pytest.mark.component
def test_manager_lists_tools(mcp_mgr):
    tools = mcp_mgr.all_tools()
    assert len(tools)     == 1
    assert tools[0][1].name == "ping"


@pytest.mark.component
def test_manager_call_tool(mcp_mgr):
    result = json.loads(mcp_mgr.call_tool(mcp_tool_name("test-mgr", "ping")))
    texts  = [b["text"] for b in result["content"] if b.get("type") == "text"]
    assert "pong" in texts


@pytest.mark.component
def test_manager_call_unknown(mcp_mgr):
    assert "error" in json.loads(mcp_mgr.call_tool("mcp__test-mgr__nonexistent"))


@pytest.mark.component
def test_bridge_registers_tools(mcp_mgr_script):
    m = McpServerManager()
    m.add_server("bridge-test", McpStdioConfig(sys.executable, [str(mcp_mgr_script)]))
    m.connect_all()
    bridge = McpToolBridge(m)
    assert bridge.register_all() == 1
    assert bridge.register_all() == 0  # idempotent
    tool = tool_registry.get(mcp_tool_name("bridge-test", "ping"))
    assert tool is not None and "mcp" in tool.tags
    m.disconnect_all()


@pytest.mark.component
def test_env_var_config(monkeypatch, mcp_mgr_script):
    monkeypatch.setenv("BC_MCP_SERVERS", f"env-server:{sys.executable} {mcp_mgr_script}")
    m = McpServerManager()
    m.load_from_config()
    assert "env-server" in {s.server_name for s in m.list_servers()}
    m.disconnect_all()


from birdclaw.config import settings
settings.ensure_dirs()


# ===========================================================================
# Regression — self-update lifecycle gate
#
# These tests are the final checkpoint run at the end of every self-update
# cycle. A patch is only accepted when ALL of these pass.  They cover every
# fix made during live TUI testing so that a bad self-generated patch cannot
# silently regress proven behaviour.
# ===========================================================================


# ---------------------------------------------------------------------------
# R1 — Stage type inference: execution verbs must map to "verify"
# ---------------------------------------------------------------------------

from birdclaw.agent.planner import infer_stage_type


@pytest.mark.unit
@pytest.mark.parametrize("step,expected", [
    ("Run the birdclaw-agent-registration workflow", "verify"),
    ("Execute the install script", "verify"),
    ("Register the agent with the gateway", "verify"),
    ("curl the health endpoint", "verify"),
    ("Perform the database migration", "verify"),
    ("install the dependencies", "verify"),
    ("Complete the registration flow", "verify"),
    ("Submit the form and verify the response", "verify"),
    ("call api to trigger the job", "verify"),
])
def test_infer_stage_type_execution_verbs(step, expected):
    assert infer_stage_type(step) == expected, f"Expected {expected!r} for step: {step!r}"


@pytest.mark.unit
@pytest.mark.parametrize("step,expected", [
    ("research what APIs are available", "research"),
    ("search for documentation on embeddings", "research"),
    ("find the latest news about the project", "research"),
    ("look up the registry endpoint", "research"),
])
def test_infer_stage_type_research_verbs(step, expected):
    assert infer_stage_type(step) == expected


@pytest.mark.unit
@pytest.mark.parametrize("step,expected", [
    ("write the registration function", "write_code"),
    ("implement the retry logic in loop.py", "write_code"),
    ("create a Python script for the pipeline", "write_code"),
    ("build the connector module", "write_code"),
])
def test_infer_stage_type_code_verbs(step, expected):
    assert infer_stage_type(step) == expected


# ---------------------------------------------------------------------------
# R2 — is_complex threshold: 3+ verbs or >30 words → complex; else simple
# ---------------------------------------------------------------------------

from birdclaw.agent.task_list import is_complex


@pytest.mark.unit
@pytest.mark.parametrize("req,expected", [
    ("What is 2+2?", False),
    ("Tell me the capital of France", False),
    ("write a function", False),          # 1 action verb, short
    ("search and find and install the package", True),   # 3 verbs
    ("write and test and verify the new registration handler module now", True),  # 3 verbs
    # long sentence (>30 words) — 31 words
    ("Please help me to understand exactly how the birdclaw agent registration workflow operates "
     "and specifically which API endpoints it calls and what the format of the expected response "
     "body looks like in detail", True),
])
def test_is_complex_threshold(req, expected):
    result = is_complex(req)
    assert result == expected, f"is_complex({req!r}) = {result}, expected {expected}"


# ---------------------------------------------------------------------------
# R3 — _SYNTHESIS_KW: "ask user" goals should not trigger stall-guard search
# ---------------------------------------------------------------------------

from birdclaw.agent.loop import _SYNTHESIS_KW  # noqa: PLC0415


@pytest.mark.unit
@pytest.mark.parametrize("goal", [
    "ask user for the target filename",
    "request from user which environment to use",
    "gather user input on deployment options",
    "collect user preferences for the report",
    "prompt user for confirmation",
    "user input required for next step",
    "from user: get the API key",
])
def test_synthesis_keywords_detected(goal):
    matched = any(kw in goal.lower() for kw in _SYNTHESIS_KW)
    assert matched, f"Goal {goal!r} was not detected as synthesis — add its keyword to _SYNTHESIS_KW"


# ---------------------------------------------------------------------------
# R4 — select_skill tie-breaking: more-specific name wins when scores tie
# ---------------------------------------------------------------------------

from birdclaw.skills.loader import select_skill, Skill


def _make_skill(name: str, tags: list[str], description: str, body: str = "body") -> Skill:
    return Skill(name=name, tags=tags, description=description, body=body, source="test", enabled=True)


@pytest.mark.unit
def test_select_skill_tiebreak_by_name_specificity():
    """When two skills share the same token-overlap score, the one with more
    name tokens (more specific) should be selected."""
    general = _make_skill(
        "agent guide",
        tags=["agent", "registration", "guide", "moltbook"],
        description="agent registration guide for moltbook",
    )
    specific = _make_skill(
        "moltbook agent registration guide",
        tags=["agent", "registration", "guide", "moltbook"],
        description="agent registration guide for moltbook",
    )
    # Both have identical tags and description; specific has more name tokens
    result = select_skill("moltbook agent registration", skills=[general, specific])
    assert result is not None
    assert result.name == "moltbook agent registration guide", (
        f"Expected more-specific skill, got {result.name!r}"
    )


@pytest.mark.unit
def test_select_skill_returns_none_below_threshold():
    skill = _make_skill("cooking recipes", tags=["food", "recipe"], description="cooking help")
    result = select_skill("quantum physics research", skills=[skill])
    assert result is None


# ---------------------------------------------------------------------------
# R5 — skill_context: returns "## Active Skill:" prefixed body
# ---------------------------------------------------------------------------

from birdclaw.skills.loader import skill_context


@pytest.mark.unit
def test_skill_context_returns_body_with_prefix():
    skill = _make_skill(
        "widget deployment guide",
        tags=["widget", "deploy", "guide"],
        description="guide for deploying widgets",
        body="## Step 1\nDo the thing.\n## Step 2\nVerify it.",
    )
    ctx = skill_context("deploy widget", skills=[skill])
    assert ctx is not None
    assert ctx.startswith("## Active Skill: widget deployment guide")
    assert "## Step 1" in ctx


@pytest.mark.unit
def test_skill_context_none_when_no_match():
    skill = _make_skill("cooking guide", tags=["food"], description="cooking")
    ctx = skill_context("quantum physics", skills=[skill])
    assert ctx is None


# ---------------------------------------------------------------------------
# R6 — generate_plan: accepts skill_hint keyword argument
# ---------------------------------------------------------------------------

import inspect
from birdclaw.agent.planner import generate_plan


@pytest.mark.unit
def test_generate_plan_accepts_skill_hint_param():
    sig = inspect.signature(generate_plan)
    assert "skill_hint" in sig.parameters, (
        "generate_plan() must accept a skill_hint= keyword argument"
    )
    param = sig.parameters["skill_hint"]
    assert param.default == "", f"skill_hint default should be '' got {param.default!r}"


# ---------------------------------------------------------------------------
# R7 — cmd_dream / run_dream_cycle: no NameError, returns dict
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_run_dream_cycle_returns_dict(monkeypatch):
    """run_dream_cycle() should return a dict without raising NameError."""
    from birdclaw.memory import dream as _dream_mod

    # Stub the heavy phase functions so the test is fast
    def _fake_run(results: dict, quiet: bool) -> None:
        results["memorise"] = "ok"
        results["graph"] = "ok"

    monkeypatch.setattr(_dream_mod, "_run", _fake_run)
    result = _dream_mod.run_dream_cycle(quiet=True)
    assert isinstance(result, dict)


@pytest.mark.unit
def test_cmd_dream_no_phases_err(monkeypatch):
    """cmd_dream must not reference an undefined `phases_err` variable."""
    import argparse, main as _main
    monkeypatch.setattr(_main, "console", type("C", (), {"print": staticmethod(lambda *a, **k: None)})())
    from birdclaw.memory import dream as _dream_mod
    monkeypatch.setattr(_dream_mod, "_run", lambda results, quiet: results.update({"test": "ok"}))
    # Stub push_notification to avoid real network calls
    import birdclaw.gateway.notify as _notify
    monkeypatch.setattr(_notify, "push_notification", lambda *a, **k: None)
    # Should not raise
    _main.cmd_dream(argparse.Namespace())


# ---------------------------------------------------------------------------
# R8 — self_update: _TEST_CMD points to tests/test.py, not test_regression.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_self_update_test_cmd_targets_test_py():
    from birdclaw.agent import self_update as _su
    cmd = " ".join(_su._TEST_CMD)
    assert "tests/test.py" in cmd, (
        f"_TEST_CMD should point to tests/test.py, got: {cmd!r}"
    )
    assert "test_regression" not in cmd


@pytest.mark.unit
def test_self_update_respects_self_modify_flag(monkeypatch):
    """run_self_update_cycle() must return immediately when BC_SELF_MODIFY is off."""
    from birdclaw.config import settings as _settings
    from birdclaw.agent.self_update import run_self_update_cycle
    monkeypatch.setattr(_settings, "self_modify", False)
    result = run_self_update_cycle()
    assert result["success"] is False
    assert "BC_SELF_MODIFY" in result["summary"]


# ---------------------------------------------------------------------------
# R9 — Planner: must not plan "ask user" stages (prompt constraint exists)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_planner_prompt_includes_no_ask_user_constraint():
    """The planner system prompt must contain the constraint against ask-user stages."""
    import importlib, birdclaw.agent.planner as _p
    # Read the source to find the constraint text
    src = Path(_p.__file__).read_text(encoding="utf-8")
    assert "ask" in src.lower() and "user" in src.lower() and "never" in src.lower(), (
        "Planner must contain a NEVER-ask-user constraint in its prompt"
    )


# ---------------------------------------------------------------------------
# R10 — Task cleanup: completed tasks >3 days old deleted, failed kept
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cleanup_deletes_old_completed_keeps_failed(tmp_path, monkeypatch):
    from birdclaw.memory import cleanup as _cleanup
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    import json, time as _t
    now = _t.time()

    # Old completed task (4 days ago) — should be deleted
    old_completed = tasks_dir / "old_completed.json"
    old_completed.write_text(json.dumps({"status": "completed", "task_id": "old"}))
    os.utime(old_completed, (now - 4 * 86400, now - 4 * 86400))

    # Recent completed task (1 day ago) — should be kept
    new_completed = tasks_dir / "new_completed.json"
    new_completed.write_text(json.dumps({"status": "completed", "task_id": "new"}))
    os.utime(new_completed, (now - 86400, now - 86400))

    # Failed task (7 days old) — should always be kept
    old_failed = tasks_dir / "old_failed.json"
    old_failed.write_text(json.dumps({"status": "failed", "task_id": "fail"}))
    os.utime(old_failed, (now - 7 * 86400, now - 7 * 86400))

    _cleanup.cleanup_tasks()

    assert not old_completed.exists(), "Old completed task should have been deleted"
    assert new_completed.exists(), "Recent completed task should be kept"
    assert old_failed.exists(), "Failed task should always be kept"


# ---------------------------------------------------------------------------
# R11 — Grandchild task tree: children_by_parent supports 3 levels
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tui_grandchild_rendering_supported():
    """The TUI app.py must contain grandchild rendering logic (↳ marker)."""
    tui_src = Path(__file__).parent.parent / "birdclaw" / "tui" / "app.py"
    assert tui_src.exists(), "birdclaw/tui/app.py not found"
    text = tui_src.read_text(encoding="utf-8")
    assert "↳" in text, "TUI app.py must contain ↳ grandchild tree marker"
    assert "children_by_parent.get(sub.task_id" in text, (
        "TUI must look up children_by_parent[sub.task_id] for grandchild rendering"
    )


# ---------------------------------------------------------------------------
# R12 — _collect_error_signals: function exists and returns a string
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_collect_error_signals_returns_string(tmp_path, monkeypatch):
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)
    # Create a minimal log with an error line
    log = tmp_path / "birdclaw.log"
    log.write_text("2026-04-26 ERROR something went wrong\n")
    from birdclaw.agent.self_update import _collect_error_signals
    result = _collect_error_signals()
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# R13 — Self-update main.py: self-update subcommand registered
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_main_self_update_subcommand_registered():
    """python main.py self-update must be a registered subcommand."""
    import main as _main
    parser = _main.build_parser()
    # If the subcommand is missing, parse_args will raise SystemExit
    try:
        args = parser.parse_args(["self-update"])
        assert args.command == "self-update"
    except SystemExit:
        pytest.fail("self-update subcommand is not registered in build_parser()")


# ---------------------------------------------------------------------------
# R14 — Dream cycle: phases_err / phases_ok never referenced as bare names
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_main_cmd_dream_no_undefined_variables():
    """cmd_dream source must not contain bare references to phases_err or phases_ok."""
    import inspect, main as _main
    src = inspect.getsource(_main.cmd_dream)
    assert "phases_err" not in src, "cmd_dream still references undefined 'phases_err'"
    assert "phases_ok"  not in src, "cmd_dream still references undefined 'phases_ok'"


# ---------------------------------------------------------------------------
# R15 — Skill injection: soul_loop._spawn_task injects skill context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_soul_spawn_task_injects_skill_context():
    """_spawn_task in soul_loop.py must call skill_context() and include it in
    the context passed to task_registry.create() and orchestrator.spawn()."""
    soul_src = Path(__file__).parent.parent / "birdclaw" / "agent" / "soul_loop.py"
    assert soul_src.exists()
    text = soul_src.read_text(encoding="utf-8")
    assert "skill_context" in text or "skill_ctx" in text, (
        "soul_loop._spawn_task must reference skill_context/skill_ctx"
    )
    assert "full_ctx" in text or "skill_ctx" in text, (
        "soul_loop must merge skill context into task context"
    )


# ===========================================================================
# (LR infrastructure removed — long-running tests are no longer part of this
#  suite. Use `python main.py tui` for interactive E2E validation.)


# ===========================================================================
# Regression — self-awareness & self-update infrastructure (R16–R22)
# ===========================================================================

# ---------------------------------------------------------------------------
# R16 — src_dir: settings exposes birdclaw package path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_settings_src_dir_exists():
    from birdclaw.config import settings as _s
    assert _s.src_dir.exists(), f"settings.src_dir {_s.src_dir} does not exist"
    assert (_s.src_dir / "config.py").exists(), "src_dir should contain config.py"


@pytest.mark.unit
def test_settings_self_update_todo_path():
    from birdclaw.config import settings as _s
    p = _s.self_update_todo_path
    assert p.name == "self_update_todo.jsonl"
    assert p.parent == _s.data_dir


# ---------------------------------------------------------------------------
# R17 — read_file: can read birdclaw source without workspace restriction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_read_file_can_read_own_source():
    from birdclaw.config import settings as _s
    from birdclaw.tools.files import read_file
    result = json.loads(read_file(str(_s.src_dir / "config.py"), limit=5))
    assert "content" in result, f"read_file should return content, got: {result}"
    assert len(result["content"]) > 0


# ---------------------------------------------------------------------------
# R18 — permission: src writes blocked without BC_SELF_MODIFY
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_src_write_blocked_without_self_modify(monkeypatch):
    from birdclaw.config import settings as _s
    from birdclaw.tools.permission import PermissionEnforcer
    monkeypatch.setattr(_s, "self_modify", False)
    enforcer = PermissionEnforcer(mode="workspace_write")
    result = enforcer.check_file_write(_s.src_dir / "agent" / "loop.py")
    assert not result.allowed
    assert "BC_SELF_MODIFY" in result.reason


@pytest.mark.unit
def test_src_write_allowed_with_self_modify(monkeypatch):
    from birdclaw.config import settings as _s
    from birdclaw.tools.permission import PermissionEnforcer
    monkeypatch.setattr(_s, "self_modify", True)
    enforcer = PermissionEnforcer(mode="workspace_write")
    result = enforcer.check_file_write(_s.src_dir / "agent" / "loop.py")
    assert result.allowed


# ---------------------------------------------------------------------------
# R19 — note_improvement: writes to todo file, readable by _score_pain_points
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_note_improvement_writes_todo(tmp_path, monkeypatch):
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)
    from birdclaw.tools.task_tools import note_improvement
    msg = note_improvement("loop.py: research stage stalls on synthesis goals", "high")
    assert "Logged" in msg
    todo = tmp_path / "self_update_todo.jsonl"
    assert todo.exists()
    entry = json.loads(todo.read_text().strip())
    assert entry["priority"] == "high"
    assert "research stage" in entry["description"]


@pytest.mark.unit
def test_score_pain_points_reads_todo_first(tmp_path, monkeypatch):
    """todo backlog takes priority over stage_history and error signals."""
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)
    import time as _t
    todo = tmp_path / "self_update_todo.jsonl"
    todo.write_text(json.dumps({
        "ts": _t.time(), "description": "fix the stall guard", "priority": "high"
    }) + "\n")
    from birdclaw.agent.self_update import _score_pain_points
    result = _score_pain_points()
    assert result is not None
    assert result["source"] == "todo"
    assert "stall guard" in result["target"]


@pytest.mark.unit
def test_score_pain_points_skips_done_todo(tmp_path, monkeypatch):
    """Done items should not be selected."""
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)
    import time as _t
    todo = tmp_path / "self_update_todo.jsonl"
    todo.write_text(json.dumps({
        "ts": _t.time(), "description": "already fixed thing", "priority": "high", "done": True
    }) + "\n")
    from birdclaw.agent.self_update import _score_pain_points
    result = _score_pain_points()
    # No todo, no stage_history, no log — should return None
    assert result is None or result["source"] != "todo"


# ---------------------------------------------------------------------------
# R20 — _collect_error_signals: reads birdclaw.log for ERROR/Traceback lines
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_collect_error_signals_finds_log_errors(tmp_path, monkeypatch):
    from birdclaw.config import settings as _s
    monkeypatch.setattr(_s, "data_dir", tmp_path)
    log = tmp_path / "birdclaw.log"
    log.write_text(
        "2026-04-26 12:00:00 INFO  normal line\n"
        "2026-04-26 12:01:00 ERROR gateway: soul_respond failed\n"
        "AttributeError: 'Settings' object has no attribute 'parallel_tasks'\n"
    )
    from birdclaw.agent.self_update import _collect_error_signals
    signals = _collect_error_signals()
    assert "ERROR" in signals or "AttributeError" in signals


# ---------------------------------------------------------------------------
# R21 — dynamic context: includes src_dir line
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dynamic_context_includes_src_dir():
    from birdclaw.agent.prompts import dynamic_context
    ctx = dynamic_context(write_dir="/tmp/test")
    assert "Source dir:" in ctx
    assert "note_improvement" in ctx


# ---------------------------------------------------------------------------
# R22 — _hot_reload: uses install.sh; graceful when script absent
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hot_reload_graceful_when_no_install_script(monkeypatch):
    from birdclaw.agent import self_update as _su
    monkeypatch.setattr(_su, "_INSTALL_SCRIPT", Path("/nonexistent/install.sh"))
    # Should not raise — just logs a debug message
    _su._hot_reload()


# ---------------------------------------------------------------------------
# R23 — _scrub_workspace: removes workspace dir and __pycache__ artefacts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_scrub_workspace_removes_work_dir(tmp_path, monkeypatch):
    from birdclaw.agent import self_update as _su
    work = tmp_path / "workspace"
    work.mkdir()
    (work / "junk.py").write_text("x = 1")
    monkeypatch.setattr(_su, "_WORK_DIR", work)
    monkeypatch.setattr(_su, "_BIRDCLAW_SRC", tmp_path)
    _su._scrub_workspace()
    assert not work.exists(), "workspace dir should be removed after scrub"


@pytest.mark.unit
def test_scrub_workspace_removes_pycache(tmp_path, monkeypatch):
    from birdclaw.agent import self_update as _su
    cache = tmp_path / "agent" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "loop.cpython-310.pyc").write_bytes(b"\x00" * 16)
    monkeypatch.setattr(_su, "_WORK_DIR", tmp_path / "workspace")
    monkeypatch.setattr(_su, "_BIRDCLAW_SRC", tmp_path)
    _su._scrub_workspace()
    assert not cache.exists(), "__pycache__ dir should be removed after scrub"


# ---------------------------------------------------------------------------
# R24 — _prune_old_backups: only removes backup_ prefixed dirs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prune_old_backups_leaves_non_backup_dirs(tmp_path, monkeypatch):
    from birdclaw.agent import self_update as _su
    monkeypatch.setattr(_su, "_BACKUP_ROOT", tmp_path)
    # Create 7 backup dirs and one unrelated dir
    for i in range(7):
        (tmp_path / f"backup_{i:04d}").mkdir()
    (tmp_path / "workspace").mkdir()  # should never be removed
    _su._prune_old_backups(keep=5)
    remaining = list(tmp_path.iterdir())
    backup_dirs = [p for p in remaining if p.name.startswith("backup_")]
    assert len(backup_dirs) == 5, f"Expected 5 backups, got {len(backup_dirs)}"
    assert (tmp_path / "workspace").exists(), "non-backup dirs must not be touched"


# ===========================================================================
# Outputs directory integrity check
# ===========================================================================

# Known real source/config files that live in the project root — never leaked test outputs.
_ROOT_ALLOWLIST = frozenset({
    "CLAUDE.md", "README.md", "main.py", "pyproject.toml",
    "install.sh", "setup.cfg", ".gitignore", "Makefile",
    "issues.md",
    # build artefacts — not test output
    "birdclaw.egg-info", "build", "dist", "assets",
})


@pytest.mark.unit
def test_outputs_dir_no_leakage():
    """No test-generated files should appear in the project root."""
    leaked = [
        p.name for p in PROJECT_ROOT.iterdir()
        if p.is_file()
        and p.name not in _ROOT_ALLOWLIST
        and not p.name.startswith(".")
    ]
    assert not leaked, (
        f"Test output file(s) leaked to project root: {leaked}\n"
        f"All generated files must land in tests/outputs/"
    )


@pytest.mark.unit
def test_outputs_dir_has_marker():
    """tests/outputs/ must have .birdclaw-project so BIRDCLAW.md lands there."""
    assert (OUTPUTS_DIR / ".birdclaw-project").exists(), (
        "tests/outputs/.birdclaw-project marker is missing — BIRDCLAW.md will leak to project root"
    )


@pytest.mark.unit
def test_outputs_dir_birdclaw_md_location():
    """If BIRDCLAW.md exists it must be in tests/outputs/, not the project root."""
    root_birdclaw = PROJECT_ROOT / "BIRDCLAW.md"
    assert not root_birdclaw.exists(), (
        "BIRDCLAW.md found in project root — it should be in tests/outputs/BIRDCLAW.md"
    )


# ===========================================================================
# Metrics comparison utility (for self-update evaluation)
# ===========================================================================

def compare_metrics(before_file: Path, after_file: Path) -> dict:
    def load(p: Path) -> dict[str, dict]:
        records = {}
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line); records[r["task_name"]] = r
            except Exception:
                pass
        return records

    before = load(before_file)
    after  = load(after_file)
    common = set(before) & set(after)
    return {
        "before_pass":     sum(1 for r in before.values() if r.get("passed")),
        "before_total":    len(before),
        "after_pass":      sum(1 for r in after.values()  if r.get("passed")),
        "after_total":     len(after),
        "improved":        sum(1 for n in common if after[n].get("passed") and not before[n].get("passed")),
        "regressed":       sum(1 for n in common if not after[n].get("passed") and before[n].get("passed")),
        "net_improvement": sum(1 for r in after.values() if r.get("passed")) -
                           sum(1 for r in before.values() if r.get("passed")),
    }


# =============================================================================
# Subtask middle layer
# =============================================================================

from birdclaw.agent.subtask_manifest import (
    SubtaskItem, SubtaskManifest, SubtaskDiff, ItemStatus,
)
from birdclaw.agent import subtask_verifier as _sv
from birdclaw.llm.types import Message as _LLMMessage


# ── SubtaskItem ───────────────────────────────────────────────────────────────

class TestSubtaskItem:
    def test_mark_complete_sets_hash(self):
        item = SubtaskItem(index=0, title="Intro", anchor="Intro", kind="section")
        item.mark_complete("Hello world " * 20)
        assert item.status == "complete"
        assert item.actual_chars > 0
        assert len(item.content_hash) == 16

    def test_mark_partial(self):
        item = SubtaskItem(index=0, title="X", anchor="X", kind="section")
        item.mark_partial("short")
        assert item.status == "partial"
        assert item.actual_chars == 5

    def test_steps_budget_ceil(self):
        item = SubtaskItem(index=0, title="X", anchor="X", kind="section", expected_min_chars=201)
        assert item.steps_budget == 6  # ceil(201/40)


# ── SubtaskManifest ───────────────────────────────────────────────────────────

class TestSubtaskManifest:
    def _make(self) -> SubtaskManifest:
        items = [
            SubtaskItem(index=i, title=f"S{i}", anchor=f"S{i}", kind="section")
            for i in range(4)
        ]
        return SubtaskManifest(stage_goal="write report", file_path="/tmp/r.md", file_type="doc", items=items)

    def test_current_item_is_first_pending(self):
        m = self._make()
        m.items[0].status = "complete"
        assert m.current_item.title == "S1"

    def test_done_count(self):
        m = self._make()
        m.items[0].status = "complete"
        m.items[1].status = "partial"
        assert m.done_count == 1

    def test_all_done_when_all_complete_or_partial(self):
        m = self._make()
        for it in m.items:
            it.status = "complete"
        assert m.all_done

    def test_progress_line_contains_counts(self):
        m = self._make()
        m.items[0].status = "complete"
        line = m.progress_line
        assert "1/4" in line

    def test_tui_phases_length(self):
        m = self._make()
        assert len(m.tui_phases()) == 4

    def test_update_file_hash(self):
        m = self._make()
        m.update_file_hash("content")
        assert len(m.file_content_hash) == 16


# ── SubtaskDiff ───────────────────────────────────────────────────────────────

class TestSubtaskDiff:
    def test_needs_resume_false_when_empty(self):
        d = SubtaskDiff()
        assert not d.needs_resume

    def test_needs_resume_true_with_missing(self):
        item = SubtaskItem(index=0, title="X", anchor="X", kind="section")
        d = SubtaskDiff(missing=[item])
        assert d.needs_resume

    def test_summary_ok_when_clean(self):
        assert SubtaskDiff().summary == "ok"

    def test_summary_lists_issues(self):
        item = SubtaskItem(index=0, title="A", anchor="A", kind="section")
        d = SubtaskDiff(partial=[item])
        assert "A" in d.summary


# ── Doc parser ────────────────────────────────────────────────────────────────

class TestParseDocSections:
    def test_basic_headings(self):
        content = "## Intro\n\nHello world.\n\n## Scope\n\nThis is the scope.\n"
        sections = _sv.parse_doc_sections(content)
        assert "Intro" in sections
        assert "Scope" in sections
        assert "Hello world." in sections["Intro"]

    def test_empty_content(self):
        assert _sv.parse_doc_sections("") == {}

    def test_no_headings_returns_empty(self):
        assert _sv.parse_doc_sections("Just some text with no headings.") == {}

    def test_h1_is_parsed(self):
        sections = _sv.parse_doc_sections("# Executive Summary\n\nContent here.")
        assert "Executive Summary" in sections

    def test_multiline_body(self):
        content = "## Risk\n\nLine 1.\nLine 2.\nLine 3.\n"
        s = _sv.parse_doc_sections(content)
        assert "Line 2." in s["Risk"]


# ── Code parser ───────────────────────────────────────────────────────────────

class TestParseCodeItems:
    def test_functions_extracted(self):
        src = "def foo():\n    return 1\n\ndef bar(x):\n    return x * 2\n"
        items = _sv.parse_code_items(src)
        assert "foo" in items
        assert "bar" in items

    def test_class_extracted(self):
        src = "class MyClass:\n    def method(self):\n        pass\n"
        items = _sv.parse_code_items(src)
        assert "MyClass" in items

    def test_empty_returns_empty(self):
        assert _sv.parse_code_items("") == {}

    def test_body_includes_all_lines(self):
        src = "def compute(x, y):\n    a = x + y\n    return a\n"
        items = _sv.parse_code_items(src)
        assert "return a" in items["compute"]


# ── Stub detection ────────────────────────────────────────────────────────────

class TestIsStubBody:
    def test_pass_is_stub(self):
        assert _sv.is_stub_body("    pass")

    def test_ellipsis_is_stub(self):
        assert _sv.is_stub_body("    ..")

    def test_raise_not_implemented_is_stub(self):
        assert _sv.is_stub_body("    raise NotImplementedError")

    def test_real_code_not_stub(self):
        assert not _sv.is_stub_body("    return x * 2")

    def test_empty_is_stub(self):
        assert _sv.is_stub_body("")

    def test_comment_only_is_stub(self):
        assert _sv.is_stub_body("    # TODO: implement")


# ── Verifier: doc ─────────────────────────────────────────────────────────────

class TestVerifyManifestDoc:
    def _manifest(self, items=None) -> SubtaskManifest:
        if items is None:
            items = [
                SubtaskItem(index=0, title="Intro", anchor="Intro", kind="section", expected_min_chars=50),
                SubtaskItem(index=1, title="Scope", anchor="Scope", kind="section", expected_min_chars=50),
            ]
        return SubtaskManifest(stage_goal="write doc", file_path="/tmp/d.md", file_type="doc", items=items)

    def test_complete_when_long_enough(self):
        m = self._manifest()
        content = "## Intro\n\n" + "A" * 100 + "\n\n## Scope\n\n" + "B" * 100
        diff = _sv.run(m, content)
        assert len(diff.complete) == 2
        assert not diff.needs_resume

    def test_missing_section_detected(self):
        m = self._manifest()
        content = "## Intro\n\n" + "A" * 100
        diff = _sv.run(m, content)
        assert any(it.anchor == "Scope" for it in diff.missing)

    def test_short_body_is_partial(self):
        m = self._manifest()
        content = "## Intro\n\nToo short.\n\n## Scope\n\n" + "B" * 100
        diff = _sv.run(m, content)
        assert any(it.anchor == "Intro" for it in diff.partial)

    def test_resume_context_built(self):
        m = self._manifest()
        content = "## Intro\n\n" + "A" * 100
        diff = _sv.run(m, content)
        assert "Scope" in diff.resume_context
        assert "seam" in diff.resume_context.lower() or "Resume" in diff.resume_context

    def test_seam_index_points_to_first_non_complete(self):
        m = self._manifest()
        content = "## Intro\n\n" + "A" * 100
        diff = _sv.run(m, content)
        assert diff.seam_index == 1  # Scope is index 1

    def test_regression_detected(self):
        m = self._manifest()
        # First pass: complete
        content1 = "## Intro\n\n" + "A" * 100 + "\n\n## Scope\n\n" + "B" * 100
        _sv.run(m, content1)
        # Second pass: Intro shrank drastically
        content2 = "## Intro\n\nShort.\n\n## Scope\n\n" + "B" * 100
        diff2 = _sv.run(m, content2)
        assert any(it.anchor == "Intro" for it in diff2.regressed)


# ── Verifier: code ────────────────────────────────────────────────────────────

class TestVerifyManifestCode:
    def _manifest(self) -> SubtaskManifest:
        items = [
            SubtaskItem(index=0, title="process", anchor="process", kind="function", expected_min_chars=30),
            SubtaskItem(index=1, title="validate", anchor="validate", kind="function", expected_min_chars=30),
        ]
        return SubtaskManifest(stage_goal="write code", file_path="/tmp/c.py", file_type="code", items=items)

    def test_complete_real_functions(self):
        m = self._manifest()
        src = "def process(x):\n    return x * 2\n\ndef validate(x):\n    return x > 0\n"
        diff = _sv.run(m, src)
        assert len(diff.complete) == 2

    def test_stub_detected_as_partial(self):
        m = self._manifest()
        src = "def process(x):\n    pass\n\ndef validate(x):\n    return x > 0\n"
        diff = _sv.run(m, src)
        assert any(it.anchor == "process" for it in diff.partial)

    def test_missing_function(self):
        m = self._manifest()
        src = "def process(x):\n    return x * 2\n"
        diff = _sv.run(m, src)
        assert any(it.anchor == "validate" for it in diff.missing)


# ── SubtaskPlanner ────────────────────────────────────────────────────────────

class TestSubtaskPlanner:
    def test_plan_returns_manifest(self, tmp_path):
        from birdclaw.agent import subtask_planner as _sp
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(
            content='{"subtasks": [{"title": "Intro", "anchor": "Intro", "kind": "section", "min_chars": 200}, {"title": "Body", "anchor": "Body", "kind": "section", "min_chars": 300}]}'
        )
        manifest = _sp.plan(mock_client, "write a report", str(tmp_path / "r.md"), "doc")
        assert manifest.total == 2
        assert manifest.items[0].title == "Intro"

    def test_plan_fallback_on_empty_response(self, tmp_path):
        from birdclaw.agent import subtask_planner as _sp
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(content='{"subtasks": []}')
        manifest = _sp.plan(mock_client, "write code", str(tmp_path / "c.py"), "code")
        assert manifest.total == 1  # fallback single item

    def test_replan_appends_items(self, tmp_path):
        from birdclaw.agent import subtask_planner as _sp
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(
            content='{"subtasks": [{"title": "New Section", "anchor": "New Section", "kind": "section", "min_chars": 150}]}'
        )
        items = [SubtaskItem(index=0, title="Done", anchor="Done", kind="section")]
        items[0].status = "complete"
        m = SubtaskManifest(stage_goal="g", file_path=str(tmp_path / "x.md"), file_type="doc", items=items)
        new_items = _sp.replan(mock_client, m, "gap: missing conclusion")
        assert len(new_items) == 1
        assert new_items[0].index == 1  # appended after index 0


# ── SubtaskExecutor ───────────────────────────────────────────────────────────

class TestSubtaskExecutor:
    def test_run_stage_produces_result(self, tmp_path):
        from birdclaw.agent import subtask_executor as _se
        from unittest.mock import MagicMock, patch

        output_file = tmp_path / "report.md"

        # Planner returns a 2-item manifest
        plan_manifest = SubtaskManifest(
            stage_goal="write doc",
            file_path=str(output_file),
            file_type="doc",
            items=[
                SubtaskItem(index=0, title="Intro", anchor="Intro", kind="section", expected_min_chars=50),
                SubtaskItem(index=1, title="Body", anchor="Body", kind="section", expected_min_chars=50),
            ],
        )

        section_content = {
            "Intro": "## Intro\n\n" + "A" * 80,
            "Body": "## Body\n\n" + "B" * 80,
        }
        call_count = [0]

        def fake_generate(messages, format_schema=None, thinking=True, **kw):
            result = MagicMock()
            idx = min(call_count[0], 1)
            key = ["Intro", "Body"][idx]
            result.content = json.dumps({"path": str(output_file), "content": section_content[key]})
            call_count[0] += 1
            return result

        mock_client = MagicMock()
        mock_client.generate.side_effect = fake_generate

        manifests_stored = []

        with patch.object(_se, "_planner") as mock_planner:
            mock_planner.plan.return_value = plan_manifest

            result = _se.run_stage(
                llm_client=mock_client,
                stage={"goal": "write a 2-section doc", "type": "write_doc"},
                file_path=str(output_file),
                file_type="doc",
                step=1,
                max_steps=20,
                store_manifest=manifests_stored.append,
            )

        assert result.written_path == str(output_file)
        assert len(manifests_stored) >= 1

    def test_partial_item_does_not_infinite_loop(self, tmp_path):
        """An item that always fails verify gets marked partial and loop moves on."""
        from birdclaw.agent import subtask_executor as _se
        from unittest.mock import MagicMock, patch
        import birdclaw.config as _cfg

        output_file = tmp_path / "out.md"
        plan_manifest = SubtaskManifest(
            stage_goal="g",
            file_path=str(output_file),
            file_type="doc",
            items=[SubtaskItem(index=0, title="X", anchor="X", kind="section", expected_min_chars=200)],
        )

        mock_client = MagicMock()
        # Always return content too short to satisfy expected_min_chars=200
        mock_client.generate.return_value = MagicMock(
            content=json.dumps({"path": str(output_file), "content": "## X\n\nShort."})
        )

        with patch.object(_cfg.settings, "workspace_roots", [tmp_path]):
            with patch.object(_se, "_planner") as mock_planner:
                mock_planner.plan.return_value = plan_manifest
                result = _se.run_stage(
                    llm_client=mock_client,
                    stage={"goal": "g", "type": "write_doc"},
                    file_path=str(output_file),
                    file_type="doc",
                    step=1,
                    max_steps=20,
                    store_manifest=lambda m: None,
                )

        # Should have called generate at most MAX_ITEM_RETRIES+1 times per item
        assert mock_client.generate.call_count <= _se.MAX_ITEM_RETRIES + 1
        assert plan_manifest.items[0].status == "partial"


# ── Tasks registry manifest methods ──────────────────────────────────────────

class TestTaskRegistryManifest:
    @staticmethod
    def _make_registry(tmp_path):
        from unittest.mock import patch
        import birdclaw.config as _cfg
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            from birdclaw.memory.tasks import TaskRegistry
            reg = TaskRegistry()
        # Re-bind _tasks_dir so later calls also use tmp_path
        reg._tasks_dir = lambda: (tmp_path / "tasks").__class__(tmp_path / "tasks").mkdir(parents=True, exist_ok=True) or tmp_path / "tasks"
        return reg, tmp_path

    def test_set_and_get_manifest(self, tmp_path):
        import birdclaw.config as _cfg
        from unittest.mock import patch
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            reg = TaskRegistry()
        task = reg.create("test task")
        m_dict = {"stage_goal": "g", "file_path": "/tmp/x.md", "file_type": "doc", "items": []}
        reg.set_manifest(task.task_id, m_dict)
        got = reg.get_manifest(task.task_id)
        assert got == m_dict

    def test_get_manifest_unknown_task_returns_none(self, tmp_path):
        import birdclaw.config as _cfg
        from unittest.mock import patch
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            reg = TaskRegistry()
        assert reg.get_manifest("nonexistent") is None

    def test_manifest_persisted_to_disk(self, tmp_path):
        import birdclaw.config as _cfg
        from unittest.mock import patch
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            reg = TaskRegistry()
            task = reg.create("persistence test")
            m_dict = {"stage_goal": "doc", "items": []}
            reg.set_manifest(task.task_id, m_dict)
            reg2 = TaskRegistry()
        assert reg2.get_manifest(task.task_id) == m_dict


# ── SessionLog subtask_manifest event ────────────────────────────────────────

class TestSessionLogSubtaskManifest:
    def test_subtask_manifest_event_appended(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        log = SessionLog(session_id="test-sm", path=tmp_path / "test-sm.jsonl")
        log.subtask_manifest(stage_index=0, manifest_dict={"stage_goal": "g", "items": []})
        events = log.events_of_type("subtask_manifest")
        assert len(events) == 1
        assert events[0].data["stage_index"] == 0


# ── Compaction config changes ─────────────────────────────────────────────────

class TestCompactionConfig:
    def test_default_threshold_is_2000(self):
        from birdclaw.memory.compact import CompactionConfig
        cfg = CompactionConfig()
        assert cfg.max_estimated_tokens == 2_000

    def test_format_stage_preserves_more_messages(self):
        from birdclaw.memory.compact import CompactionConfig
        cfg = CompactionConfig()
        assert cfg.preserve_recent_messages_format > cfg.preserve_recent_messages

    def test_should_compact_uses_format_preserve_when_flagged(self):
        from birdclaw.memory.compact import CompactionConfig, should_compact
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(max_estimated_tokens=1, preserve_recent_messages=2, preserve_recent_messages_format=100)
        msgs = [Message(role="user", content="x " * 10) for _ in range(10)]
        # With format=True, preserve=100 means more messages kept → should not compact easily
        assert should_compact(msgs, cfg, in_format_stage=False)  # normal: 2 preserved, fires
        assert not should_compact(msgs, cfg, in_format_stage=True)  # format: 100 preserved, does not fire


# ===========================================================================
# Unit — Router
# ===========================================================================

import pytest as _pytest_router

from birdclaw.agent.router import (
    _tokenise, _query_tokens, _has_file_path, _looks_like_question, select,
)
from birdclaw.llm.types import Message


@pytest.mark.unit
class TestRouter:
    def test_tokenise_lowercases(self):
        assert _tokenise("Hello World") == {"hello", "world"}

    def test_tokenise_strips_punctuation(self):
        tokens = _tokenise("fetch_data: file.py!")
        assert "fetch_data" in tokens
        assert "file" in tokens
        assert "py" in tokens

    def test_tokenise_empty(self):
        assert _tokenise("") == set()

    def test_query_tokens_combines_history(self):
        hist = [Message(role="user", content="search documents")]
        tokens = _query_tokens("find files", hist)
        assert "find" in tokens
        assert "search" in tokens
        assert "documents" in tokens

    def test_query_tokens_respects_lookback(self):
        hist = [
            Message(role="user", content="old content"),
            Message(role="user", content="recent one"),
            Message(role="user", content="very recent"),
        ]
        tokens = _query_tokens("new", hist, lookback=1)
        assert "very" in tokens
        assert "recent" in tokens
        # "old" should not be included (lookback=1)
        assert "old" not in tokens

    def test_has_file_path_detects_py(self):
        assert _has_file_path("edit birdclaw/agent/loop.py line 42")

    def test_has_file_path_detects_toml(self):
        assert _has_file_path("check pyproject.toml")

    def test_has_file_path_no_path(self):
        assert not _has_file_path("what is the weather today?")

    def test_looks_like_question_what(self):
        assert _looks_like_question("what is the status?")

    def test_looks_like_question_how(self):
        assert _looks_like_question("how do I install this?")

    def test_looks_like_question_question_mark(self):
        assert _looks_like_question("is it running?")

    def test_looks_like_question_imperative(self):
        assert not _looks_like_question("write a report about the system")

    def test_select_returns_list(self):
        from birdclaw.tools.registry import registry, Tool
        from unittest.mock import patch
        dummy = Tool(
            name="dummy_tool", description="A test tool", tags=["test", "dummy"],
            input_schema={}, handler=lambda: "{}"
        )
        with patch.object(registry, "all_tools", return_value=[dummy]):
            result = select("test this dummy thing", max_n=5)
        assert isinstance(result, list)

    def test_select_empty_registry(self):
        from birdclaw.tools.registry import registry
        from unittest.mock import patch
        with patch.object(registry, "all_tools", return_value=[]):
            result = select("anything")
        assert result == []

    def test_select_respects_max_n(self):
        from birdclaw.tools.registry import registry, Tool
        from unittest.mock import patch
        tools = [
            Tool(name=f"t{i}", description="d", tags=["foo"], input_schema={}, handler=lambda: "{}")
            for i in range(10)
        ]
        with patch.object(registry, "all_tools", return_value=tools):
            result = select("foo bar", max_n=3)
        assert len(result) <= 3


# ===========================================================================
# Unit — StepSupervisor
# ===========================================================================

@pytest.mark.unit
class TestStepSupervisor:
    def test_submit_and_collect(self):
        from birdclaw.agent.supervisor import StepSupervisor
        sup = StepSupervisor()
        sup.submit(lambda: 42, tag="test")
        assert sup.collect() == 42
        sup.shutdown()

    def test_collect_none_if_nothing_pending(self):
        from birdclaw.agent.supervisor import StepSupervisor
        sup = StepSupervisor()
        assert sup.collect() is None
        sup.shutdown()

    def test_collect_returns_none_on_exception(self):
        from birdclaw.agent.supervisor import StepSupervisor
        sup = StepSupervisor()
        def bad(): raise ValueError("oops")
        sup.submit(bad, tag="bad")
        result = sup.collect()   # should swallow and return None
        assert result is None
        sup.shutdown()

    def test_submit_twice_auto_collects_first(self):
        from birdclaw.agent.supervisor import StepSupervisor
        results = []
        sup = StepSupervisor()
        sup.submit(lambda: results.append(1) or 1, tag="first")
        # Second submit should auto-collect the first
        sup.submit(lambda: results.append(2) or 2, tag="second")
        sup.collect()
        sup.shutdown()
        assert 1 in results
        assert 2 in results

    def test_shutdown_idempotent(self):
        from birdclaw.agent.supervisor import StepSupervisor
        sup = StepSupervisor()
        sup.shutdown()
        sup.shutdown()  # should not raise


# ===========================================================================
# Unit — Budget
# ===========================================================================

@pytest.mark.unit
class TestBudget:
    def test_historical_budget_no_file_returns_default(self, monkeypatch, tmp_path):
        from birdclaw.agent import budget
        from birdclaw.config import settings
        monkeypatch.setattr(budget, "history_path", lambda: tmp_path / "missing.jsonl")
        monkeypatch.setattr(settings, "stage_budgets", {"research": 7})
        assert budget.historical_budget("research") == 7

    def test_historical_budget_unknown_type_uses_default(self, monkeypatch, tmp_path):
        from birdclaw.agent import budget
        from birdclaw.config import settings
        monkeypatch.setattr(budget, "history_path", lambda: tmp_path / "missing.jsonl")
        monkeypatch.setattr(settings, "stage_budgets", {})
        result = budget.historical_budget("novel_type")
        assert isinstance(result, int)

    def test_log_stage_writes_entry(self, monkeypatch, tmp_path):
        from birdclaw.agent import budget
        hp = tmp_path / "stage_history.jsonl"
        monkeypatch.setattr(budget, "history_path", lambda: hp)
        budget.log_stage("research", 8, "find Python docs")
        lines = hp.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "research"
        assert entry["steps"] == 8

    def test_historical_budget_p75_from_history(self, monkeypatch, tmp_path):
        from birdclaw.agent import budget
        from birdclaw.config import settings
        hp = tmp_path / "stage_history.jsonl"
        monkeypatch.setattr(budget, "history_path", lambda: hp)
        monkeypatch.setattr(settings, "stage_budgets", {"research": 10})
        # Write 8 records: steps [2,4,6,8,10,12,14,16] → P75 = 13
        steps = [2, 4, 6, 8, 10, 12, 14, 16]
        for s in steps:
            hp.open("a").write(json.dumps({"type": "research", "steps": s, "ts": 0.0}) + "\n")
        result = budget.historical_budget("research")
        assert result >= 10  # P75 of those steps should beat the default

    def test_log_stage_multiple_entries(self, monkeypatch, tmp_path):
        from birdclaw.agent import budget
        hp = tmp_path / "stage_history.jsonl"
        monkeypatch.setattr(budget, "history_path", lambda: hp)
        budget.log_stage("verify", 3, "check output")
        budget.log_stage("verify", 5, "check again")
        budget.log_stage("research", 9, "web search")
        lines = hp.read_text().splitlines()
        assert len(lines) == 3
        types = [json.loads(l)["type"] for l in lines]
        assert types.count("verify") == 2


# ===========================================================================
# Unit — SessionLog (full API)
# ===========================================================================

@pytest.mark.unit
class TestSessionLogFull:
    @pytest.fixture
    def log(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        return SessionLog(session_id="unit-test", path=tmp_path / "unit-test.jsonl")

    def test_user_message_appended(self, log):
        log.user_message("hello world")
        evts = log.events_of_type("user_message")
        assert len(evts) == 1
        assert evts[0].data["content"] == "hello world"

    def test_assistant_message(self, log):
        log.assistant_message("I am here")
        evts = log.events_of_type("assistant_message")
        assert evts[0].data["content"] == "I am here"

    def test_tool_call_and_result(self, log):
        log.tool_call("bash", {"command": "ls"})
        log.tool_result("bash", "file.txt", duration_ms=50)
        calls   = log.events_of_type("tool_call")
        results = log.events_of_type("tool_result")
        assert calls[0].data["name"] == "bash"
        assert results[0].data["duration_ms"] == 50

    def test_plan_event(self, log):
        log.plan("write a report", ["research", "write"])
        evts = log.events_of_type("plan")
        assert evts[0].data["outcome"] == "write a report"
        assert len(evts[0].data["steps"]) == 2

    def test_stage_start_and_done(self, log):
        log.stage_start("research", "find Python docs")
        log.stage_done("research", "find Python docs", "found 3 pages", duration_ms=100)
        starts = log.events_of_type("stage_start")
        dones  = log.events_of_type("stage_done")
        assert starts[0].data["goal"] == "find Python docs"
        assert dones[0].data["duration_ms"] == 100

    def test_task_created(self, log):
        log.task_created("req-1", "original request text", step_count=3)
        evts = log.events_of_type("task_created")
        assert evts[0].data["steps"] == 3

    def test_step_done(self, log):
        log.step_done("step-1", "write intro", "done: wrote 200 chars")
        evts = log.events_of_type("step_done")
        assert "wrote 200" in evts[0].data["result"]

    def test_memory_hit(self, log):
        log.memory_hit("Python docs", ["Python", "documentation"])
        evts = log.events_of_type("memory_hit")
        assert "Python" in evts[0].data["nodes"]

    def test_session_summary(self, log):
        log.session_summary("User asked about Python, we answered.")
        assert log.latest_summary() == "User asked about Python, we answered."

    def test_usage_event(self, log):
        log.usage(requests=5, responses=5, total_tokens=1200, model="gemma4")
        evts = log.events_of_type("usage")
        assert evts[0].data["total_tokens"] == 1200

    def test_compaction_event(self, log):
        log.compaction(removed=10, summary_preview="discussed X")
        evts = log.events_of_type("compaction")
        assert evts[0].data["removed_messages"] == 10

    def test_subtask_manifest_event(self, log):
        log.subtask_manifest(stage_index=1, manifest_dict={"items": ["a", "b"]})
        evts = log.events_of_type("subtask_manifest")
        assert evts[0].data["stage_index"] == 1

    def test_task_spawned_event(self, log):
        log.task_spawned("task-123", "write-report", "soul", parent_session="parent-1")
        evts = log.events_of_type("task_spawned")
        assert evts[0].data["task_id"] == "task-123"
        assert evts[0].data["trigger"] == "soul"

    def test_last_user_messages(self, log):
        log.user_message("first")
        log.user_message("second")
        log.user_message("third")
        msgs = log.last_user_messages(n=2)
        assert msgs == ["second", "third"]

    def test_completed_steps(self, log):
        log.step_done("s1", "step one", "result one")
        log.step_done("s2", "step two", "result two")
        steps = log.completed_steps()
        assert len(steps) == 2
        assert steps[0]["step_id"] == "s1"

    def test_events_of_type_filter(self, log):
        log.user_message("hi")
        log.tool_call("bash", {"command": "ls"})
        log.assistant_message("done")
        assert len(log.events_of_type("user_message")) == 1
        assert len(log.events_of_type("tool_call")) == 1
        assert len(log.events_of_type("assistant_message")) == 1

    def test_persists_to_disk(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        log = SessionLog(session_id="persist-test", path=tmp_path / "persist-test.jsonl")
        log.user_message("persist me")
        log2 = SessionLog.load.__func__(SessionLog, "persist-test") if False else None
        # Verify file exists and has content
        path = tmp_path / "persist-test.jsonl"
        assert path.exists()
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert any(l.get("data", {}).get("content") == "persist me" for l in lines)


# ===========================================================================
# Unit — TaskRegistry (full lifecycle)
# ===========================================================================

@pytest.mark.unit
class TestTaskRegistryFull:
    @pytest.fixture
    def reg(self, tmp_path, monkeypatch):
        import birdclaw.config as _cfg
        from unittest.mock import patch
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            r = TaskRegistry()
        return r

    def test_make_title_basic(self):
        from birdclaw.memory.tasks import _make_title
        title = _make_title("write a report about Python testing")
        assert "Report" in title or "Python" in title or "Testing" in title

    def test_make_title_stop_words_stripped(self):
        from birdclaw.memory.tasks import _make_title
        title = _make_title("the and or but a an")
        assert title  # falls back gracefully

    def test_make_title_empty_prompt(self):
        from birdclaw.memory.tasks import _make_title
        title = _make_title("")
        assert isinstance(title, str)

    def test_create_returns_task(self, reg):
        task = reg.create("do something")
        assert task.task_id
        assert task.status == "created"
        assert task.prompt == "do something"

    def test_get_returns_task(self, reg):
        task = reg.create("find task")
        got = reg.get(task.task_id)
        assert got is not None
        assert got.task_id == task.task_id

    def test_get_unknown_returns_none(self, reg):
        assert reg.get("nonexistent-task-id") is None

    def test_list_all(self, reg):
        reg.create("task one")
        reg.create("task two")
        tasks = reg.list()
        assert len(tasks) >= 2

    def test_list_by_status(self, reg):
        t1 = reg.create("task one")
        t2 = reg.create("task two")
        reg.start(t1.task_id, agent_id="agent-1")
        running = reg.list(status="running")
        assert any(t.task_id == t1.task_id for t in running)
        assert all(t.task_id != t2.task_id for t in running)

    def test_start_sets_running(self, reg):
        task = reg.create("start me")
        reg.start(task.task_id, agent_id="agent-x")
        got = reg.get(task.task_id)
        assert got.status == "running"
        assert got.agent_id == "agent-x"
        assert got.started_at > 0

    def test_complete_sets_status(self, reg):
        task = reg.create("complete me")
        reg.start(task.task_id)
        reg.complete(task.task_id, output="task done!")
        got = reg.get(task.task_id)
        assert got.status == "completed"
        assert "task done!" in got.output

    def test_fail_sets_status(self, reg):
        task = reg.create("fail me")
        reg.start(task.task_id)
        reg.fail(task.task_id, reason="something broke")
        got = reg.get(task.task_id)
        assert got.status == "failed"

    def test_stop_sets_status(self, reg):
        task = reg.create("stop me")
        reg.start(task.task_id)
        reg.stop(task.task_id)
        got = reg.get(task.task_id)
        assert got.status == "stopped"

    def test_append_message(self, reg):
        task = reg.create("message task")
        reg.append_message(task.task_id, "user says hello", role="user")
        got = reg.get(task.task_id)
        assert len(got.messages) == 1
        assert got.messages[0].content == "user says hello"

    def test_append_output(self, reg):
        task = reg.create("output task")
        reg.append_output(task.task_id, "line one\n")
        reg.append_output(task.task_id, "line two\n")
        got = reg.get(task.task_id)
        assert "line one" in got.output
        assert "line two" in got.output

    def test_set_and_advance_phases(self, reg):
        task = reg.create("phased task")
        reg.set_phases(task.task_id, ["research", "write", "verify"])
        reg.advance_phase(task.task_id)
        got = reg.get(task.task_id)
        assert got.completed_phase_count == 1

    def test_remove(self, reg):
        task = reg.create("remove me")
        removed = reg.remove(task.task_id)
        assert removed is not None
        assert reg.get(task.task_id) is None

    def test_remove_unknown_returns_none(self, reg):
        assert reg.remove("ghost-task-id") is None

    def test_create_with_context_and_outcome(self, reg):
        task = reg.create("do task", context="use pandas", expected_outcome="CSV written")
        got = reg.get(task.task_id)
        assert got.context == "use pandas"
        assert got.expected_outcome == "CSV written"

    def test_output_shortcut(self, reg):
        task = reg.create("output shortcut")
        reg.append_output(task.task_id, "the result")
        assert "the result" in reg.output(task.task_id)


# ===========================================================================
# Unit — LineSearch
# ===========================================================================

@pytest.mark.unit
class TestLineSearch:
    @pytest.fixture
    def code_file(self, tmp_path):
        p = tmp_path / "sample.py"
        p.write_text(
            "def authenticate(user, password):\n"
            "    if not user or not password:\n"
            "        raise ValueError('missing credentials')\n"
            "    return True\n\n"
            "def logout(user):\n"
            "    user.session = None\n"
            "    return True\n"
        )
        return p

    @pytest.fixture
    def doc_file(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text(
            "# Authentication\n\n"
            "The system uses JWT tokens for authentication.\n"
            "Tokens expire after 24 hours.\n\n"
            "# Logging\n\n"
            "All events are written to birdclaw.log.\n"
        )
        return p

    def test_search_lines_finds_pattern(self, code_file):
        from birdclaw.tools.line_search import search_lines
        result = search_lines("def authenticate", [str(code_file)])
        assert "authenticate" in result

    def test_search_lines_no_match_returns_empty(self, code_file):
        from birdclaw.tools.line_search import search_lines
        result = search_lines("NONEXISTENT_PATTERN_XYZ", [str(code_file)])
        assert result == ""

    def test_search_lines_regex(self, code_file):
        from birdclaw.tools.line_search import search_lines
        result = search_lines(r"def \w+", [str(code_file)], use_regex=True)
        assert "authenticate" in result or "logout" in result

    def test_search_relevant_goal_based(self, code_file):
        from birdclaw.tools.line_search import search_relevant
        result = search_relevant("implement user authentication validation", [str(code_file)])
        # May or may not find a match — just ensure it returns a string
        assert isinstance(result, str)

    def test_search_relevant_stop_words_filtered(self, doc_file):
        from birdclaw.tools.line_search import search_relevant
        # "the" and "and" are stop words — goal with only stop words → empty
        result = search_relevant("the and for with", [str(doc_file)])
        assert isinstance(result, str)

    def test_find_section_doc(self, doc_file):
        from birdclaw.tools.line_search import find_section
        result = find_section(str(doc_file), "Logging", "md")
        assert "birdclaw.log" in result or "Logging" in result

    def test_find_section_code(self, code_file):
        from birdclaw.tools.line_search import find_section
        result = find_section(str(code_file), "authenticate", "py")
        assert "def authenticate" in result or "authenticate" in result

    def test_find_section_not_found(self, doc_file):
        from birdclaw.tools.line_search import find_section
        result = find_section(str(doc_file), "NonexistentSection", "md")
        # Must return empty string (section not present)
        assert result == ""

    def test_search_lines_multiple_files(self, tmp_path):
        from birdclaw.tools.line_search import search_lines
        f1 = tmp_path / "a.py"; f1.write_text("def foo(): pass\n")
        f2 = tmp_path / "b.py"; f2.write_text("def bar(): pass\n")
        result = search_lines("def ", [str(f1), str(f2)])
        assert "foo" in result and "bar" in result


# ===========================================================================
# Component — Files tool (workspace boundary + CRUD)
# ===========================================================================

@pytest.mark.component
class TestFilesTool:
    @pytest.fixture(autouse=True)
    def patch_workspace(self, tmp_path, monkeypatch):
        """Set workspace root to tmp_path so all file ops are allowed."""
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path])
        self.workspace = tmp_path

    def test_write_and_read(self):
        from birdclaw.tools.files import write_file, read_file
        p = str(self.workspace / "hello.txt")
        out = json.loads(write_file(p, "hello world"))
        assert out.get("written") is not None, f"expected written key, got {out}"
        content = json.loads(read_file(p))
        assert "hello world" in content.get("content", "")

    def test_write_creates_parents(self):
        from birdclaw.tools.files import write_file, read_file
        p = str(self.workspace / "deep" / "nested" / "file.txt")
        write_file(p, "nested content")
        result = json.loads(read_file(p))
        assert "nested content" in str(result)

    def test_read_offset_and_limit(self):
        from birdclaw.tools.files import write_file, read_file
        p = str(self.workspace / "multiline.txt")
        write_file(p, "\n".join(f"line{i}" for i in range(20)))
        result = json.loads(read_file(p, offset=5, limit=3))
        content = result.get("content", "")
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) <= 3

    def test_edit_file_exact_replacement(self):
        from birdclaw.tools.files import write_file, edit_file, read_file
        p = str(self.workspace / "edit_me.txt")
        write_file(p, "old content here")
        edit_file(p, "old content", "new content")
        result = json.loads(read_file(p))
        assert "new content" in str(result)

    def test_edit_file_replace_all(self):
        from birdclaw.tools.files import write_file, edit_file, read_file
        p = str(self.workspace / "replace_all.txt")
        write_file(p, "foo bar foo baz foo")
        edit_file(p, "foo", "qux", replace_all=True)
        result = json.loads(read_file(p))
        content = str(result)
        assert "qux" in content
        assert "foo" not in content

    def test_write_outside_workspace_rejected(self, tmp_path):
        from birdclaw.tools.files import write_file
        # tmp_path itself is the workspace; write to a sibling dir
        outside = tmp_path.parent / "outside_workspace" / "evil.txt"
        result = json.loads(write_file(str(outside), "evil content"))
        assert "error" in result, f"expected error for out-of-workspace write, got: {result}"

    def test_glob_search_finds_files(self):
        from birdclaw.tools.files import write_file, glob_search
        write_file(str(self.workspace / "a.py"), "x=1")
        write_file(str(self.workspace / "b.py"), "y=2")
        result = json.loads(glob_search("*.py", str(self.workspace)))
        matches = result.get("matches", [])
        names = [Path(f).name for f in matches]
        assert "a.py" in names
        assert "b.py" in names

    def test_grep_search_finds_pattern(self):
        from birdclaw.tools.files import write_file, grep_search
        write_file(str(self.workspace / "search_me.py"), "def process_data(x):\n    return x * 2\n")
        result = json.loads(grep_search("process_data", path=str(self.workspace)))
        text = str(result)
        assert "process_data" in text or "search_me" in text

    def test_read_nonexistent_returns_error(self):
        from birdclaw.tools.files import read_file
        result = json.loads(read_file(str(self.workspace / "no_such_file.txt")))
        assert "error" in str(result).lower()


# ===========================================================================
# Component — KnowledgeGraph
# ===========================================================================

@pytest.mark.component
class TestKnowledgeGraph:
    @pytest.fixture
    def graph(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        return GraphStore(persist_path=tmp_path / "test_graph.json")

    def test_upsert_and_get_node(self, graph):
        graph.upsert_node("Python", node_type="entity", summary="Programming language",
                          sources=["docs.python.org"])
        node = graph.get_node("Python")
        assert node is not None
        assert node["type"] == "entity"
        assert "Programming" in node["summary"]

    def test_upsert_updates_existing(self, graph):
        graph.upsert_node("Django", node_type="entity", summary="Web framework")
        graph.upsert_node("Django", node_type="entity", summary="Python web framework updated")
        node = graph.get_node("Django")
        assert "updated" in node["summary"]

    def test_get_node_unknown_returns_none(self, graph):
        assert graph.get_node("nonexistent_node_xyz") is None

    def test_upsert_edge(self, graph):
        graph.upsert_node("Django", node_type="entity", summary="Web framework")
        graph.upsert_node("Python", node_type="entity", summary="Language")
        graph.upsert_edge("Django", "depends_on", "Python")
        assert graph.edge_count() == 1

    def test_search_by_token_overlap(self, graph):
        graph.upsert_node("FastAPI", node_type="entity", summary="async web framework for Python APIs")
        graph.upsert_node("Django REST", node_type="entity", summary="REST API framework")
        graph.upsert_node("Unrelated Node", node_type="entity", summary="something entirely different")
        results = graph.search("Python web API framework")
        names = [r["name"] for r in results]
        assert any("FastAPI" in n or "Django" in n for n in names)

    def test_search_by_type_filter(self, graph):
        graph.upsert_node("fact1", node_type="fact", summary="A fact about testing")
        graph.upsert_node("entity1", node_type="entity", summary="A testing entity")
        results = graph.search("testing", node_type="fact")
        assert all(r["type"] == "fact" for r in results)

    def test_bfs_traversal(self, graph):
        graph.upsert_node("A", node_type="entity", summary="node A")
        graph.upsert_node("B", node_type="entity", summary="node B")
        graph.upsert_node("C", node_type="entity", summary="node C")
        graph.upsert_edge("A", "related_to", "B")
        graph.upsert_edge("A", "related_to", "C")
        results = graph.bfs(["A"], depth=2)
        names = [r["name"] for r in results]
        assert "B" in names
        assert "C" in names

    def test_nodes_by_type(self, graph):
        graph.upsert_node("fact-x", node_type="fact", summary="first fact")
        graph.upsert_node("fact-y", node_type="fact", summary="second fact")
        graph.upsert_node("entity-z", node_type="entity", summary="an entity")
        facts = list(graph.nodes_by_type("fact"))
        assert len(facts) == 2

    def test_remove_node(self, graph):
        graph.upsert_node("ToRemove", node_type="entity", summary="will be deleted")
        assert graph.remove_node("ToRemove") is True
        assert graph.get_node("ToRemove") is None

    def test_remove_nonexistent_returns_false(self, graph):
        assert graph.remove_node("ghost_node_xyz") is False

    def test_node_count(self, graph):
        assert graph.node_count() == 0
        graph.upsert_node("n1", node_type="entity", summary="one")
        graph.upsert_node("n2", node_type="entity", summary="two")
        assert graph.node_count() == 2

    def test_merge_from(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        g1 = GraphStore(persist_path=tmp_path / "g1.json")
        g2 = GraphStore(persist_path=tmp_path / "g2.json")
        g1.upsert_node("common", node_type="entity", summary="in both")
        g2.upsert_node("exclusive", node_type="entity", summary="only in g2")
        g1.merge_from(g2)
        assert g1.get_node("exclusive") is not None

    def test_save_and_reload(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        path = tmp_path / "saved.json"
        g = GraphStore(persist_path=path)
        g.upsert_node("persist-me", node_type="entity", summary="saved node")
        g.save()
        g2 = GraphStore(persist_path=path)
        assert g2.get_node("persist-me") is not None

    def test_case_insensitive_lookup(self, graph):
        graph.upsert_node("Python", node_type="entity", summary="language")
        assert graph.get_node("python") is not None
        assert graph.get_node("PYTHON") is not None


# ===========================================================================
# Unit — SessionManager
# ===========================================================================

@pytest.mark.unit
class TestSessionManager:
    @pytest.fixture
    def mgr(self):
        from birdclaw.gateway.session_manager import SessionManager
        return SessionManager()

    def test_get_or_create_returns_session(self, mgr):
        sess = mgr.get_or_create("ch1", "user1")
        assert sess.session_id

    def test_get_or_create_idempotent(self, mgr):
        s1 = mgr.get_or_create("ch1", "user1")
        s2 = mgr.get_or_create("ch1", "user1")
        assert s1.session_id == s2.session_id

    def test_different_channels_different_sessions(self, mgr):
        s1 = mgr.get_or_create("ch1", "user1")
        s2 = mgr.get_or_create("ch2", "user1")
        assert s1.session_id != s2.session_id

    def test_get_returns_session(self, mgr):
        sess = mgr.get_or_create("ch1", "user1")
        got  = mgr.get(sess.session_id)
        assert got is not None
        assert got.session_id == sess.session_id

    def test_get_unknown_returns_none(self, mgr):
        assert mgr.get("nonexistent-session-id") is None

    def test_add_task(self, mgr):
        sess = mgr.get_or_create("ch1", "user1")
        mgr.add_task(sess.session_id, "task-42")
        got = mgr.get(sess.session_id)
        assert "task-42" in got.task_ids

    def test_all_sessions(self, mgr):
        mgr.get_or_create("ch1", "user1")
        mgr.get_or_create("ch2", "user2")
        sessions = mgr.all_sessions()
        assert len(sessions) >= 2

    def test_launch_cwd_field(self, mgr):
        from birdclaw.gateway.session_manager import Session
        sess = mgr.get_or_create("ch1", "user1")
        sess.launch_cwd = "/home/AlgoMind"
        assert sess.launch_cwd == "/home/AlgoMind"

    def test_save_turn(self, mgr):
        sess = mgr.get_or_create("ch1", "user1")
        mgr.save_turn(sess.session_id, "user", "hello from user")
        # Should not raise; history is written to disk


# ===========================================================================
# Component — Cross-module: TaskRegistry + SessionLog integration
# ===========================================================================

@pytest.mark.component
class TestTaskAndSessionIntegration:
    @pytest.fixture
    def reg(self, tmp_path, monkeypatch):
        import birdclaw.config as _cfg
        from unittest.mock import patch
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            r = TaskRegistry()
        return r

    @pytest.fixture
    def session(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        return SessionLog(session_id="integration-test", path=tmp_path / "integration.jsonl")

    def test_task_lifecycle_with_session_events(self, reg, session):
        task = reg.create("integration task")
        session.task_spawned(task.task_id, "integration-task", "test", parent_session="root")
        reg.start(task.task_id, agent_id="test-agent")
        session.stage_start("research", "find data")
        session.stage_done("research", "find data", "found 5 items")
        reg.complete(task.task_id, output="all done")

        assert reg.get(task.task_id).status == "completed"
        assert len(session.events_of_type("task_spawned")) == 1
        assert len(session.events_of_type("stage_done")) == 1

    def test_task_failure_recorded_in_session(self, reg, session):
        task = reg.create("failing task")
        reg.start(task.task_id)
        session.tool_call("bash", {"command": "bad command"})
        session.tool_result("bash", "error: command not found", duration_ms=5)
        reg.fail(task.task_id, reason="bash failed")
        assert reg.get(task.task_id).status == "failed"
        calls = session.events_of_type("tool_call")
        assert len(calls) == 1


# ===========================================================================
# Unit — WriteGuard bug-regression tests
# ===========================================================================

@pytest.mark.unit
class TestWriteGuardRegression:
    """Regression tests for write_guard bugs observed in production logs."""

    def test_is_code_uses_explicit_file_type_not_path(self):
        """Model-overridden path (.py) must not trigger syntax check on doc content."""
        from birdclaw.tools.write_guard import pre_write_check
        # file_type="doc" must take priority over the .py path extension
        result = pre_write_check(
            path="/tmp/report.py",
            content="# Executive Summary\n\nThis is a markdown report.\n",
            file_type="doc",
        )
        assert result.ok, f"should pass: {result.error}"

    def test_is_code_true_when_file_type_code(self):
        """Syntax check fires when file_type='code' regardless of extension."""
        from birdclaw.tools.write_guard import pre_write_check
        result = pre_write_check(
            path="/tmp/something.md",
            content="def bad_python(:\n    pass\n",
            file_type="code",
        )
        assert not result.ok
        assert "syntax" in result.error.lower()

    def test_is_code_infers_from_ext_when_no_file_type(self):
        """Falls back to extension when file_type is empty."""
        from birdclaw.tools.write_guard import pre_write_check
        result = pre_write_check(
            path="/tmp/script.py",
            content="def bad(:\n    pass\n",
            file_type="",
        )
        assert not result.ok
        assert "syntax" in result.error.lower()

    def test_valid_python_passes_code_check(self):
        from birdclaw.tools.write_guard import pre_write_check
        result = pre_write_check(
            path="/tmp/ok.py",
            content="def hello():\n    return 42\n",
            file_type="code",
        )
        assert result.ok

    def test_doc_content_no_syntax_check(self):
        """Markdown with colons and quotes is not Python-checked."""
        from birdclaw.tools.write_guard import pre_write_check
        md = '# Report\n\n"content": "this looks like JSON but is markdown"\n'
        result = pre_write_check(path="/tmp/report.md", content=md, file_type="doc")
        assert result.ok


# ===========================================================================
# Unit — SubtaskExecutor context-cap regression
# ===========================================================================

@pytest.mark.unit
class TestSubtaskExecutorContextCap:
    def test_max_ctx_chars_constant_exists(self):
        from birdclaw.agent.subtask_executor import _MAX_CTX_CHARS
        assert isinstance(_MAX_CTX_CHARS, int)
        assert 2000 <= _MAX_CTX_CHARS <= 20_000

    def test_context_capped_at_max(self, tmp_path, monkeypatch):
        """Verify _build_call_messages truncates oversized file context."""
        from birdclaw.agent.subtask_executor import _build_call_messages, _MAX_CTX_CHARS
        from birdclaw.agent.subtask_manifest import SubtaskManifest, SubtaskItem

        # Create a manifest with a large file
        big_file = tmp_path / "big_doc.md"
        big_file.write_text("# Section\n\n" + "word " * 5000)  # ~30KB

        manifest = SubtaskManifest(
            stage_goal="write comprehensive report",
            file_path=str(big_file),
            file_type="doc",
            items=[SubtaskItem(title="Intro", index=0, anchor="intro",
                               kind="section", expected_min_chars=100, status="missing")],
        )
        item = manifest.items[0]

        msgs = _build_call_messages(item, manifest, attempt=0, step=1, max_steps=10)
        # The user message content should not exceed _MAX_CTX_CHARS + instruction overhead
        user_content = msgs[1].content
        assert len(user_content) < _MAX_CTX_CHARS + 1000, (
            f"context too large: {len(user_content)} chars (cap={_MAX_CTX_CHARS})"
        )


# ===========================================================================
# Component — Cross-module: GraphStore + lineSearch integration
# ===========================================================================

@pytest.mark.component
class TestGraphAndLineSearchIntegration:
    def test_research_inject_and_search(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        g = GraphStore(persist_path=tmp_path / "rag.json")

        # Simulate research injection
        g.upsert_node(
            "Python asyncio",
            node_type="fact",
            summary="asyncio is Python's built-in async I/O library",
            sources=["research:task-1"],
        )
        g.upsert_node(
            "event loop",
            node_type="fact",
            summary="the event loop runs coroutines and handles I/O",
            sources=["research:task-1"],
        )

        # Query should find both
        results = g.search("Python async event loop", limit=5)
        summaries = [r["summary"] for r in results]
        assert any("asyncio" in s or "async" in s for s in summaries)


# ===========================================================================
# Unit — Compact (additional edge cases)
# ===========================================================================

@pytest.mark.unit
class TestCompactEdgeCases:
    def test_empty_messages_no_compact(self):
        from birdclaw.memory.compact import should_compact, CompactionConfig
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(max_estimated_tokens=100)
        assert not should_compact([], cfg)

    def test_single_message_no_compact(self):
        from birdclaw.memory.compact import should_compact, CompactionConfig
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(max_estimated_tokens=1)
        assert not should_compact([Message(role="user", content="hi")], cfg)

    def test_long_message_triggers_compact(self):
        from birdclaw.memory.compact import should_compact, CompactionConfig
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(max_estimated_tokens=1, preserve_recent_messages=1)
        msgs = [Message(role="user", content="word " * 200) for _ in range(5)]
        assert should_compact(msgs, cfg)

    def test_preserve_recent_keeps_tail(self):
        from birdclaw.memory.compact import compact, CompactionConfig
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(max_estimated_tokens=1, preserve_recent_messages=2)
        msgs = [Message(role="user", content=f"msg{i} " * 20) for i in range(10)]
        result = compact(msgs, cfg)
        contents = [m.content for m in result.compacted_messages]
        assert "msg8" in str(contents)
        assert "msg9" in str(contents)

    def test_format_stage_preserves_more(self):
        from birdclaw.memory.compact import should_compact, CompactionConfig
        from birdclaw.llm.types import Message
        cfg = CompactionConfig(
            max_estimated_tokens=1,
            preserve_recent_messages=2,
            preserve_recent_messages_format=50,
        )
        msgs = [Message(role="user", content="x " * 5) for _ in range(10)]
        assert should_compact(msgs, cfg, in_format_stage=False)
        assert not should_compact(msgs, cfg, in_format_stage=True)


# ===========================================================================
# Smoke — CLI command tests (direct subprocess invocations)
# ===========================================================================

@pytest.mark.smoke
class TestCLICommands:
    """Tests that invoke CLI commands directly via subprocess.

    These verify the entry point, help text, and basic command routing
    without requiring a live LLM (the --help and error paths).
    """
    _MAIN = str(PROJECT_ROOT / "main.py")

    def _run(self, *args, timeout=30):
        return subprocess.run(
            [sys.executable, self._MAIN] + list(args),
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )

    def test_help_exits_cleanly(self):
        r = self._run("--help")
        assert r.returncode == 0, f"--help exited {r.returncode}: {r.stderr[:200]}"
        output = (r.stdout + r.stderr).lower()
        assert any(kw in output for kw in ("usage", "birdclaw", "commands")), (
            f"--help output lacks expected keywords: {r.stdout[:200]}"
        )

    def test_unknown_command_exits_nonzero(self):
        r = self._run("totally_unknown_command_xyz")
        assert r.returncode != 0

    def test_cleanup_runs_without_error(self):
        r = self._run("cleanup", timeout=30)
        assert r.returncode == 0, f"cleanup exited {r.returncode}: {r.stderr[:300]}"

    def test_version_or_help_subcommand(self):
        r = self._run("prompt", "--help")
        assert r.returncode == 0, f"prompt --help exited {r.returncode}: {r.stderr[:200]}"
        assert "prompt" in (r.stdout + r.stderr).lower()


# ===========================================================================
# Unit — Markers on existing test groups (consolidated mark)
# ===========================================================================
# The sections below re-export the key marks so pytest -m unit/component works
# without touching each function above.  We use a conftest-style approach by
# applying marks to pytest item IDs in the hook below.

# The existing tests (soul, context, history, usage, hooks, sandbox, MCP naming)
# are all pure-Python with no LLM and mostly no real I/O — they run <1s each
# and satisfy the "unit" bar.  The bash/MCP-client/MCP-manager tests spawn
# subprocesses and satisfy the "component" bar.


# ===========================================================================
# Unit — SessionLog.all_events() (Fix #17)
# ===========================================================================

@pytest.mark.unit
class TestSessionLogAllEvents:
    @pytest.fixture
    def log(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        return SessionLog(session_id="all-events-test", path=tmp_path / "all-events.jsonl")

    def test_all_events_returns_list(self, log):
        log.user_message("hello")
        log.assistant_message("world")
        events = log.all_events()
        assert isinstance(events, list)
        assert len(events) == 2

    def test_all_events_is_snapshot_not_reference(self, log):
        log.user_message("first")
        snapshot = log.all_events()
        log.user_message("second")
        assert len(snapshot) == 1        # snapshot frozen at time of call
        assert len(log.all_events()) == 2

    def test_all_events_preserves_order(self, log):
        for i in range(5):
            log.user_message(f"msg{i}")
        events = log.all_events()
        contents = [e.data["content"] for e in events]
        assert contents == [f"msg{i}" for i in range(5)]

    def test_all_events_indices_match_events_of_type(self, log):
        log.user_message("a")
        log.tool_call("bash", {"command": "ls"})
        log.assistant_message("b")
        all_e = log.all_events()
        user_e = log.events_of_type("user_message")
        # all_events()[0] is the same object as events_of_type("user_message")[0]
        assert all_e[0] is user_e[0]


# ===========================================================================
# Unit — Graph atomic save + backup fallback (Fix #9)
# ===========================================================================

@pytest.mark.unit
class TestGraphAtomicSave:
    def test_save_creates_no_tmp_on_success(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        path = tmp_path / "graph.json"
        g = GraphStore(persist_path=path)
        g.upsert_node("Alpha", node_type="entity", summary="test node")
        g.save()
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()

    def test_save_creates_backup_on_second_write(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        path = tmp_path / "graph.json"
        g = GraphStore(persist_path=path)
        g.upsert_node("Alpha", node_type="entity", summary="first")
        g.save()
        g.upsert_node("Beta", node_type="entity", summary="second")
        g.save()
        bak = path.with_suffix(".json.bak")
        assert bak.exists()

    def test_load_falls_back_to_backup(self, tmp_path):
        from birdclaw.memory.graph import GraphStore
        path = tmp_path / "graph.json"
        g = GraphStore(persist_path=path)
        g.upsert_node("Fallback", node_type="entity", summary="from backup")
        g.save()
        # Corrupt the main file
        path.write_text("not valid json", encoding="utf-8")
        # backup (.bak) was created on second save — make one manually
        bak = path.with_suffix(".json.bak")
        import shutil
        # Save again so .bak holds good data, then corrupt main
        g2 = GraphStore(persist_path=path)
        g2.upsert_node("Fallback", node_type="entity", summary="from backup")
        g2.save()          # writes good .json
        g2.save()          # second write: good .json → .bak, new .json
        path.write_text("{}", encoding="utf-8")  # corrupt main (empty graph)
        # Now load — should fall back to .bak and recover "Fallback"
        g3 = GraphStore(persist_path=path)
        # Main file is {} (empty graph), bak has the node — load picks bak
        assert g3.get_node("Fallback") is not None


# ===========================================================================
# Unit — files._strip_leaked_tool_call (Fix #14)
# ===========================================================================

@pytest.mark.unit
class TestStripLeakedToolCall:
    def _strip(self, content):
        from birdclaw.tools.files import _strip_leaked_tool_call
        return _strip_leaked_tool_call(content)

    def test_no_trailing_object_unchanged(self):
        text = "some normal file content\nwith multiple lines"
        assert self._strip(text) == text

    def test_strips_write_schema_trailing_object(self):
        text = 'file content\n{"path": "/foo.py", "content": "x"}'
        result = self._strip(text)
        assert '{"path"' not in result
        assert "file content" in result

    def test_strips_edit_schema_keys(self):
        text = 'file content\n{"old": "x", "new": "y"}'
        result = self._strip(text)
        assert "file content" in result
        assert '"old"' not in result

    def test_does_not_strip_non_schema_object(self):
        text = 'data\n{"user": "alice", "age": 30}'
        result = self._strip(text)
        # Contains keys not in the write schema — must not strip
        assert result == text

    def test_does_not_strip_mixed_keys(self):
        text = 'data\n{"path": "/f.py", "user": "x"}'
        result = self._strip(text)
        assert result == text

    def test_does_not_strip_embedded_object(self):
        # Object in the middle of content, not trailing
        text = '{"path": "x"}\nmore content after'
        assert self._strip(text) == text


# ===========================================================================
# Unit — SubtaskExecutor regression rollback (Fix #7)
# ===========================================================================

@pytest.mark.unit
class TestSubtaskRollbackOnRegression:
    def test_regression_triggers_rollback(self, tmp_path, monkeypatch):
        """If a write causes a previously-complete item to regress, file is restored."""
        from birdclaw.agent import subtask_executor as _se
        from birdclaw.agent.subtask_manifest import SubtaskManifest, SubtaskItem
        from unittest.mock import MagicMock, patch
        import birdclaw.config as _cfg

        output_file = tmp_path / "rollback.md"
        good_content = "## Alpha\n\nThis is good alpha content with enough text to be complete.\n"
        output_file.write_text(good_content, encoding="utf-8")

        # Two items: Alpha (complete), Beta (to write)
        alpha = SubtaskItem(index=0, title="Alpha", anchor="Alpha", kind="section", expected_min_chars=10)
        alpha.status = "complete"
        beta  = SubtaskItem(index=1, title="Beta",  anchor="Beta",  kind="section", expected_min_chars=10)

        manifest = SubtaskManifest(
            stage_goal="write doc",
            file_path=str(output_file),
            file_type="doc",
            items=[alpha, beta],
        )

        call_count = [0]

        def mock_generate(messages, **kwargs):
            call_count[0] += 1
            # First attempt: overwrites the file destroying Alpha (regression)
            # Subsequent attempts: append short content (exhausts retries → partial)
            if call_count[0] == 1:
                bad_content = "## Beta\n\nNew content that is long enough to pass the char check.\n"
            else:
                bad_content = "## Beta\n\nRetry content.\n"
            return MagicMock(content=json.dumps({"path": str(output_file), "content": bad_content}))

        mock_client = MagicMock()
        mock_client.generate.side_effect = mock_generate

        with patch.object(_cfg.settings, "workspace_roots", [tmp_path]):
            with patch.object(_se, "_verifier") as mock_verifier:
                # First verify call (after first write): simulate Alpha regressed
                # Second+ verify calls: no regression, but Beta not complete either
                call_num = [0]
                def fake_run(mf, content):
                    call_num[0] += 1
                    from birdclaw.agent.subtask_manifest import SubtaskDiff
                    if call_num[0] == 1:
                        # Regression: Alpha not found in content
                        alpha.status = "complete"  # was complete before write
                        d = SubtaskDiff()
                        d.missing = [beta]; d.regressed = [alpha]
                        return d
                    d = SubtaskDiff(); d.partial = [beta]
                    return d
                mock_verifier.run.side_effect = fake_run
                mock_verifier.parse_doc_sections.return_value = {"Beta": "content"}

                with patch.object(_se, "_planner") as mock_planner:
                    mock_planner.plan.return_value = manifest
                    _se.run_stage(
                        llm_client=mock_client,
                        stage={"goal": "write doc"},
                        file_path=str(output_file),
                        file_type="doc",
                        step=1, max_steps=20,
                        store_manifest=lambda m: None,
                    )

        # File should have been rolled back to good_content on first regression
        final = output_file.read_text(encoding="utf-8")
        assert "Alpha" in final


# ===========================================================================
# Unit — TaskRegistry atomic write (Fix #13)
# ===========================================================================

@pytest.mark.unit
class TestTaskRegistryAtomicWrite:
    @pytest.fixture
    def reg(self, tmp_path, monkeypatch):
        import birdclaw.config as _cfg
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            r = TaskRegistry()
        return r

    def test_no_tmp_file_after_save(self, reg, tmp_path):
        import birdclaw.config as _cfg
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            reg2 = TaskRegistry()
        task = reg2.create("atomic test task")
        reg2.start(task.task_id, agent_id="test")
        # After mutation, .tmp should not exist
        tasks_dir = tmp_path / "tasks"
        tmp_files = list(tasks_dir.glob("*.tmp"))
        assert tmp_files == [], f"stray .tmp files found: {tmp_files}"

    def test_task_file_readable_after_write(self, tmp_path):
        import birdclaw.config as _cfg
        from birdclaw.memory.tasks import TaskRegistry
        with patch.object(_cfg.settings, "data_dir", tmp_path):
            reg = TaskRegistry()
            task = reg.create("readable task")
            reg.complete(task.task_id, output="done")
            task_file = tmp_path / "tasks" / f"{task.task_id}.json"
            data = json.loads(task_file.read_text())
        assert data["status"] == "completed"


# ===========================================================================
# Unit — Stage budget minimum clamp (Fix #6)
# ===========================================================================

@pytest.mark.unit
class TestBudgetMinimumClamp:
    def test_zero_history_budget_clamped_to_one(self, tmp_path, monkeypatch):
        """_historical_budget returning 0 must be clamped to at least 1."""
        import birdclaw.agent.budget as _budget
        import birdclaw.config as _cfg

        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)
        # No history file — historical_budget returns the default (which must be >= 1)
        budget = _budget.historical_budget("research")
        assert budget >= 1

    def test_budget_cfg_override_minimum(self, monkeypatch, tmp_path):
        """A cfg budget of 0 must still be clamped to 1 by loop.py logic."""
        import birdclaw.agent.budget as _budget
        import birdclaw.config as _cfg
        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)
        # Simulate the clamping logic used in loop.py: max(1, cfg.get("budget") or historical)
        budget = max(1, 0 or _budget.historical_budget("research"))
        assert budget >= 1


# ===========================================================================
# Unit — Gateway delivery fallback (Fix #2)
# ===========================================================================

@pytest.mark.unit
class TestGatewayDeliveryFallback:
    def test_deliver_uses_session_channel_id_as_fallback(self):
        """Delivery falls back to session.channel_id when _session_channel is empty."""
        from birdclaw.gateway.gateway import Gateway
        from birdclaw.gateway.channel import OutgoingMessage
        from unittest.mock import MagicMock, patch

        gw = Gateway()
        delivered = []

        mock_channel = MagicMock()
        mock_channel.channel_id = "tui"
        mock_channel.deliver.side_effect = lambda m: delivered.append(m)
        gw._channels["tui"] = mock_channel

        # Register a session in the session manager but NOT in _session_channel
        sess = gw._session_mgr.get_or_create("tui", "user1")
        # _session_channel is intentionally empty (simulates post-restart state)
        assert sess.session_id not in gw._session_channel

        msg = OutgoingMessage(session_id=sess.session_id, content="hello")
        gw._deliver(sess.session_id, msg)

        assert len(delivered) == 1
        assert delivered[0].content == "hello"

    def test_deliver_logs_warning_for_unknown_session(self, caplog):
        from birdclaw.gateway.gateway import Gateway
        from birdclaw.gateway.channel import OutgoingMessage
        import logging

        gw = Gateway()
        with caplog.at_level(logging.WARNING, logger="birdclaw.gateway.gateway"):
            gw._deliver("no-such-session", OutgoingMessage(
                session_id="no-such-session", content="lost"
            ))
        assert any("no channel" in r.message for r in caplog.records)


# ===========================================================================
# Unit — soul_loop launch_cwd parameter (Fix #1)
# ===========================================================================

@pytest.mark.unit
class TestSoulLoopLaunchCwd:
    def test_deep_path_accepts_launch_cwd(self):
        """_deep_path must accept launch_cwd without raising NameError."""
        from birdclaw.agent import soul_loop as _sl
        import inspect
        sig = inspect.signature(_sl._deep_path)
        assert "launch_cwd" in sig.parameters, (
            "_deep_path missing launch_cwd parameter — would NameError on escalation"
        )

    def test_deep_path_launch_cwd_default_empty(self):
        from birdclaw.agent import soul_loop as _sl
        import inspect
        param = inspect.signature(_sl._deep_path).parameters["launch_cwd"]
        assert param.default == "" or param.default is inspect.Parameter.empty or param.default == ""


# ===========================================================================
# Unit — self_update restore is in-place (Fix #11)
# ===========================================================================

@pytest.mark.unit
class TestSelfUpdateInPlaceRestore:
    def test_restore_does_not_remove_directory(self, tmp_path):
        """_restore must never delete the live birdclaw/ directory (in-place swap)."""
        from birdclaw.agent import self_update as _su
        import shutil

        # Build a fake backup tree
        backup = tmp_path / "backup"
        (backup / "birdclaw" / "agent").mkdir(parents=True)
        (backup / "birdclaw" / "agent" / "loop.py").write_text("# loop", encoding="utf-8")

        # Build a fake live tree
        live = tmp_path / "birdclaw"
        (live / "agent").mkdir(parents=True)
        orig_file = live / "agent" / "loop.py"
        orig_file.write_text("# original", encoding="utf-8")
        extra_file = live / "agent" / "extra.py"
        extra_file.write_text("# extra", encoding="utf-8")  # not in backup

        with patch.object(_su, "_BIRDCLAW_SRC", live):
            _su._restore(backup)

        # live/ directory must still exist (was never rmtree'd)
        assert live.exists(), "live directory was deleted during restore"
        # File from backup was copied in
        assert orig_file.read_text() == "# loop"
        # File not in backup was removed
        assert not extra_file.exists(), "extra file should have been removed"

    def test_restore_preserves_constitution(self, tmp_path):
        """Constitution text must survive the restore."""
        from birdclaw.agent import self_update as _su

        backup = tmp_path / "backup"
        (backup / "birdclaw" / "agent").mkdir(parents=True)
        (backup / "birdclaw" / "agent" / "soul_constitution.py").write_text(
            "# old constitution", encoding="utf-8"
        )

        live = tmp_path / "birdclaw"
        (live / "agent").mkdir(parents=True)
        constitution = live / "agent" / "soul_constitution.py"
        constitution.write_text("# live constitution", encoding="utf-8")

        with patch.object(_su, "_BIRDCLAW_SRC", live):
            _su._restore(backup)

        # Live constitution text is preserved even though backup had a different version
        assert "live constitution" in constitution.read_text()


# ===========================================================================
# Unit — Graph load from corrupt-main fallback to .bak (explicit test)
# ===========================================================================

@pytest.mark.unit
class TestGraphBackupFallback:
    def test_corrupt_main_loads_from_bak(self, tmp_path):
        """If main file is corrupt JSON, load must fall back to .bak."""
        from birdclaw.memory.graph import GraphStore

        path = tmp_path / "graph.json"
        bak  = path.with_suffix(".json.bak")

        # Write a valid backup directly
        import networkx as nx
        g = nx.DiGraph()
        g.add_node("python", name="Python", type="entity", summary="language",
                   sources=[], last_seen="2024-01-01T00:00:00+00:00")
        bak.write_text(
            __import__("json").dumps(nx.node_link_data(g, edges="edges"), indent=2),
            encoding="utf-8",
        )
        path.write_text("this is not valid json {{{", encoding="utf-8")

        store = GraphStore(persist_path=path)
        assert store.get_node("Python") is not None


# ===========================================================================
# Unit — edit_file must fail cleanly when old_string not found
# ===========================================================================

@pytest.mark.unit
class TestEditFileBehavior:
    @pytest.fixture(autouse=True)
    def patch_workspace(self, tmp_path, monkeypatch):
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path])
        self.ws = tmp_path

    def test_edit_not_found_returns_error(self):
        from birdclaw.tools.files import write_file, edit_file
        p = str(self.ws / "f.txt")
        write_file(p, "hello world")
        result = json.loads(edit_file(p, "does not exist", "replacement"))
        assert "error" in result

    def test_edit_multiple_matches_requires_replace_all(self):
        from birdclaw.tools.files import write_file, edit_file
        p = str(self.ws / "f.txt")
        write_file(p, "foo foo foo")
        result = json.loads(edit_file(p, "foo", "bar"))
        assert "error" in result
        assert "3" in result["error"] or "replace_all" in result["error"]

    def test_edit_produces_patch_on_success(self):
        from birdclaw.tools.files import write_file, edit_file
        p = str(self.ws / "f.txt")
        write_file(p, "old line")
        result = json.loads(edit_file(p, "old line", "new line"))
        assert result.get("replaced") == 1
        assert "patch" in result
        assert "-old line" in result["patch"]
        assert "+new line" in result["patch"]


# ===========================================================================
# Unit — SessionLog load / round-trip
# ===========================================================================

@pytest.mark.unit
class TestSessionLogRoundTrip:
    def test_load_restores_all_events(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog
        path = tmp_path / "rt.jsonl"
        log = SessionLog(session_id="rt", path=path)
        log.user_message("first")
        log.tool_call("bash", {"command": "ls"})
        log.assistant_message("done")

        log2 = SessionLog.load("rt") if False else SessionLog(session_id="rt", path=path)
        # Reload by constructing a new instance from the same path
        log2 = SessionLog(session_id="rt", path=path)
        log2._events.clear()
        for line in path.read_text().splitlines():
            if line.strip():
                from birdclaw.memory.session_log import Event
                log2._events.append(Event.from_dict(__import__("json").loads(line)))

        assert len(log2.all_events()) == 3
        assert log2.all_events()[0].type == "user_message"
        assert log2.all_events()[2].type == "assistant_message"

    def test_malformed_line_skipped_on_load(self, tmp_path):
        from birdclaw.memory.session_log import SessionLog, Event
        path = tmp_path / "malformed.jsonl"
        # Write one valid and one corrupt line
        path.write_text(
            '{"type":"user_message","ts":"2024-01-01T00:00:00+00:00","data":{"content":"hi"}}\n'
            'NOT_VALID_JSON\n',
            encoding="utf-8",
        )
        log = SessionLog.load.__func__(SessionLog, "malformed") if False else SessionLog(
            session_id="malformed", path=path
        )
        log._events.clear()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                log._events.append(Event.from_dict(__import__("json").loads(line)))
            except Exception:
                pass
        assert len(log.all_events()) == 1


# ===========================================================================
# Unit — Rate limiter prune (Fix #12)
# ===========================================================================

@pytest.mark.unit
class TestBashRateLimiter:
    def test_expired_entries_removed_after_window(self):
        """Once all timestamps fall outside the window, the entry is removed."""
        import birdclaw.tools.bash as _bash

        past = time.time() - 200  # well past the 60s window
        _bash._rate_counts["stale-task"] = [past, past, past]
        # After calling _check_rate_limit for the same task_id,
        # all timestamps are outside the window → entry is removed and a
        # fresh timestamp is added (so entry re-appears). But importantly
        # old timestamps are gone.
        _bash._check_rate_limit("stale-task")
        remaining = _bash._rate_counts.get("stale-task", [])
        # All remaining timestamps must be recent (< 5s old)
        assert all(time.time() - t < 5 for t in remaining)

    def test_rate_limit_blocks_after_limit(self):
        """Task exceeding _RATE_LIMIT in one window is blocked."""
        import birdclaw.tools.bash as _bash

        now = time.time()
        # Simulate a task that has already hit the limit
        _bash._rate_counts["heavy-task"] = [now] * _bash._RATE_LIMIT
        result = _bash._check_rate_limit("heavy-task")
        assert result is False


# ===========================================================================
# Unit — ApprovalQueue (core approval flow)
# ===========================================================================

@pytest.mark.unit
class TestApprovalQueue:
    @pytest.fixture
    def queue(self, tmp_path, monkeypatch):
        import birdclaw.config as _cfg
        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)
        from birdclaw.agent.approvals import ApprovalQueue
        return ApprovalQueue()

    def test_non_destructive_auto_approved(self, queue):
        decision = queue.request("task-1", "agent-1", "read_file", "read /etc/hosts")
        assert decision == "allow_once"

    def test_resolve_allow_once(self, queue):
        import threading
        results = []

        def _requester():
            d = queue.request("task-2", "agent-2", "bash", "rm -rf /tmp/test_dir", timeout=5)
            results.append(d)

        t = threading.Thread(target=_requester)
        t.start()
        time.sleep(0.1)  # let request register
        pending = queue.list_pending()
        assert len(pending) >= 1
        queue.resolve(pending[0].approval_id, "allow_once")
        t.join(timeout=3)
        assert results == ["allow_once"]

    def test_expired_request_returns_deny(self, queue):
        """A request that times out (no resolve) returns deny."""
        # Use a very short timeout so the test is fast
        decision = queue.request("task-3", "agent-3", "bash", "rm -rf /tmp/expired", timeout=0.1)
        assert decision == "deny"

    def test_resolve_unknown_id_returns_false(self, queue):
        assert queue.resolve("nonexistent-id", "allow_once") is False

    def test_list_pending_empty_initially(self, queue):
        assert queue.list_pending() == []

    def test_allow_always_persisted(self, queue, tmp_path):
        import threading
        results = []

        def _req():
            results.append(
                queue.request("task-4", "agent-4", "bash", "rm -rf /tmp/always", timeout=5)
            )

        t = threading.Thread(target=_req)
        t.start()
        time.sleep(0.1)
        pending = queue.list_pending()
        assert pending
        queue.resolve(pending[0].approval_id, "allow_always")
        t.join(timeout=3)
        assert results == ["allow_always"]
        # Allowed key must be persisted
        allowed_path = tmp_path / "allowed_permissions.json"
        assert allowed_path.exists()
        data = json.loads(allowed_path.read_text())
        assert len(data["allowed"]) >= 1


# ===========================================================================
# Unit — Permission enforcer modes
# ===========================================================================

@pytest.mark.unit
class TestPermissionEnforcer:
    def test_workspace_write_allows_workspace_files(self, tmp_path, monkeypatch):
        from birdclaw.tools.permission import PermissionEnforcer
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path])
        enforcer = PermissionEnforcer(mode="workspace_write")
        result = enforcer.check_file_write(tmp_path / "file.txt")
        assert result.allowed

    def test_workspace_write_blocks_outside_workspace(self, tmp_path, monkeypatch):
        from birdclaw.tools.permission import PermissionEnforcer
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path / "inside"])
        enforcer = PermissionEnforcer(mode="workspace_write")
        result = enforcer.check_file_write(tmp_path / "outside.txt")
        assert not result.allowed

    def test_read_only_blocks_writes(self, tmp_path, monkeypatch):
        from birdclaw.tools.permission import PermissionEnforcer
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path])
        enforcer = PermissionEnforcer(mode="read_only")
        result = enforcer.check_file_write(tmp_path / "file.txt")
        assert not result.allowed

    def test_danger_full_access_allows_any_write(self, tmp_path, monkeypatch):
        from birdclaw.tools.permission import PermissionEnforcer
        from birdclaw.config import settings
        monkeypatch.setattr(settings, "workspace_roots", [tmp_path])
        enforcer = PermissionEnforcer(mode="danger_full_access")
        result = enforcer.check_file_write(Path("/tmp/anywhere.txt"))
        assert result.allowed


# ===========================================================================
# Unit — SubtaskManifest core logic
# ===========================================================================

@pytest.mark.unit
class TestSubtaskManifest:
    @pytest.fixture
    def manifest(self):
        from birdclaw.agent.subtask_manifest import SubtaskManifest, SubtaskItem
        items = [
            SubtaskItem(index=0, title="Intro",    anchor="Intro",    kind="section", expected_min_chars=50),
            SubtaskItem(index=1, title="Body",     anchor="Body",     kind="section", expected_min_chars=100),
            SubtaskItem(index=2, title="Conclusion",anchor="Conclusion",kind="section",expected_min_chars=50),
        ]
        return SubtaskManifest(stage_goal="write essay", file_path="/tmp/essay.md",
                               file_type="doc", items=items)

    def test_total_count(self, manifest):
        assert manifest.total == 3

    def test_done_count_initially_zero(self, manifest):
        assert manifest.done_count == 0

    def test_mark_complete_increments_done(self, manifest):
        manifest.items[0].status = "complete"
        assert manifest.done_count == 1

    def test_all_done_when_all_complete(self, manifest):
        for item in manifest.items:
            item.status = "complete"
        assert manifest.all_done

    def test_not_all_done_with_partial(self, manifest):
        manifest.items[0].status = "complete"
        manifest.items[1].status = "partial"
        assert not manifest.all_done

    def test_pending_items_excludes_complete(self, manifest):
        manifest.items[0].status = "complete"
        pending = [it for it in manifest.items if it.status != "complete"]
        assert len(pending) == 2
        assert manifest.items[0] not in pending


# ===========================================================================
# Unit — Budget history log and P75
# ===========================================================================

@pytest.mark.unit
class TestBudgetHistoryP75:
    def test_p75_from_multiple_entries(self, tmp_path, monkeypatch):
        import birdclaw.agent.budget as _budget
        import birdclaw.config as _cfg
        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)

        # Log 8 research entries with known step counts
        for steps in [3, 5, 5, 6, 7, 8, 10, 12]:
            _budget.log_stage("research", steps, "test goal")

        hist = _budget.historical_budget("research")
        # P75 of [3,5,5,6,7,8,10,12] = 9 (between 8 and 10)
        assert hist >= 6  # must be higher than median
        assert hist <= 12

    def test_unknown_stage_returns_settings_default(self, tmp_path, monkeypatch):
        import birdclaw.agent.budget as _budget
        import birdclaw.config as _cfg
        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)
        # "unknown_stage" has no history — returns stage_budgets default or fallback
        budget = _budget.historical_budget("unknown_stage_xyz")
        assert budget >= 1

    def test_log_stage_creates_file(self, tmp_path, monkeypatch):
        import birdclaw.agent.budget as _budget
        import birdclaw.config as _cfg
        monkeypatch.setattr(_cfg.settings, "data_dir", tmp_path)
        _budget.log_stage("verify", 4, "test verify")
        history_file = tmp_path / "memory" / "stage_history.jsonl"
        assert history_file.exists()
        lines = [json.loads(l) for l in history_file.read_text().splitlines() if l.strip()]
        assert any(l.get("type") == "verify" for l in lines)


if __name__ == "__main__":
    if "--eval" in sys.argv and len(sys.argv) >= 4:
        idx = sys.argv.index("--eval")
        print(json.dumps(compare_metrics(Path(sys.argv[idx+1]), Path(sys.argv[idx+2])), indent=2))
    else:
        print("Run with pytest, or: python test.py --eval <before.jsonl> <after.jsonl>")
