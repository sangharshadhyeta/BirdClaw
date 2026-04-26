# BirdClaw Known Issues & Implementation Plan

## Core Philosophy: Small Brain + Progressive External Memory

The LLM is stateless. Files are the memory. Each call reads only what it needs.
- **File = memory**: what's written to disk persists across calls, crashes, sessions
- **Progressive disclosure**: each call gets only the relevant slice, not the whole file
- **Index = BIRDCLAW.md**: points to files; files are read on demand via line search
- **Brain = processor**: reasons over a small fresh context, writes result to file, forgets

```
Call N:  [system] + [last 30 lines of file] + [what to write next]  →  writes to file
Call N+1:[system] + [last 30 lines of file] + [what to write next]  →  writes to file
```
Context window stays constant. File grows. Intelligence scales without growing memory.

---

## Active Bugs

### 1. Research Stage Misclassification
Steps like "Summarise research into a guidebook draft" classified as `research`
but have no bash/web tool → model loops on think() → force-advanced to web_search.
Fix: keyword routing "summarise" → reflect or write_doc.

---

## Implementation Plan

### Phase 1 — Infrastructure ✅ DONE
- [x] llama.cpp `--parallel 4` → `--parallel 2` (16k per slot, fixes 400 errors)
- [x] Compaction threshold 5000 → 2000 (safety net)
- [x] 400 auto-trim: context overflow → trim messages → retry (forgiving)
- [x] BIRDCLAW.md task IDs: full task_id not task_id[:8]
- [x] write_dir: always Path.cwd() (launch directory)

### Phase 2 — Stage Reset ✅ DONE
- [x] Messages reset to [system + stage_ctx] at every stage boundary
- [x] completed_stages summaries replace raw message history
- [x] write_doc/write_code continuation: inject last 20 lines of file

### Phase 3 — Progressive Memory ✅ DONE
The file IS the memory. Line search is the mechanism to read it precisely.

- [x] 3a. `birdclaw/tools/line_search.py` — search files for exact lines
        search_lines(pattern, files, context=3) → formatted string for LLM injection
        Used by: context builder, supervisor verification, thinker as a tool

- [x] 3b. Smart file reading in subtask_executor._build_call_messages
        find_section: finds section/function matching item title → returns just that block
        4-step: BIRDCLAW.md → find_section → search_relevant → find_continuation_point
        Snake_case split: "run_stage" matches "run stage" item title

- [x] 3c. line_search as thinker tool
        search_notes tool registered; available in research/verify/reflect stages
        Goal-driven (search_relevant) + regex (search_lines) + default paths (*.md in cwd)

- [x] 3d. Research stage notes file + task subfolder
        Each task gets {task_slug}/ folder — notes.md + output files, all off cwd root
        think() + tool results appended to {task_slug}/notes.md after every research/reflect call
        Next step: search_relevant(goal, notes.md) injected into step_msg before model call
        _default_search_paths() also picks up */notes.md from task subdirs

### Phase 4 — Doer Split ✅ DONE
- [x] subtask_executor: removed shared messages list
- [x] Each item call: fresh [system, file_tail, instruction] — 3 messages
- [x] error_hint threaded through retries instead of appending to messages
- [x] loop.py: removed messages/Message params from run_stage call

### Phase 5 — Stream Guard
- [ ] llm_client.generate: optional stream_guard callback
- [ ] First 50 tokens checked against expected format pattern
- [ ] On mismatch: cancel HTTP request, retry with correction
- [ ] Doer guard: catch {"content": "..."} wrapper, markdown blocks

### Phase 6 — Supervisor
- [ ] supervisor.py: StepSupervisor with async submit/collect
- [ ] Runs in parallel slot while next thinker runs
- [ ] Uses line_search to verify expected output exists in file
- [ ] Returns: ok | redirect(correction) | stop(reason)
- [ ] Correction injected into next thinker's stage_ctx

### Phase 7 — Parity Repo Ports (from claw-code-parity audit)
Patterns from `repo/claw-code-parity/rust/crates/runtime/src/` not yet ported.

#### 7a. Symlink Escape Detection (safety gap — quick win)
- [ ] `files.py _validate_workspace`: resolve symlink targets and re-check boundary
        Current check passes `workspace/link → /etc/passwd` — target is outside but string check passes
        Fix: `if path.is_symlink(): path = path.resolve()` before `starts_with` check

#### 7b. SYSTEM_PROMPT_DYNAMIC_BOUNDARY (cache efficiency)
- [ ] Split SYSTEM prompt into static section (never changes) + dynamic section (workspace, date, CLAUDE.md)
        Static section = cache-stable across every call → llama.cpp KV cache hit on every turn
        Dynamic section appended after a `<!-- DYNAMIC -->` boundary marker
        Source: `prompt.rs` SystemPromptBuilder with static/dynamic split
        Impact: re-encoding the static system prompt costs tokens on every single call today

#### 7c. Prompt Cache Awareness (smarter compaction trigger)
- [ ] Track cache read vs write token counts in llm_client (from API response headers/body)
        Trigger compaction when cache-write tokens dominate (cache thrashing = paying full price every turn)
        Currently compact on message count alone — misses the actual cost signal
        Source: `conversation.rs` + `api/prompt_cache.rs` PromptCacheEvent

#### 7d. Session Fork / Parent Tracking
- [ ] When a task spawns a sub-task, record `{parent_task_id, branch_name}` in session_log
        Enables post-run analysis: "what spawned what", "which subtask caused the failure"
        Source: `session.rs` SessionFork struct

#### 7e. Container / Environment Detection
- [ ] Detect at startup: /.dockerenv, cgroup markers, namespace availability
        Adapt sandbox behaviour: use unshare if available, skip if inside container
        Source: `sandbox.rs` container detection logic

---

## Fixed This Session

- functiongemma (270M) fully removed; all calls use main_profile (4B)
- Soul loop rewritten: format_schema + flat routing schema, run_command action added
- Session ID isolation: each TUI launch gets unique ID, tasks tagged, filter by ID not time
- Orphan reap runs before first render (no flash of dead tasks on startup)
- Stale task fix: running tasks >2h from dead sessions marked stopped at load
- Workspace write_dir: always cwd (launch directory)
- Research stage: bash tool added
- Planner: write_doc only for explicit file requests; system status → verify stages only
- TUI: call_after_refresh fix for #am-body NoMatches crash
- CLI: waits for task completion, shows output inline with Markdown
- Logging: WARNING-level stderr suppressed in TUI mode; ERROR-only in CLI
- llama.cpp parallel 4→2 (16k per slot, fixes context overflow)
- 400 context overflow: auto-trim oldest messages and retry
- BIRDCLAW.md task IDs: full ID not truncated 8-char
- Compaction threshold: 5000 → 2000
- Stage reset: messages rebuilt fresh at each stage boundary
- Subtask executor: file-as-memory pattern, no shared messages list
