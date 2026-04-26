# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision

BirdClaw is a persistent Python AI agent — not a chatbot, not a coding assistant you invoke and dismiss. It is a long-term autonomous worker powered by two local models (Gemma 4 4B as thinker, functiongemma 270M as format worker) via llama.cpp. It runs as a daemon, accumulates knowledge, works toward goals while you are away, reports back through messaging channels, and improves its own code over time. Think of it as an AGI scaffold built for the real world: small models, big memory, never sleeps.

## Environment

- **Runtime**: conda env `claude-code` — Python 3.10, bare metal (no Docker)
- **Activate**: `conda activate claude-code`
- **Main model**: `gemma-4-e4b-it-Q8_0.gguf` via llama.cpp at `http://localhost:8081/v1` (OpenAI-compatible)
- **Hands model**: `functiongemma-270m.gguf` via llama.cpp at `http://localhost:8082/v1` (optional, falls back to main)
- **Working directory**: `/home/Projects/BirdClaw/`

## Commands

```bash
# Interactive REPL (soul loop — multi-turn, task-spawning)
python main.py cli

# Three-pane multi-agent TUI
python main.py tui

# One-shot prompt (runs agent loop directly, no soul layer)
python main.py prompt "your task here"

# Start the gateway daemon (persistent, no TUI)
python main.py daemon

# Drain session logs into the knowledge graph
python main.py memorise [session_id]

# Expose knowledge graph as an MCP stdio server
python main.py graph-server

# Run a full dreaming cycle (graph merge + reflection + inner life + user knowledge + cleanup)
python main.py dream

# Prune stale sessions, tasks, pages, and self-update backups
python main.py cleanup

# Run all tests
pytest

# Run a single test by name
pytest tests/test.py::test_function_name -x

# Run tests excluding long-running ones
pytest -m "not long_running"

# Install / sync deps
pip install -e .
```

Disable the LLM scheduler in dev/testing: `BC_LLM_SCHEDULER_ENABLED=false`

## Project Structure

```
birdclaw/           Python package — all source code
  llm/             Model client, scheduler (priority queue), model_profile, usage tracking
  tools/           bash, files, web, search, MCP bridge, hooks, sandbox, registry
  agent/           Agent loop, soul layer, planner, budget, subtask pipeline, self-update
  memory/          GraphRAG graph, session log, history, dreaming/memorise, cleanup, tasks
  gateway/         Persistent daemon, session manager, channel protocol
  channels/        Messaging channel adapters (TUI built; Telegram/Discord stubbed)
  skills/          Named runbooks / standing goals, cron scheduler
  tui/             Three-pane Textual TUI (tasks + output + conversation)
  config.py        Settings (env + ~/.birdclaw/config.toml)
main.py             Entry point — subcommands: cli, tui, prompt, daemon, memorise, graph-server, dream, cleanup
pyproject.toml      Dependencies
repo/               Reference projects (read-only)
  openclaw/         TypeScript — channel/gateway/skills/plugin ideas
  claw-code-parity/ Rust — execution safety patterns
  Logic_Engine/     Python — small model loop, GraphRAG, tool registry
```

## Reference Map — What We Take From Each

### From `repo/Logic_Engine/` (Python)
- `agent/loop.py` — `think → tool → answer` loop, `tool_choice="required"`, max-step enforcement with forced answer fallback
- `llm/llama_client.py` — `Message / ToolCall / GenerationResult` dataclass interface (we use llama.cpp/httpx)
- `graphrag/` — NetworkX-based knowledge graph, entity extraction, semantic chunking, BFS navigation
- `pipeline/ingest_pipeline.py` — document → proposition → entity → graph pipeline
- `llm/history.py` — sliding-window context management
- `agent/tool_registry.py` — `Tool` dataclass + registry pattern

### From `repo/claw-code-parity/rust/crates/runtime/` (Rust → Python patterns)
- `bash.rs` — bash with configurable timeout, 16 KB output truncation, background process support
- `file_ops.rs` — workspace boundary validation, binary file detection (NUL bytes), 10 MB read/write limits, structured diffs
- `permission_enforcer.rs` — 5-mode permission system: `ReadOnly`, `WorkspaceWrite`, `DangerFullAccess`, `Prompt`, `Allow`; heuristic dangerous-command detection
- `task_registry.rs` — task lifecycle (Created → Running → Completed/Failed/Stopped), sub-agent spawning, output accumulation
- `team_cron_registry.rs` — cron scheduling, team task coordination
- `mcp_tool_bridge.rs` — MCP client for connecting to external tool servers

### From `repo/openclaw/` (TypeScript → Python ideas)
- Channel architecture — adapter per platform, unified message type, send/receive symmetry
- Gateway/daemon — persistent process, session management, graceful restart
- Skills — named markdown runbooks that drive autonomous behavior
- Plugin system — register new tools/channels/skills without touching core
- Doctor command — health check across all subsystems

## Architecture Layers

```
CHANNELS          TUI (built) · Telegram · Discord · Slack (stubbed)
GATEWAY/DAEMON    Persistent service · session manager · graceful restart
SKILLS/CRON       Named runbooks · standing goals · schedule triggers
SOUL LAYER        Conversational entry · create_task · resolve_approval · multi-agent awareness
AGENT LOOP        plan → stage_queue → execute · outcome-first · grounded context every step
SUBTASK PIPELINE  manifest → planner → executor → verifier  (write_code / write_doc stages)
MEMORY            GraphRAG (knowledge only) · session_log (orchestration) · workspace state
TOOLS             bash · files · web · search · MCP bridge · hooks · sandbox
LLM SCHEDULER     Priority queue (INTERACTIVE → AGENT → BACKGROUND → CRON → MEMORY)
MODELS            llama.cpp · 4B thinker (port 8081) · 270M hands (port 8082)
```

## Dual-Model Architecture — 4B Thinker + 270M Hands

Every LLM call is routed through a `ModelProfile` (`birdclaw/llm/model_profile.py`):

| Call | Model | Why |
|------|-------|-----|
| Soul routing | 270M | 3-class classification — no reasoning needed |
| Plan structuring | 4B pre-think → 270M schema | 4B reasons, 270M enforces schema |
| Tool-call stages (research, verify, reflect) | 4B, thinking=True | Full reasoning on every turn |
| edit_file patch | 4B pre-think → 270M JSON | 4B identifies exact text, 270M formats diff |
| Post-stage reflect gate | 270M | 4-class classification of 4B's output |
| Content generation (write_code, write_doc) | 4B, thinking=True | Only the 4B can write coherent prose/code |
| Forced final answer | 4B, thinking=True | Full synthesis of all completed stages |

The 270M handles only structured decisions. It never has to reason — it sees output the 4B already produced and converts it to schema-valid JSON. This means the 4B never runs with `thinking=False`, eliminating the llama.cpp thinking+format conflict entirely.

When `BC_LLM_HANDS_BASE_URL` is not set, `hands_profile()` falls back to `main_profile()` — the system works on a single model, just slightly slower.

## Soul Layer — Conversational Entry Point

`cli` and `tui` go through `soul_respond()` (`birdclaw/agent/soul_loop.py`), not `run_agent_loop()` directly.

**Dual-model routing (270M-first):**
1. 270M handles the message first — no history pre-fetched, keeps it fast. Terminal tool calls end the loop immediately.
2. If 270M calls `escalate()` (vague reference, deep question), the 4B reads 50 history turns, reasons with `thinking=True`, then 270M formats the result into a tool call.

**Tool tiers available to the 270M:**
- **Search**: `search_tasks`, `get_task_output`, `search_knowledge` (GraphRAG) — up to 3 calls
- **Action**: `remember_user`, `read_inner_life`, `read_skill` — up to 2 calls
- **Terminal**: `answer`, `create_task`, `resolve_approval` — ends the loop
- **Escalate**: `escalate(reason)` → hands off to 4B deep path

`prompt` mode bypasses the soul and calls `run_agent_loop()` directly (one-shot, no task spawning).

## Agent Loop — Plan → Stage Queue → Execute

Every task goes through three phases:

**1. Plan (pre-loop)**
4B thinks about the task (`thinking=True`), then 270M structures the output into `{ "outcome": "...", "steps": "step1 | step2 | ..." }`.
Simple Q&A returns `stages: []` → free tool mode.

**2. Stage loop**
Each stage is popped from the queue. Stage type determines mode + tools:

| Type | Mode | Tools | Advances when |
|------|------|-------|---------------|
| `research` | tool_call | web_search, web_fetch, read_file, think | think() called |
| `write_code` | subtask executor | — | subtask pipeline completes |
| `write_doc` | subtask executor | — | subtask pipeline completes |
| `edit_file` | 4B pre-think → 270M patch JSON | — | `{"old":"DONE"}` sentinel |
| `verify` | tool_call | bash, read_file | bash returns no error |
| `reflect` | tool_call | think | think() called |

Per-step context message (injected every turn):
`[N/max] Outcome: <criteria> | Done: <stage1> → <stage2> | Now (type): <current goal>`

GraphRAG is queried for each stage's specific goal — relevant knowledge injected as `[context]`.

**3. Forced answer**
When queue is empty, completed stage summaries are injected and `answer()` is forced on the 4B with `thinking=True`.

**Budget tracking:**
Each stage gets a step budget (from planner override → historical P75 → `settings.stage_budgets`). Budget exhaustion force-advances to the next stage. The model can call `request_budget(additional_steps, reason)` when it detects more work remains. Historical data is stored in `~/.birdclaw/memory/stage_history.jsonl`.

**Post-stage reflection gate:**
After each `write_code`, `write_doc`, `research`, or `reflect` stage completes, a cheap 270M format-mode call evaluates quality and may: `continue` (proceed), `deepen` (re-run same stage with a gap goal), `insert` (add a new `edit_file` or `research` stage), or `done` (clear remaining queue). Deepen is capped at 2 per stage type to prevent loops.

**Auto-compaction:**
When message history grows too large, `birdclaw/memory/compact.py` removes mid-conversation turns while preserving the plan and recent context. After compaction during a format stage, the subtask manifest is re-injected so the model can resume mid-document.

## Subtask Pipeline — write_code / write_doc Stages

`write_code` and `write_doc` stages go through a 5-component pipeline instead of raw format-mode calls:

1. **`subtask_planner.py`** — breaks the stage goal into a list of named items (functions, sections)
2. **`subtask_manifest.py`** — `SubtaskManifest` dataclass tracks each item's `status` (missing / partial / complete / regressed)
3. **`subtask_executor.py`** — drives the LLM through one item at a time; retries up to `MAX_ITEM_RETRIES=2`
4. **`subtask_verifier.py`** — after writing, re-reads the file and scores each item → updates manifest
5. **`loop.py` (injection point C)** — after auto-compaction, re-injects the manifest resume context

The manifest is stored in `task_registry` and `session_log` (injection points A and B) so it survives context compaction.

## File Cleanup — Data Retention Policy

`birdclaw/memory/cleanup.py` runs at the end of every dream cycle and via `python main.py cleanup`:

| Store | Retention |
|-------|-----------|
| `sessions/` | Keep if un-memorised OR < 7 days old. Memorised sessions are safe to delete. |
| `tasks/` | Keep running/failed indefinitely. Delete completed tasks > 3 days old. |
| `pages/` | Delete after 24 hours — web content is transient. |
| `self_update/` | Keep last 5 backup snapshots. |
| `stage_history.jsonl` | Trim to last 1000 entries — older entries don't improve P75 estimates. |

## State Separation

| What | Where | Never in |
|------|-------|----------|
| Knowledge (research findings, code facts, entities) | GraphRAG | session_log as raw data |
| Orchestration (plan, stage summaries, tool calls) | session_log JSONL | GraphRAG |
| Live stage progress | in-memory `completed_stages` | anywhere |

## Design Principles

- **Outcome first.** Every task — even two-line code additions — defines success criteria before execution. The criteria follows every step.
- **Grounded always.** GraphRAG context for the task + per-stage context for each goal. Model never reasons from parametric memory alone.
- **Dynamic stages.** The model generates its own workflow at plan time. No hardcoded stage configs per task type.
- **Thinker + hands.** 4B reasons on every call (thinking=True always). 270M converts that reasoning into schema-valid JSON. Python drives everything else.
- **Show ≤6 tools per turn.** Each stage type offers only the tools relevant to it.
- **Permission before action.** All bash and file writes go through the permission enforcer.
- **Discuss before implementing.** Each new feature is discussed and agreed before code is written.

## Planned Features (not yet built)

- **Research ingestion** — findings from `research` stages auto-ingested into GraphRAG during dreaming (ingest pipeline exists; loop injection missing).
- **Telegram/Discord channels** — `channels/` adapters stubbed; gateway protocol defined but channel adapters not wired.
- **Context handoff improvement** — currently 4B pre-think and 270M schema call are separate; future: stream 4B thinking directly into 270M context for lower latency.

## Built Features (previously planned)

- **Dreaming** — `python main.py dream` runs graph merge + reflection + inner life synthesis + user knowledge extraction + cleanup (`birdclaw/memory/memorise.py`, `birdclaw/memory/inner_life.py`, `birdclaw/memory/cleanup.py`).
- **Self-update** — `birdclaw/agent/self_update.py` — agent reads, modifies, tests, redeploys own source; every change is a git commit.
- **LLM Scheduler** — `birdclaw/llm/scheduler.py` — priority queue gating all `generate()` calls; 5 levels from INTERACTIVE to MEMORY.
- **Dual-model routing** — `birdclaw/llm/model_profile.py` — `ModelProfile` routes each call to the right endpoint; 270M for structure, 4B for reasoning.
- **File cleanup** — `birdclaw/memory/cleanup.py` — retention policy across sessions, tasks, pages, backups, stage history.
- **SearXNG** — local search backend for `web_search` (`/opt/searxng`), installed alongside llama.cpp by `install.sh`.
- **TUI polish** — card-based output pane (ToolCard/StageHeader/PlanBanner/ApprovalCard), GIF buddy companion, theme persistence via `watch_theme`, MCP loaded in background worker, write_doc/write_code progress visible as ToolCards, standing tab deduplication.
- **Skill injection** — `soul_loop._spawn_task()` calls `skill_context(text)` before spawning; matching skill's full `SKILL.md` body injected into agent context (`extra_system`) so agent sees runbook steps instead of hallucinating tools.
- **Prior task carry-forward** — `_recent_task_context()` finds last completed task in session (within 10 min); output snippet injected into new task context to prevent re-research on "do it" follow-ups.
- **Write stage filenames** — fallback output path uses `{task_slug}_s{N}.py/md` where N = write stage count; prevents second write stage from overwriting first.
- **Subtask section markers** — subtask executor injects `Start the section with exactly: ## {item.title}` into item writing prompt so verifier can find sections (was always scoring `parsed_keys=0`).
- **`data_dir` write permission** — `~/.birdclaw/` always writable in `workspace_write` mode; agent can write skill files and task outputs to its own data store.
- **`skills_dir` in dynamic context** — `prompts.dynamic_context()` appends `Skills dir: ~/.birdclaw/skills/ (write skill files here)` so agent knows the target path.
- **Stall guard synthesis detection** — research stage stall guard skips `web_search` for synthesis goals (summarise, consolidate, review gathered content, etc.); uses `web_fetch` when goal contains a URL; expanded `_SYNTHESIS_KW` set.
- **`add_phase_after_current` TUI fix** — dynamically inserted stages set `current_phase_index = insert_at` so new stages show as current in TUI rather than all appearing pre-completed.

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Language | Python 3.10 | Logic Engine patterns directly reusable; fastest path to working system |
| Model backend | llama.cpp (OpenAI-compat API, two instances) | Local, private, no API costs; two ports for 4B + 270M |
| Dual-model split | 4B thinker + 270M hands | 4B always thinks; 270M handles schema — eliminates thinking+format conflict |
| Memory | NetworkX GraphRAG + JSONL session log | No vector DB required; graph navigation works without embeddings |
| Tool safety | Python port of Rust patterns | Rust harness already battle-tested these exact boundaries |
| Context budget | 8192 tokens (of 32K available) | Speed over maximum context; inject only relevant memory slices |
