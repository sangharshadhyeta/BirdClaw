# 🐦 BirdClaw

**A persistent, self-improving AI agent that works on your behalf — day and night.**

BirdClaw is not a chatbot. It is a long-term autonomous worker powered by local language models via llama.cpp. It runs as a daemon, accumulates knowledge across sessions, executes multi-step tasks while you are away, writes its own skills from successful patterns, and improves its own code over time.

**Everything runs locally. No API keys. No cloud. No cost per token.**

---

## What makes it different

| Other AI tools | BirdClaw |
|----------------|----------|
| Stateless — forgets after every response | Persistent — remembers tasks, facts, reflections across sessions |
| One model, one role | Dual-model: 4B thinker reasons, 270M hands formats |
| You stay online to get results | Daemon runs while you sleep, notifies when done |
| Fixed capabilities | Self-update cycle — agent patches its own source code |
| Blind to its own code | Reads its own source at runtime; knows what it can and can't do |
| You define all tools | Crystallises new skills from successful task patterns automatically |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CHANNELS        TUI · CLI · Daemon (gateway socket)            │
├─────────────────────────────────────────────────────────────────┤
│  GATEWAY         Session manager · push worker                  │
│                  Routes messages → soul → reply                 │
├─────────────────────────────────────────────────────────────────┤
│  SOUL LAYER      270M-first routing loop (fast)                 │
│                  search_tasks · get_task_output                 │
│                  remember_user · read_inner_life                │
│                  answer() · create_task() · resolve_approval()  │
│                  Escalates to 4B for deep questions             │
├─────────────────────────────────────────────────────────────────┤
│  SKILLS / CRON   Named markdown runbooks · standing goals       │
│                  Auto-crystallised from successful task pattern │
│                  Scheduled: cron expression or every:N          │
├─────────────────────────────────────────────────────────────────┤
│  AGENT LOOP      plan → stage_queue → execute                   │
│                  research · write_code · write_doc              │
│                  verify · reflect · edit_file                   │
│                  Post-stage reflect gate (270M quality check)   │
│                  Budget tracking · auto-compaction              │
├─────────────────────────────────────────────────────────────────┤
│  MEMORY          GraphRAG knowledge graph (NetworkX, no vectors)│
│                  User knowledge · Inner life · Self-concept     │
│                  Session log (JSONL) · Task registry            │
│                  Stage history (P75 budget learning)            │
├─────────────────────────────────────────────────────────────────┤
│  TOOLS           bash · files · web · search (SearXNG) · MCP    │
│                  search_tasks · get_task_output                 │
│                  graph_search · graph_add · graph_relate        │
│                  note_improvement (self-update backlog)         │
├─────────────────────────────────────────────────────────────────┤
│  SELF-UPDATE     Pain-point scoring · agent patches own source  │
│                  pytest gate · install.sh hot-reload            │
│                  Workspace isolation · backup/restore           │
├─────────────────────────────────────────────────────────────────┤
│  MODELS          llama.cpp · 4B thinker (port 8081)             │
│                  270M hands / format worker (port 8082)         │
└─────────────────────────────────────────────────────────────────┘
```

### Dual-model design

Every LLM call is routed through a `ModelProfile`:

| Call | Model | Why |
|------|-------|-----|
| Soul routing | 270M | 3-class classification — no reasoning needed |
| Plan structuring | 4B pre-think → 270M schema | 4B reasons, 270M enforces JSON schema |
| Research / verify / reflect stages | 4B, thinking=True | Full reasoning on every turn |
| Post-stage reflect gate | 270M | 4-class quality check on 4B output |
| Content generation (write_code, write_doc) | 4B, thinking=True | Only the 4B writes coherent code/prose |
| Forced final answer | 4B, thinking=True | Full synthesis of completed stages |

The 270M never reasons — it only converts 4B output into schema-valid JSON. This eliminates the llama.cpp thinking+format conflict entirely. When `BC_LLM_HANDS_BASE_URL` is not set, it falls back to the 4B — everything works on a single model, just slightly slower.

### How a task flows

1. **Soul** receives your message. Checks prior tasks (`search_tasks`). Decides: answer directly or create a background task.
2. **Orchestrator** spawns an agent thread. You get an immediate reply.
3. **Planner** generates `outcome` + `stages[]` (e.g. research → write_code → verify). Skill runbook injected if a matching skill exists.
4. **Stage execution**: each stage runs with only the tools it needs. GraphRAG context injected per stage goal.
5. **Post-stage reflect gate**: 270M evaluates quality → `continue`, `deepen` (re-run), `insert` (add a stage), or `done`.
6. **Forced answer**: when queue is empty, 4B synthesises all completed stage summaries with `thinking=True`.
7. **Dream cycle** (background): graph merge → reflection → inner life synthesis → user knowledge extraction → cleanup.

---

## Requirements

- Linux (systemd recommended for daemon mode; bare metal works too)
- NVIDIA GPU with CUDA toolkit **or** Vulkan-capable GPU **or** CPU-only
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.10+
- `sudo` access for the installer (installs to `/opt/birdclaw` by default)

The installer handles everything else: llama.cpp build, model download, SearXNG, systemd services, conda env.

---

## Installation

```bash
git clone https://github.com/sangharshadhyeta/BirdClaw.git
cd BirdClaw
sudo ./install.sh install
```

The installer will:
1. Detect your GPU (CUDA / Vulkan / ROCm / CPU) and install the right llama.cpp build
2. Download the Gemma 2B GGUF model (or prompt for a custom path)
3. Create a conda environment at `/opt/birdclaw/env/`
4. Install SearXNG as the local search backend
5. Register three systemd services: `llama-server`, `birdclaw-daemon`, `searxng`
6. Write `~/.birdclaw/.env` with your chosen ports and paths
7. Install a `/usr/local/bin/birdclaw` CLI wrapper

**Non-interactive install** (accepts all defaults):

```bash
sudo BC_YES=1 ./install.sh install
```

**After install, launch the TUI:**

```bash
birdclaw tui
# or
python main.py tui
```

### Update

```bash
sudo ./install.sh update
```

Syncs source → `/opt/birdclaw/`, updates Python deps, restarts services.

### Manual / dev setup (no sudo)

```bash
conda create -n claude-code python=3.10
conda activate claude-code
pip install -e .

# Point at a running llama.cpp instance
export BC_LLM_BASE_URL=http://localhost:8081/v1
export BC_LLM_MODEL=gemma-2b

python main.py tui
```

---

## Usage

```bash
# Three-pane multi-agent TUI (recommended)
python main.py tui

# Interactive REPL
python main.py cli

# One-shot task (no soul layer — direct agent loop)
python main.py prompt "research the latest Python async patterns and write a summary"

# Persistent daemon (no TUI — for background service use)
python main.py daemon

# Full dream cycle: graph merge + reflection + inner life + user knowledge + cleanup
python main.py dream

# Drain session logs into the knowledge graph only
python main.py memorise [session_id]

# Run one self-update cycle (hypothesis → patch → test → accept/revert)
python main.py self-update

# Prune stale sessions, tasks, pages, and self-update backups
python main.py cleanup

# Expose knowledge graph + task tools as MCP stdio server
python main.py graph-server
```

---

## TUI

```
┌─ 🐦 BirdClaw · tui:a1b2c3 ──────────────────────────────────────────┐
│ TASKS [Active|Standing]    │ OUTPUT [task-id]                        │
│  ⠼ research async python   │  ▷ [plan] research → write_code        │
│    ↳ ⠼ web search subtask  │  › web_search("async Python patterns"  │
│  ✔ refactor auth module    │  ← web_search  (340ms, 4 results)      │
│  ○ daily health check      │  ▶ [write_code] writing module         │
├────────────────────────────┴─────────────────────────────────────────┤
│ CONVERSATION                                                         │
│  You:      research async Python patterns and write a module         │
│  BirdClaw: On it. Spawned task [a1b2c3]                              │
│ > _                                                                  │
│                              [+] [-]                                 │
│ ╔═══════════════════╗  ←── Buddy panel (pixel-art bird, resizable)   │
└─ F1 tasks · F2 standing · Ctrl+Q quit ───────────────────────────────┘
```

**Keybindings**

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+C` | Clear input → warn → stop focused task (3-state) |
| `Ctrl+S` | Shell overlay |
| `Ctrl+G` | Show running agents |
| `Ctrl+O` | Toggle raw / pretty output |
| `Ctrl+T` | Toggle thinking display |
| `[` / `]` | Resize conversation pane |
| `\` | Layout picker (heights, widths, arrangement, buddy size) |
| `F1` / `F2` | Standing / Active task tabs |
| Click header | Status popup (tasks, approvals, skills) |

**Slash commands**

| Command | Description |
|---------|-------------|
| `/approve [id] [allow\|always\|deny]` | Resolve a pending approval (or list all) |
| `/abort` | Stop the focused task |
| `/skills [name]` | List skills or show a skill's details |
| `/cron` | List scheduled skills |
| `/cron run <skill>` | Trigger a skill immediately |
| `/cron enable\|disable <id>` | Toggle a scheduled skill |
| `/tasks` | Show all tasks |
| `/clear` | Clear conversation log |
| `/shell` | Open shell overlay |
| `?` | Help overlay |

---

## Memory

BirdClaw maintains several memory layers:

| Layer | What | Where |
|-------|------|-------|
| **Knowledge graph** | Entities, facts, relationships from research | `~/.birdclaw/memory/graph.json` |
| **User knowledge** | Facts about you revealed through conversation | `~/.birdclaw/user_knowledge.md` |
| **Inner life** | Evolving reflections and accumulated experience | `~/.birdclaw/inner_life.md` |
| **Self-concept** | Agent's sense of identity and capabilities | `~/.birdclaw/self_concept.md` |
| **Session log** | Tool calls, stage summaries, orchestration events | `~/.birdclaw/sessions/<id>.jsonl` |
| **Task registry** | Task lifecycle, outputs, status | `~/.birdclaw/tasks/<id>.json` |
| **Stage history** | Per-stage-type step counts for P75 budget learning | `~/.birdclaw/memory/stage_history.jsonl` |
| **Page store** | LLM-cleaned web content cache (24h TTL) | `~/.birdclaw/pages/<hash>.json` |
| **Self-update backlog** | Capability gaps noted by agent during tasks | `~/.birdclaw/self_update_todo.jsonl` |
| **Self-update history** | Record of accepted patches and pass-rate deltas | `~/.birdclaw/self_update/history.jsonl` |

### Dreaming

`python main.py dream` runs six phases in sequence:

1. **Memorise** — ingests recent session logs into the knowledge graph
2. **Graph** — merges and deduplicates graph nodes
3. **Inner life** — synthesises task reflections into `inner_life.md`
4. **User knowledge** — mines conversation history for facts about you
5. **Self-concept** — updates the agent's sense of its own capabilities
6. **Cleanup** — prunes stale sessions, old tasks, expired pages, excess backups

A compact excerpt from each memory layer is injected into every soul prompt. When you ask "do you have a soul?" or "what have you been thinking about?", BirdClaw reads the full document and answers from actual accumulated experience — not hardcoded text.

### Self-awareness

BirdClaw can read its own source code (`birdclaw/`) at any time. This means it knows:
- What tools exist and how they're implemented
- Why certain fallbacks fire — and how to avoid them
- Where capability gaps are in the current code

When it hits something it can't do, it calls `note_improvement(description, priority)` to log it to the self-update backlog instead of silently degrading.

---

## Self-Update

`python main.py self-update` runs one improvement cycle:

```
1. Score pain points (Python — deterministic):
   ├── self_update_todo.jsonl  — gaps noted during normal tasks  (highest priority)
   ├── stage_history.jsonl     — stage types with high budget exhaustion rate
   └── birdclaw.log            — ERROR lines, tracebacks, failed tasks

2. Snapshot current source → ~/.birdclaw/self_update/backup_<ts>/

3. Run agent loop against birdclaw/ source tree:
   - Reads its own source to understand the problem
   - Uses edit_file / write_code stages exactly like any other task
   - Verifies syntax with py_compile after each change

4. pytest gate — python -m pytest tests/test.py -m "not long_running"
   ├── Pass + no regression → accept patch
   └── Fail or regression   → restore from backup

5. On accept:
   - install.sh update --yes  (rsync → /opt/birdclaw/ + restart daemon)
   - Mark todo item done
   - Log to ~/.birdclaw/self_update/history.jsonl
```

**Safety model:**
- `BC_SELF_MODIFY=1` required — off by default
- All writes to `birdclaw/` blocked by the permission enforcer unless self_modify is enabled
- `soul_constitution.py` is constitutionally protected — self-update can never touch it
- Every attempt creates a backup; any failure triggers atomic restore
- Workspace directory scrubbed after each attempt — no garbage left in the repo

---

## Skills

Skills are markdown runbooks that guide the agent through specific task types.

**Locations** (highest priority first):
1. `~/.birdclaw/skills/<name>/SKILL.md` — user-defined
2. `birdclaw/skills/<name>/SKILL.md` — built-in

**Built-in skills:**
- `code-generation` — surgical code writing, one function at a time
- `document-creation` — section-by-section document writing
- `system_health` — daily disk / memory / CPU / service check (runs at 9am UTC)

**Example skill:**

```markdown
---
name: api-client
description: Write a Python API client with error handling and retries
tags: [api, client, http, requests, python, integration]
stages: 3
---

## stage:1 plan
Call think(). List: base URL, endpoints, auth method, output filename.

## stage:2 implement
Write the client using requests + tenacity for retries. One method at a time.
next_tools: write_file, edit_file

## stage:3 verify
Run: bash("python -m py_compile <file>"). Fix errors. Then answer().
next_tools: bash, answer
```

Skills with a `schedule` field become **standing goals** that run automatically:

```markdown
schedule: "0 9 * * *"    # cron: daily at 9am UTC
schedule: "every:3600"   # interval: every hour
schedule: "every:30m"    # shorthand
```

---

## Permissions

| Mode | Behaviour |
|------|-----------|
| `read_only` | Read-only commands only; no file writes |
| `workspace_write` | Writes inside workspace roots only **(default)** |
| `danger_full_access` | Unrestricted (still warns on destructive patterns) |
| `prompt` | Read-only ops pass; mutating ops block in approval queue |
| `allow` | Alias for `danger_full_access` |

`birdclaw/` source is always readable (self-awareness), but writes require `BC_SELF_MODIFY=1`.  
`~/.birdclaw/` data directory is always writable.

Set via `BC_PERMISSION_MODE` in `.env` or environment.

---

## MCP Server

```bash
python main.py graph-server
```

Exposes knowledge graph and task registry as an MCP stdio server. Claude Code, Cursor, and any MCP client can connect.

| Tool | Description |
|------|-------------|
| `graph_search` | Search knowledge graph nodes by keyword |
| `graph_get` | Retrieve a specific knowledge node |
| `graph_add` | Add or update a knowledge node |
| `graph_relate` | Link two knowledge nodes |
| `search_tasks` | Search past and current tasks by keyword |
| `get_task_output` | Get full output + saved document path for a task |

---

## Configuration

All settings via `BC_` environment variables or `~/.birdclaw/.env`:

```bash
# ── Models ────────────────────────────────────────────────────────────
BC_LLM_BASE_URL=http://localhost:8081/v1     # 4B thinker (main model)
BC_LLM_MODEL=gemma-2b                        # model name sent to API
BC_LLM_HANDS_BASE_URL=http://localhost:8082/v1  # 270M hands (optional)
BC_LLM_HANDS_MODEL=functiongemma-270m

# ── Performance ───────────────────────────────────────────────────────
BC_LLAMACPP_PARALLEL=1         # parallel slots per server
BC_LLAMACPP_GPU_LAYERS=99      # layers to offload to GPU
BC_N_CTX=32768                 # context window per slot
BC_MAX_TOKENS=4096             # output token budget
BC_TEMPERATURE=0.0             # deterministic (recommended for tool calling)

# ── Workspace ─────────────────────────────────────────────────────────
BC_WORKSPACE=/home/user/projects    # comma-separated workspace roots
BC_PERMISSION_MODE=workspace_write  # read_only | workspace_write | danger_full_access | prompt

# ── Search ────────────────────────────────────────────────────────────
BC_SEARXNG_URL=http://localhost:8888

# ── Gateway ───────────────────────────────────────────────────────────
BC_GATEWAY_HOST=127.0.0.1
BC_GATEWAY_PORT=7823

# ── Self-modification ─────────────────────────────────────────────────
BC_SELF_MODIFY=0    # set to 1 to enable self-update cycle

# ── Sandbox ───────────────────────────────────────────────────────────
BC_SANDBOX_ENABLED=0                # Linux namespace isolation for bash
BC_SANDBOX_NETWORK_ISOLATION=0

# ── Scheduler ─────────────────────────────────────────────────────────
BC_LLM_SCHEDULER_ENABLED=true   # set false to bypass in tests/dev

# ── TUI ───────────────────────────────────────────────────────────────
BC_THEME=dark    # dark | light | solarized | catppuccin
```

---

## Project Structure

```
birdclaw/
  llm/          LLM client · priority scheduler · model profiles · usage tracking
  agent/        Soul loop · orchestrator · agent loop · planner · budget
                subtask pipeline · approvals · self-update · soul constitution
  tools/        bash · files · web · search · graph · tasks · MCP bridge
                permission enforcer · sandbox · hooks · write guard
  memory/       GraphRAG · user knowledge · inner life · self-concept
                session log · history · tasks · page store · dream cycle
                memorise worker · cleanup · compact
  gateway/      Persistent gateway · session manager · channel ABC · notify
  channels/     TUI channel · TUI socket channel
  skills/       Skill loader · cron scheduler · built-in skills
  tui/          Three-pane Textual TUI · cards · overlays · layout prefs · buddy
  config.py     Settings (pydantic-settings, BC_* env vars / .env)
main.py         Entry point — all subcommands
install.sh      Full installer (llama.cpp · models · SearXNG · systemd · conda)
tests/test.py   369 unit + component tests (regression gate for self-update)
```

---

## Development

```bash
# Run all fast tests
pytest tests/test.py -m "not long_running" -q

# Run a single test
pytest tests/test.py::TestSubtaskManifest::test_mark_complete_sets_hash -x

# Disable LLM scheduler in dev
BC_LLM_SCHEDULER_ENABLED=false python main.py cli
```

Tests are the gate for the self-update cycle — a patch is only accepted if all 369 pass with no regression.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — LLM client | ✅ | OpenAI-compatible client, Message/ToolCall types, priority scheduler |
| 2 — Agent loop + tools | ✅ | bash, files, web, search, MCP bridge, sandbox |
| 3 — Memory layer | ✅ | GraphRAG, session log, history, page store |
| 4 — Loop redesign | ✅ | plan→stage→execute, reflect gate, budget tracking |
| 5 — claw-code-parity | ✅ | Rust pattern ports: tasks, history, sandbox, permissions |
| 6 — Multi-agent + soul | ✅ | Soul routing, orchestrator, approval queue, multi-turn soul |
| 7 — Gateway + channels | ✅ | Session manager, TUI channel, daemon |
| 8 — Skills + cron | ✅ | Runbooks, standing goals, schedule triggers, skill injection |
| 9 — TUI polish | ✅ | Three-pane TUI, themes, layout picker, buddy panel, grandchild tasks |
| 10 — Consciousness | ✅ | User knowledge, inner life, self-concept, dreaming, skill crystallisation |
| 11 — Dual-model | ✅ | 4B thinker + 270M hands, model profiles, hands fallback |
| 12 — Self-update | ✅ | Pain-point scoring, agent patches own code, pytest gate, hot-reload |
| 13 — Self-awareness | ✅ | Reads own source, note_improvement tool, self-update backlog |
| 14 — Telegram / Discord | 🔜 | Channel adapters (stubs exist in `channels/`) |
| 15 — HTTP channel | 🔜 | WebSocket + Bearer auth for remote app / mobile |

---

## Design Principles

**Outcome first.** Every task defines success criteria before execution. The criteria follows every step.

**Grounded always.** GraphRAG context injected per task and per stage. The model never reasons from parametric memory alone.

**Orchestrator is smart; model is hands.** Python drives stage advancement, context injection, file writes, and quality checks. The LLM only generates content and makes decisions within a constrained tool set.

**Show ≤ 6 tools per turn.** Each stage type offers only the tools relevant to it. Small models break with large tool lists.

**Local = unlimited calls.** No rate limits, no cost per token — the agent can iterate freely, reflect on every task, and improve itself overnight.

**Permission before action.** All bash and file writes go through the permission enforcer. Self-modification requires an explicit opt-in flag.

**Memory, not history.** Raw conversation history is never dumped into prompts. Instead: summarised recent turns, user knowledge excerpt, inner life excerpt, and searchable task history via tools.

**Self-improvement is just another task.** The self-update cycle uses the same agent loop, the same stage types, the same tools as any other task — aimed at the source tree.

---

## License

MIT
