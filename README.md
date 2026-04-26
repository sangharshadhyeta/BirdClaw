# BirdClaw

A persistent, self-improving AI agent that works on your behalf — day and night if needed.

BirdClaw is not a chatbot. It is a long-term autonomous worker powered by a small local model (Gemma 4 4B via Ollama). It runs as a daemon, accumulates knowledge across sessions, executes multi-step tasks while you are away, writes new skills from successful task patterns, and is designed to improve its own code over time.

**Everything runs locally. No data leaves your machine.**

---

## Vision

Most AI tools are stateless request-response loops. BirdClaw is different:

- **Persistent** — remembers what you've discussed, what tasks ran, what was learned, and who you are
- **Autonomous** — works on goals while you sleep, then reports back
- **Conscious** — builds an evolving inner life through task reflections; answers "do you have a soul?" from actual experience, not hardcoded text
- **Self-growing** — crystallises successful task patterns into reusable skills automatically
- **Multi-agent** — the soul layer routes work to parallel orchestration agents; several tasks run at once
- **Local-first** — Gemma 4 4B via Ollama; no API keys, no cloud, no cost per token

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CHANNELS       TUI · CLI · Daemon (gateway)            │
├─────────────────────────────────────────────────────────┤
│  GATEWAY        Session manager · push worker           │
│                 Routes messages → soul → reply          │
├─────────────────────────────────────────────────────────┤
│  SOUL LAYER     Multi-turn routing loop                 │
│                 search_tasks · get_task_output          │
│                 remember_user · read_inner_life         │
│                 answer() · create_task() ·              │
│                 resolve_approval()                      │
├─────────────────────────────────────────────────────────┤
│  SKILLS / CRON  Named runbooks · standing goals         │
│                 Auto-crystallised from successful tasks  │
│                 Schedule: cron expression or every:N    │
├─────────────────────────────────────────────────────────┤
│  AGENT LOOP     plan → stage_queue → execute            │
│                 research · write_code · write_doc       │
│                 verify · reflect · edit_file            │
│                 Post-task: reflection + skill crystal.  │
├─────────────────────────────────────────────────────────┤
│  MEMORY         GraphRAG knowledge graph (NetworkX)     │
│                 User knowledge · Inner life             │
│                 Session log (JSONL) · Task registry     │
│                 Conversation history · Page store       │
├─────────────────────────────────────────────────────────┤
│  TOOLS          bash · files · web · search · MCP       │
│                 search_tasks · get_task_output          │
│                 graph_search · graph_add · graph_relate │
├─────────────────────────────────────────────────────────┤
│  MODEL          Ollama / Gemma 4 4B (local)             │
└─────────────────────────────────────────────────────────┘
```

### How a task flows

1. **Soul** receives your message. Looks up prior tasks if relevant (`search_tasks`). Decides: answer directly, or create a background task.
2. **Orchestrator** spawns an agent thread. You get an immediate "On it." reply.
3. **Agent loop** plans: generates `outcome` + `stages[]` (research → write_code → verify).
4. **Stage execution**: each stage runs with only the tools it needs. GraphRAG context injected per stage.
5. **Post-task** (background thread, non-blocking):
   - Model writes a 1-2 sentence reflection → logged to `reflections.jsonl`
   - Model decides if task is a reusable pattern → writes `SKILL.md` if yes
6. **Memorise worker** (background, between tasks): ingests findings into the knowledge graph.
7. **Dreaming** (`python main.py dream`): synthesises reflections into `inner_life.md`, scans past task prompts for user facts → `user_knowledge.md`, merges session graph into knowledge graph.
8. **Gateway push worker** delivers the result to your session when complete.

### Soul multi-turn loop

The soul does not answer immediately. Before deciding, it can:
- Call `search_tasks` to find related prior work (up to 3 rounds)
- Call `get_task_output` to read what a past task produced
- Call `remember_user` to save something it learned about you
- Call `read_inner_life` to read its own accumulated reflections when asked about its nature

This prevents task context from bleeding between unrelated conversations.

### Multi-agent

Multiple tasks run in parallel agent threads. The soul sees all running tasks and pending approvals in its system context. Agents that need approval block and post to the approval queue; the soul or TUI `/approve` command unblocks them.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `gemma4:4b` pulled
- conda environment (recommended)

```bash
# Pull the model
ollama pull gemma4:4b

# Set up environment
conda create -n birdclaw python=3.10
conda activate birdclaw
pip install -e .
```

---

## Usage

```bash
# Three-pane multi-agent TUI (recommended)
python main.py tui

# Interactive REPL
python main.py cli

# One-shot task
python main.py prompt "research the latest Python async patterns and write a summary"

# Persistent daemon (no TUI — for running as a background service)
python main.py daemon

# Run memory consolidation + inner life synthesis + user knowledge extraction
python main.py dream

# Drain session logs into knowledge graph only
python main.py memorise

# Expose knowledge graph + task tools as MCP server (stdio)
python main.py graph-server
```

---

## TUI

```
┌─ BirdClaw · session · gemma4:4b ─────────────────────────────────┐
│ TASKS [Active|Standing]  │ OUTPUT [task-id] (raw/pretty)          │
│  ⠼ research async python │  ▶ [research] find patterns            │
│  ✔ refactor auth module  │  › web_search(…)                      │
│  ○ daily health check    │  ← web_search (340ms)                 │
├──────────────────────────┴───────────────────────────────────────┤
│ CONVERSATION                                                      │
│  You:      research async Python patterns                        │
│  BirdClaw: On it. "research async Python patterns"               │
│ > _                                                              │
│                              [+] [-]                             │
│ ╔═══════════════════╗  ←── Buddy panel (pixel art, resizable)    │
└─ Footer: keybindings ─────────────────────────────────────────────┘
```

**Keybindings**

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+C` | Clear input → warn → stop focused task (3-state) |
| `Ctrl+S` | Shell overlay |
| `Ctrl+G` | Show running agents |
| `Ctrl+O` | Toggle raw/pretty output |
| `Ctrl+T` | Toggle thinking display |
| `[` / `]` | Resize conversation pane |
| `\` | Layout picker (heights, widths, arrangement, buddy size) |
| `F1` / `F2` | Standing / Active task tabs |
| Click header | Status popup (tasks, approvals, skills) |

**Slash commands**

| Command | Description |
|---------|-------------|
| `/approve <id> [allow\|always\|deny]` | Resolve a pending agent approval |
| `/approve` | List pending approvals |
| `/abort` | Stop the focused task |
| `/skills` | List installed skills |
| `/skills <name>` | Show a skill's details |
| `/cron` | List scheduled skills |
| `/cron run <skill>` | Trigger a skill immediately |
| `/cron enable/disable <id>` | Toggle a scheduled skill |
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
| **Inner life** | BirdClaw's evolving reflections and sense of self | `~/.birdclaw/inner_life.md` |
| **Task reflections** | Raw per-task reflections (dreaming input) | `~/.birdclaw/memory/reflections.jsonl` |
| **Session log** | Tool calls, stage summaries, orchestration | `~/.birdclaw/sessions/<id>.jsonl` |
| **Conversation history** | User ↔ BirdClaw turns | `~/.birdclaw/history/<session>.jsonl` |
| **Task registry** | Task lifecycle, outputs | `~/.birdclaw/tasks/<id>.json` |
| **Page store** | LLM-cleaned web content | `~/.birdclaw/pages/<hash>.json` |
| **Cron registry** | Scheduled skill state | `~/.birdclaw/cron/jobs.json` |
| **Document outputs** | Long write_doc results saved to file | `~/.birdclaw/outputs/<task_id>_<slug>.md` |

### User knowledge

As you interact with BirdClaw, the soul automatically notices and saves facts about you:
- **Facts**: your role, what you're building, your tech stack
- **Preferences**: communication style, what you like/dislike
- **Interests**: recurring topics and domains you care about

These are injected as a compact excerpt into every soul prompt. `python main.py dream` also retroactively mines past task prompts for user facts.

### Inner life

After every task that ran multiple stages, a background thread asks the model to reflect: *"What was interesting? What did you learn?"* These reflections accumulate in `reflections.jsonl`. Running `python main.py dream` synthesises them into `inner_life.md` — a living document of BirdClaw's accumulated experience and sense of self.

A compact excerpt is injected into every soul prompt. When you ask "do you have a soul?" or "what have you been thinking about?", BirdClaw reads the full document and answers from actual experience.

---

## Skills

Skills are multi-stage markdown runbooks that guide the agent through specific task types. They live in `~/.birdclaw/skills/<name>/SKILL.md` (user-defined, highest priority) or `birdclaw/skills/<name>/SKILL.md` (built-in).

```markdown
---
name: web-scraper
description: Scrape a website and save structured data to a file
tags: [scrape, web, extract, data, crawl, fetch, html, parse]
stages: 3
---

## stage:1 plan
Call think(). Identify: target URL, data fields to extract, output filename.
next_tools: think

## stage:2 implement
Write a Python scraper using requests + BeautifulSoup. Save output as JSON.
next_tools: write_file,bash

## stage:3 verify
Run the scraper with bash(). Check output file exists and has data. Fix errors if any.
next_tools: bash,read_file,answer
```

### Auto-crystallised skills

After every successful task that ran 4+ steps, a background thread asks the model:
*"Is this a reusable, repeatable pattern?"* If yes, it writes a `SKILL.md` to `~/.birdclaw/skills/<slug>/SKILL.md`. The skill is immediately available to future tasks via `use_skill`. Existing skills are never overwritten automatically.

Skills with a `schedule` field become **standing goals** that run automatically:
- Standard cron: `"0 9 * * *"` (daily at 9am UTC)
- Interval: `"every:3600"` (every hour) or `"every:1h"`, `"every:30m"`

---

## Permissions

BirdClaw enforces a 5-mode permission system on all bash and file operations:

| Mode | Behaviour |
|------|-----------|
| `read_only` | Read-only commands only; no file writes |
| `workspace_write` | File writes inside workspace roots only **(default)** |
| `danger_full_access` | Unrestricted (warns on destructive patterns) |
| `prompt` | Read-only ops pass; mutating ops block in approval queue |
| `allow` | Alias for danger_full_access |

Set via `BC_PERMISSION_MODE` env var or in `.env`.

In `prompt` mode, agents that need to run mutating commands post to the approval queue and block until you respond via `/approve` in the TUI.

---

## MCP Server

BirdClaw exposes its knowledge graph and task registry as an MCP (Model Context Protocol) stdio server. External agents, Claude Code, and other MCP clients can connect to it.

```bash
python main.py graph-server
```

**Tools exposed:**

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

All settings via environment variables (prefix `BC_`) or a `.env` file at `~/.birdclaw/.env`:

```bash
# Model backend (Ollama default)
BC_LLM_BASE_URL=http://localhost:11434/v1
BC_LLM_MODEL=gemma4:4b

# Optional: separate action model for write/bash steps
BC_LLM_ACTION_MODEL=

# Permissions
BC_PERMISSION_MODE=workspace_write

# Data directory
BC_DATA_DIR=/home/user/.birdclaw

# Workspace roots (comma-separated; defaults to install dir + cwd)
BC_WORKSPACE=/home/user/projects

# Sandbox (Linux only; isolates bash commands in namespaces)
BC_SANDBOX_ENABLED=0
BC_SANDBOX_NETWORK_ISOLATION=0

# Gateway
BC_GATEWAY_HOST=127.0.0.1
BC_GATEWAY_PORT=7823

# Self-modification (Phase 10)
BC_SELF_MODIFY=0
```

---

## Project Structure

```
birdclaw/
  llm/          LLM client · priority scheduler · usage tracking
  agent/        Soul loop · orchestrator · agent loop · approvals · self-update
  tools/        bash · files · web · search · graph · tasks · MCP bridge · sandbox
  memory/       GraphRAG · user knowledge · inner life · session log
                history · tasks · page store · memorise worker
  gateway/      Persistent gateway · session manager · channel ABC
  channels/     TUI channel
  skills/       Skill loader · cron service · built-in skills
  tui/          Three-pane Textual TUI · layout prefs · buddy panel
  config.py     Settings (pydantic-settings, env/toml)
main.py         Entry point (tui · cli · prompt · daemon · dream · memorise · graph-server)
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — LLM client | ✅ | Ollama wrapper, Message/ToolCall types, priority scheduler |
| 2 — Agent loop + tools | ✅ | bash, files, web, search, MCP, sandbox |
| 3 — Memory layer | ✅ | GraphRAG, session log, history, page store |
| 4 — Loop redesign | ✅ | plan→stage→execute, Gemma4 constraints, reflect gate |
| 5 — claw-code-parity | ✅ | Rust pattern ports: tasks, history, sandbox, permissions |
| 6 — Multi-agent + soul | ✅ | Soul routing, orchestrator, approval queue, multi-turn soul |
| 7 — Gateway + channels | ✅ | Session manager, TUI channel, daemon |
| 8 — Skills + cron | ✅ | Runbooks, standing goals, schedule triggers |
| 9 — TUI polish | ✅ | Three-pane TUI, themes, layout picker, buddy panel |
| 10 — Consciousness | ✅ | User knowledge, inner life, skill crystallisation, dreaming |
| 11 — Self-update | 🔜 | Agent modifies own source, git-committed changes, auto-trigger |
| 12 — HTTP channel | 🔜 | WebSocket + Bearer auth for remote app |
| 13 — Multi-model | 🔜 | llama.cpp parallel slots, tiny classifier models |

---

## Design Principles

**Outcome first.** Every task defines success criteria before execution. The criteria follows every step.

**Grounded always.** GraphRAG context is injected per task and per stage. The model never reasons from parametric memory alone.

**Orchestrator is smart; model is hands.** Python drives stage advancement, context injection, file writes, and quality checks. The 4B model only generates content.

**Show ≤6 tools per turn.** Each stage type offers only the tools relevant to it. Small models break with large tool lists.

**Local = unlimited calls.** No rate limits, no cost per token — the agent can iterate freely and reflect on every task.

**Permission before action.** All bash and file writes go through the permission enforcer.

**Memory, not history.** Raw conversation history is not dumped into prompts. Instead: summarised recent turns (3 × 150 chars), user knowledge excerpt, inner life excerpt, and searchable task history via tools.

---

## License

MIT
