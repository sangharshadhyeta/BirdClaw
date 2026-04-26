---
name: dreaming
description: Run BirdClaw's nightly dream cycle — memory consolidation, knowledge graph merge, inner life synthesis, user knowledge extraction, and cleanup
tags: [dream, memory, maintenance, nightly, consolidation]
schedule: "0 3 * * *"
enabled: true
---

Run the full dream cycle and report results.

1. Run `python /opt/birdclaw/main.py dream` and capture stdout/stderr
2. Report which phases completed (memorise, graph, inner_life, user_knowledge, self_concept, cleanup)
3. Note any errors — if a phase failed, include the error message
4. Keep the summary brief: one line per phase, status and key result only
