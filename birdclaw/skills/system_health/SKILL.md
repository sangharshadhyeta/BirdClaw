---
name: system_health
description: Check system resource usage — disk, memory, CPU load, and failed services
tags: [health, system, disk, memory, cpu, monitor]
schedule: "0 9 * * *"
enabled: false
---

Check the system's health and produce a brief report. Do all of these:

1. Run `df -h` — report any filesystem at >80% usage
2. Run `free -h` — note if available memory is below 500MB
3. Run `uptime` — report load average
4. Run `systemctl --failed --no-legend` (if systemd is present) — list any failed units

Summarise findings in 3–5 bullet points. If everything is healthy, say so briefly.
Do not run any commands that modify the system.
