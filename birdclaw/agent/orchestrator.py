"""Orchestrator — manages the pool of running agent threads.

The soul layer spawns orchestration agents here. Each agent runs
run_agent_loop in a daemon thread with its own SessionLog and an
interrupt event the soul can signal at any time.

Usage:
    from birdclaw.agent.orchestrator import orchestrator

    agent_id = orchestrator.spawn(task_id, prompt, context, expected_outcome)
    orchestrator.interrupt(agent_id)   # signal graceful stop
    orchestrator.is_running(agent_id)  # True/False
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent handle
# ---------------------------------------------------------------------------

@dataclass
class AgentHandle:
    agent_id:   str
    task_id:    str
    thread:     threading.Thread
    interrupt:  threading.Event
    started_at: float = field(default_factory=time.time)

    def is_alive(self) -> bool:
        return self.thread.is_alive()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Thread-safe registry of running agent threads."""

    def __init__(self) -> None:
        self._lock:   threading.Lock              = threading.Lock()
        self._agents: dict[str, AgentHandle]      = {}
        # Semaphore(1) = serial; Semaphore(999) = effectively unlimited.
        # Controlled by BC_PARALLEL_TASKS in .env / environment.
        from birdclaw.config import settings
        limit = 999 if settings.parallel_tasks else 1
        self._serial: threading.Semaphore         = threading.Semaphore(limit)

    # ── Spawn ─────────────────────────────────────────────────────────────────

    def spawn(
        self,
        task_id:          str,
        prompt:           str,
        context:          str = "",
        expected_outcome: str = "",
        launch_cwd:       str = "",
    ) -> str:
        """Start an orchestration agent for task_id. Returns agent_id."""
        agent_id      = f"agent_{uuid.uuid4().hex[:12]}"
        interrupt_evt = threading.Event()

        t = threading.Thread(
            target=self._run,
            args=(agent_id, task_id, prompt, context, expected_outcome, interrupt_evt, launch_cwd),
            daemon=True,
            name=f"agent-{agent_id[:8]}",
        )

        handle = AgentHandle(
            agent_id=agent_id,
            task_id=task_id,
            thread=t,
            interrupt=interrupt_evt,
        )

        with self._lock:
            self._cleanup_finished()
            self._agents[agent_id] = handle

        t.start()
        logger.info("orchestrator: spawned %s for task %s", agent_id, task_id)
        return agent_id

    # ── Control ───────────────────────────────────────────────────────────────

    def interrupt(self, agent_id: str) -> bool:
        """Signal an agent to stop gracefully. Returns True if found."""
        with self._lock:
            handle = self._agents.get(agent_id)
        if handle is None:
            return False
        handle.interrupt.set()
        logger.info("orchestrator: interrupt sent to %s", agent_id)
        return True

    def interrupt_by_task(self, task_id: str) -> bool:
        """Interrupt whichever agent is running task_id."""
        with self._lock:
            for handle in self._agents.values():
                if handle.task_id == task_id:
                    handle.interrupt.set()
                    logger.info("orchestrator: interrupt task %s → agent %s",
                                task_id, handle.agent_id)
                    return True
        return False

    def is_running(self, agent_id: str) -> bool:
        with self._lock:
            handle = self._agents.get(agent_id)
        return handle is not None and handle.is_alive()

    def running_agents(self) -> list[AgentHandle]:
        with self._lock:
            return [h for h in self._agents.values() if h.is_alive()]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cleanup_finished(self) -> None:
        """Remove dead threads (call inside lock)."""
        dead = [aid for aid, h in self._agents.items() if not h.is_alive()]
        for aid in dead:
            del self._agents[aid]

    def _run(
        self,
        agent_id:         str,
        task_id:          str,
        prompt:           str,
        context:          str,
        expected_outcome: str,
        interrupt:        threading.Event,
        launch_cwd:       str = "",
    ) -> None:
        """Thread body — runs the agent loop and updates task registry.

        For genuinely multi-part prompts (3+ action verbs or >30 words),
        task_list.decompose() splits them into sequential sub-tasks so each
        step gets a fresh context window and prior results feed forward.
        Simple prompts go directly to run_agent_loop.
        """
        from birdclaw.agent.approvals import approval_queue
        from birdclaw.agent.loop import run_agent_loop
        from birdclaw.agent.task_list import decompose, is_complex
        from birdclaw.memory.session_log import SessionLog
        from birdclaw.memory.tasks import task_registry
        from birdclaw.tools.context_vars import set_task_context, clear_task_context

        self._serial.acquire()
        set_task_context(task_id, agent_id)
        task_registry.start(task_id, agent_id=agent_id)

        extra_system = None
        if context or expected_outcome:
            parts = []
            if context:
                parts.append(f"Context: {context}")
            if expected_outcome:
                parts.append(f"Expected outcome: {expected_outcome}")
            extra_system = "\n".join(parts)

        # Only decompose prompts that are clearly multi-part (3+ action verbs
        # OR very long). Single-step tasks and typical chat go straight through.
        words = prompt.split()
        _action_verbs = {
            "write", "create", "make", "build", "generate", "run", "execute",
            "install", "edit", "modify", "fix", "refactor", "delete", "test",
            "check", "verify", "send", "fetch", "download",
        }
        action_count = len(set(w.lower() for w in words) & _action_verbs)
        use_decompose = action_count >= 3 or len(words) > 30

        logger.info("agent %s: starting task %s (decompose=%s)", agent_id, task_id, use_decompose)
        try:
            if use_decompose:
                task_list = decompose(prompt)
                task_list.save()
                logger.info("agent %s: decomposed into %d steps", agent_id, len(task_list.steps))
                accumulated_context = extra_system or ""
                final_answer = ""
                total_steps = 0

                for step in task_list.pending():
                    if interrupt.is_set():
                        task_list.mark_failed(step.id, "interrupted")
                        break

                    task_list.mark_running(step.id)
                    step_system = accumulated_context or None
                    session_log = SessionLog.new(session_id=f"{task_id}_{step.id}")

                    try:
                        result = run_agent_loop(
                            step.instruction,
                            extra_system=step_system,
                            session_log=session_log,
                            interrupt_event=interrupt,
                            write_dir=launch_cwd,
                        )
                        task_list.mark_done(step.id, result=result.answer[:400])
                        # Feed this step's answer into the next step's context
                        accumulated_context = (
                            (accumulated_context + "\n\n" if accumulated_context else "")
                            + f"Completed: {step.description}\nResult: {result.answer[:400]}"
                        )
                        final_answer = result.answer
                        total_steps += result.steps
                    except Exception as step_err:
                        task_list.mark_failed(step.id, str(step_err))
                        logger.error("agent %s: step %s failed: %s", agent_id, step.id, step_err)

                task_list.save()
                # Always build a combined answer from all completed steps so the
                # task output contains the full work, not just the last step's reply.
                completed = [s for s in task_list.steps if s.status == "done"]
                if len(completed) > 1:
                    final_answer = "\n\n".join(
                        f"**{s.description}**\n{s.result}" for s in completed
                    )
                elif not final_answer:
                    final_answer = "No steps completed."

                task_registry.complete(task_id, output=final_answer)
                logger.info("agent %s: multi-step task %s done (%d steps total)", agent_id, task_id, total_steps)
                _release_waiting_tasks(task_id)
                if total_steps > 2:
                    _spawn_post_task(task_id, prompt, final_answer, total_steps)

            else:
                session_log = SessionLog.new(session_id=task_id)
                result = run_agent_loop(
                    prompt,
                    extra_system=extra_system,
                    session_log=session_log,
                    interrupt_event=interrupt,
                    write_dir=launch_cwd,
                )
                task_registry.complete(task_id, output=result.answer)
                logger.info("agent %s: task %s completed (%d steps)", agent_id, task_id, result.steps)
                _release_waiting_tasks(task_id)
                if result.steps > 2:
                    _spawn_post_task(task_id, prompt, result.answer, result.steps)

        except Exception as e:
            logger.exception("agent %s: task %s failed", agent_id, task_id)
            task_registry.fail(task_id, reason=str(e))
        finally:
            approval_queue.deny_all_for_task(task_id)
            clear_task_context()
            self._serial.release()
            try:
                from birdclaw.memory.memorise import notify_session, memorise_resume
                memorise_resume()
                notify_session(task_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Post-task: reflection + skill crystallisation
# ---------------------------------------------------------------------------

def _release_waiting_tasks(completed_task_id: str) -> None:
    """Spawn any tasks that were waiting for completed_task_id to finish."""
    from birdclaw.memory.tasks import task_registry
    try:
        waiting = task_registry.list(status="waiting")
        for t in waiting:
            if t.after_task_id != completed_task_id:
                continue
            logger.info(
                "releasing waiting task %s (dependency %s completed)",
                t.task_id[:8], completed_task_id[:8],
            )
            agent_id = orchestrator.spawn(t.task_id, t.prompt, t.context, t.expected_outcome)
            task_registry.set_agent(t.task_id, agent_id)
    except Exception:
        logger.debug("_release_waiting_tasks failed", exc_info=True)


def _spawn_post_task(task_id: str, prompt: str, answer: str, steps: int) -> None:
    """Fire-and-forget thread for post-task reflection and skill crystallisation."""
    t = threading.Thread(
        target=_post_task_work,
        args=(task_id, prompt, answer, steps),
        daemon=True,
        name=f"post-task-{task_id[:8]}",
    )
    t.start()


def _post_task_work(task_id: str, prompt: str, answer: str, steps: int) -> None:
    """Run reflection + skill crystallisation after a staged task completes."""
    try:
        _generate_reflection(task_id, prompt, answer)
    except Exception:
        logger.debug("post-task: reflection failed for %s", task_id[:8], exc_info=True)
    try:
        _crystallise_skill(task_id, prompt, answer, steps)
    except Exception:
        logger.debug("post-task: skill crystallisation failed for %s", task_id[:8], exc_info=True)


def _generate_reflection(task_id: str, prompt: str, answer: str) -> None:
    """Ask the model for a 1-2 sentence reflection and log it to inner_life."""
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.scheduler import LLMPriority
    from birdclaw.llm.types import Message
    from birdclaw.tools.context_vars import set_llm_priority
    from birdclaw.memory.inner_life import append_reflection

    set_llm_priority(LLMPriority.BACKGROUND)

    messages = [
        Message(role="system", content=(
            "You are BirdClaw reflecting on work you just completed. "
            "Write 1-2 honest sentences about what was interesting, "
            "what you learned, or what you'd do differently. "
            "Speak in first person. Be specific, not generic."
        )),
        Message(role="user", content=(
            f"Task: {prompt[:200]}\n\n"
            f"What I produced:\n{answer[:400]}\n\n"
            "Reflect briefly."
        )),
    ]

    result = llm_client.generate(messages, thinking=False)
    reflection = (result.content or "").strip()
    if reflection:
        append_reflection(task_id, prompt, reflection)
        logger.debug("post-task: reflection saved for %s", task_id[:8])


def _crystallise_skill(task_id: str, prompt: str, answer: str, steps: int) -> None:
    """Ask the model if this task represents a reusable skill; if so, write it."""
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.scheduler import LLMPriority
    from birdclaw.llm.types import Message
    from birdclaw.tools.context_vars import set_llm_priority
    from birdclaw.config import settings
    import re

    # Only crystallise tasks with enough substance
    if steps < 4 or len(answer) < 100:
        return

    set_llm_priority(LLMPriority.BACKGROUND)

    decision_messages = [
        Message(role="system", content=(
            "Decide whether a completed task represents a reusable, repeatable pattern "
            "worth saving as a skill runbook. "
            "Answer ONLY 'yes: <slug> | <name> | <description> | <tags>' or 'no'. "
            "slug: lowercase-hyphenated, 2-4 words. "
            "tags: comma-separated keywords (5-8). "
            "Say 'yes' only if the task is a clear, reusable workflow "
            "(e.g. 'scrape a website', 'set up a cron job', 'generate a report'). "
            "Say 'no' for one-off or highly specific tasks."
        )),
        Message(role="user", content=f"Task prompt: {prompt[:300]}\nSteps taken: {steps}"),
    ]

    decision_result = llm_client.generate(decision_messages, thinking=False)
    decision = (decision_result.content or "").strip().lower()

    if not decision.startswith("yes:"):
        return

    # Parse: yes: slug | name | description | tags
    parts = [p.strip() for p in decision[4:].split("|")]
    if len(parts) < 3:
        return
    slug        = re.sub(r"[^a-z0-9-]+", "-", parts[0].strip()[:40]).strip("-")
    skill_name  = parts[1].strip()[:60] if len(parts) > 1 else slug
    description = parts[2].strip()[:120] if len(parts) > 2 else ""
    tags_raw    = parts[3].strip() if len(parts) > 3 else ""
    tags        = [t.strip() for t in tags_raw.split(",") if t.strip()]

    if not slug:
        return

    # Generate the SKILL.md using the multi-stage format
    skill_messages = [
        Message(role="system", content=(
            "Write a SKILL.md runbook for a BirdClaw agent skill.\n"
            "Follow PROGRESSIVE DISCLOSURE: each stage does one thing, reveals only what the next stage needs.\n\n"
            "Use EXACTLY this format:\n\n"
            "---\n"
            "name: <slug>\n"
            "description: <one-line description>\n"
            "tags: [tag1, tag2, tag3]\n"
            "stages: <N>\n"
            "---\n\n"
            "## stage:1 Gather\n"
            "Run: df -h && free -h && uptime\n"
            "next_tools: bash\n\n"
            "## stage:2 Write Report\n"
            "Using the output from stage 1, call write_file to write a markdown report to /home/AlgoMind/report.md with sections: Summary, Disk, Memory, Uptime. Use actual values — do not invent numbers.\n"
            "next_tools: write_file\n\n"
            "## stage:3 Confirm and Answer\n"
            "Read /home/AlgoMind/report.md to confirm it was written, then call answer() with the key findings.\n"
            "next_tools: read_file, answer\n\n"
            "PROGRESSIVE DISCLOSURE RULES — you MUST follow all of these:\n"
            "1. Stage 1 is ONLY bash. Include the exact command(s) inline — not a description.\n"
            "   SAFE commands: df -h, free -h, lscpu, uptime, ps aux --sort=-%cpu | head -10,\n"
            "     systemctl list-units --type=service --state=running --no-pager --no-legend | head -30,\n"
            "     cat /proc/loadavg, ip -brief addr, curl -s --max-time 5 <url>\n"
            "   FORBIDDEN: systemctl status --all, top, htop, journalctl without --lines=N,\n"
            "     anything interactive or that paginates without --no-pager.\n"
            "2. Stage 2 uses stage 1 output directly — no new bash, no temp files.\n"
            "   If writing a file: write_file with sections populated from actual stage 1 values.\n"
            "   If just answering: answer with a structured summary.\n"
            "3. Stage 3 only exists when stage 2 writes a file. It reads the file then calls answer.\n"
            "4. NEVER reference scripts, variables, or temp files from a prior run.\n"
            "5. next_tools lists ONLY the tools used in that stage — nothing else.\n"
            "6. 2 stages for web/search tasks, 3 stages for file-producing tasks. Never 4.\n"
            "7. Do NOT add headers, commentary, or anything outside the frontmatter + stage blocks.\n"
        )),
        Message(role="user", content=(
            f"Skill to capture:\n"
            f"name: {skill_name}\n"
            f"description: {description}\n"
            f"tags: {', '.join(tags)}\n\n"
            f"Example task that demonstrated this skill:\n{prompt[:300]}\n\n"
            f"Write the SKILL.md."
        )),
    ]

    skill_result = llm_client.generate(skill_messages, thinking=True)
    skill_content = (skill_result.content or "").strip()

    if not skill_content or "## stage:" not in skill_content:
        logger.debug("post-task: skill generation produced no valid content for %s", slug)
        return

    # Deduplication: if an existing skill shares ≥3 tags, ask the model to merge
    # the new task's learnings into a better version of the existing skill rather
    # than creating a duplicate under a different slug.
    skills_root = settings.data_dir / "skills"
    existing_skill_to_improve: tuple[str, str] | None = None  # (path, content)
    if tags:
        tag_set = set(tags)
        for existing_dir in (skills_root.iterdir() if skills_root.exists() else []):
            existing_file = existing_dir / "SKILL.md"
            if not existing_file.exists():
                continue
            try:
                existing_text = existing_file.read_text(encoding="utf-8")
                for line in existing_text.splitlines():
                    if line.startswith("tags:"):
                        existing_tags = set(
                            t.strip().strip("[]").strip()
                            for t in line[5:].replace("[", "").replace("]", "").split(",")
                            if t.strip()
                        )
                        if len(tag_set & existing_tags) >= 3:
                            existing_skill_to_improve = (str(existing_file), existing_text)
                        break
            except Exception:
                continue
            if existing_skill_to_improve:
                break

    if existing_skill_to_improve:
        import pathlib
        from birdclaw.llm.schemas import EDIT_PATCH_SCHEMA
        existing_path, current_content = existing_skill_to_improve
        logger.debug("post-task: patching existing skill at %s", existing_path)
        patch_system = (
            "You are patching a SKILL.md file. Apply targeted fixes using JSON patches.\n"
            "FORBIDDEN in skills: top, systemctl status --all, temp file references "
            "(raw_metrics.txt, etc.), more than 3 stages, [bracket] syntax in next_tools.\n"
            "REQUIRED: safe bash only, exact commands inline, progressive disclosure.\n"
            "Two choices per turn:\n"
            '  Fix something: {"old": "exact text to replace", "new": "replacement"}\n'
            '  All done:      {"old": "DONE", "new": ""}'
        )
        patch_messages = [
            Message(role="system", content=patch_system),
            Message(role="user", content=(
                f"Skill file:\n```\n{current_content}\n```\n\n"
                f"Recent task that used this skill:\n{prompt[:200]}"
            )),
        ]
        for _patch_step in range(8):
            patch_result = llm_client.generate(
                patch_messages, format_schema=EDIT_PATCH_SCHEMA, thinking=True,
            )
            parsed = json.loads(patch_result.content or "{}")
            old_text = parsed.get("old", "")
            new_text = parsed.get("new", "")
            if old_text == "DONE" or not old_text:
                break
            if old_text in current_content:
                current_content = current_content.replace(old_text, new_text, 1)
                patch_messages.append(Message(role="assistant", content=patch_result.content or ""))
                patch_messages.append(Message(role="user", content=(
                    f"Applied. Updated skill:\n```\n{current_content}\n```\nNext fix or DONE."
                )))
            else:
                break
        pathlib.Path(existing_path).write_text(current_content, encoding="utf-8")
        logger.info("post-task: skill patched → %s  (%d steps)", existing_path, _patch_step + 1)
        return

    skill_dir = skills_root / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    if skill_file.exists():
        logger.debug("post-task: skill %r already exists, skipping", slug)
        return

    skill_file.write_text(skill_content, encoding="utf-8")
    logger.info("post-task: skill crystallised → %s", skill_file)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

orchestrator = Orchestrator()
