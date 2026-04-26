"""BirdClaw entry point.

Usage:
    python main.py cli              Interactive REPL (rich terminal)
    python main.py tui              Three-pane multi-agent TUI
    python main.py prompt "task"    One-shot prompt
    python main.py daemon           Start gateway daemon (persistent, no TUI)
    python main.py dream            Run memory consolidation (dreaming agent)
    python main.py memorise         Drain session logs into knowledge graph
    python main.py graph-server     Expose knowledge graph as MCP stdio server
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()
_log = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    File handler: always DEBUG — captures every event for pipeline debugging.
    Stderr handler: ERROR only in CLI, suppressed in TUI.

    Log format includes milliseconds so timing between steps is visible.
    Log file: ~/.birdclaw/birdclaw.log  (tail -f it to watch the pipeline live)
    """
    log_dir = Path.home() / ".birdclaw"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "birdclaw.log"

    fmt = "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-36s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # always capture everything

    handlers: list[logging.Handler] = [file_handler]

    import sys as _sys
    if not any(a == "tui" for a in _sys.argv):
        _stderr = logging.StreamHandler()
        _stderr.setLevel(logging.ERROR)    # console stays quiet; tail the log file
        handlers.append(_stderr)

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party debug loggers that flood the log file
    for _noisy in ("markdown_it", "httpcore", "PIL", "asyncio"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ASCII splash animation for CLI
# ---------------------------------------------------------------------------

_SPLASH_SPLIT = 40   # lines 0-39 = bird, lines 40+ = BIRDCLAW text

# ANSI colour sequences
_C_BIRD_DIM    = "\033[2;36m"   # dim cyan — upper bird
_C_BIRD_MID    = "\033[36m"     # cyan — mid bird
_C_BIRD_BRIGHT = "\033[1;36m"   # bold cyan — lower bird
_C_TEXT        = "\033[1;96m"   # bold bright cyan — BIRDCLAW letters
_C_RESET       = "\033[0m"


def _animate_splash() -> None:
    if not sys.stdout.isatty():
        return
    import shutil, textwrap
    _ascii = Path(__file__).parent / "assets" / "ascii_art"
    try:
        raw = _ascii.read_text(encoding="utf-8")
    except OSError:
        return

    lines = textwrap.dedent(raw).splitlines()
    if not lines:
        return

    # Clamp art height to terminal so every line stays in the scrollback-free
    # visible area — this lets the cursor-up erase phase reach all lines.
    term_rows = shutil.get_terminal_size((80, 24)).lines
    max_rows  = max(5, term_rows - 4)   # leave room for the panel below
    lines     = lines[:max_rows]

    bird   = lines[:_SPLASH_SPLIT]
    text   = lines[_SPLASH_SPLIT:]
    total  = len(lines)
    n_bird = len(bird)

    # ── Phase 1: draw bird top→bottom with colour gradient ──────────────────
    for i, ln in enumerate(bird):
        colour = _C_BIRD_DIM if i < n_bird // 3 else (
                 _C_BIRD_MID if i < 2 * n_bird // 3 else _C_BIRD_BRIGHT)
        sys.stdout.write(f"{colour}{ln}{_C_RESET}\n")
        sys.stdout.flush()
        time.sleep(0.022)

    # ── Phase 2: draw BIRDCLAW text top→bottom in bright cyan ───────────────
    for ln in text:
        sys.stdout.write(f"{_C_TEXT}{ln}{_C_RESET}\n")
        sys.stdout.flush()
        time.sleep(0.028)

    time.sleep(0.55)

    # ── Phase 3: erase — clear lines top→bottom so art appears to rise ──────
    # All `total` lines are within the visible area (clamped above), so
    # cursor-up can always reach the first line.
    sys.stdout.write(f"\033[{total}A")   # jump to first art line
    sys.stdout.flush()
    for _ in range(total):
        sys.stdout.write("\033[2K\n")    # clear line, advance
        sys.stdout.flush()
        time.sleep(0.013)
    sys.stdout.write(f"\033[{total}A")   # park cursor at art start
    sys.stdout.flush()


# CLI mode — interactive REPL
# ---------------------------------------------------------------------------

def cmd_cli(args: argparse.Namespace) -> None:
    from birdclaw.agent.soul_loop import soul_respond
    from birdclaw.config import settings
    from birdclaw.memory.history import History

    settings.ensure_dirs()

    history = History.new()

    _animate_splash()
    try:
        from birdclaw.llm.model_profile import combined_display_name as _cdn
        _model_display = _cdn()
    except Exception:
        _model_display = settings.llm_model
    console.print(Panel(
        "🐦 [bold cyan]BirdClaw[/] [dim]— local autonomous agent[/]\n"
        f"[dim]model: {_model_display}  |  type [bold]exit[/bold] to quit[/]",
        border_style="cyan",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold green]>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        _log.info("[cli] user input  chars=%d  preview=%r", len(question), question[:60])
        with console.status("[dim]thinking…[/]", spinner="dots"):
            try:
                response = soul_respond(question, history=history)
            except KeyboardInterrupt:
                _log.info("[cli] cancelled by user")
                console.print("\n[dim]Cancelled.[/]")
                continue
            except Exception as e:
                _log.error("[cli] soul_respond error: %s", e)
                console.print(f"[red]Error:[/] {e}")
                continue

        console.print()
        if response.reply:
            _log.info("[cli] reply  chars=%d  preview=%r", len(response.reply), response.reply[:80])
            console.print(Markdown(response.reply))

        if response.task_id:
            # Wait for the background task then print its output
            from birdclaw.memory.tasks import task_registry
            import time as _time

            task_id = response.task_id
            _log.info("[cli] task spawned  task_id=%s", task_id)
            try:
                with console.status("[dim]working…[/]", spinner="dots"):
                    while True:
                        task = task_registry.get(task_id)
                        if task is None or task.status in ("completed", "failed", "stopped"):
                            break
                        _time.sleep(0.3)
            except KeyboardInterrupt:
                _log.info("[cli] wait cancelled  task_id=%s", task_id)
                console.print("\n[dim]Cancelled — task still running in background.[/]")

            task = task_registry.get(task_id)
            task_output = (task.output or "").strip() if task else ""
            if task_output:
                _log.info("[cli] task output  task_id=%s  chars=%d", task_id, len(task_output))
                console.print()
                console.print(Markdown(task_output))
            elif task and task.status == "failed":
                _log.warning("[cli] task failed  task_id=%s", task_id)
                console.print(f"[red]Task failed.[/]")

            history.add_turn("user", question)
            if task_output:
                history.add_turn("assistant", task_output)
        else:
            history.add_turn("user", question)
            if response.reply:
                history.add_turn("assistant", response.reply)


# ---------------------------------------------------------------------------
# Prompt mode — one-shot
# ---------------------------------------------------------------------------

def cmd_prompt(args: argparse.Namespace) -> None:
    from birdclaw.agent.loop import run_agent_loop
    from birdclaw.config import settings

    settings.ensure_dirs()

    question = " ".join(args.text)
    with console.status("[dim]working…[/]", spinner="dots"):
        result = run_agent_loop(question)

    console.print(Markdown(result.answer))
    if result.sources:
        console.print(Text("Sources: " + ", ".join(result.sources), style="dim"))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def cmd_tui(_args: argparse.Namespace) -> None:
    import os as _os
    # Ensure UTF-8 locale before Textual initialises its terminal input thread.
    # Without this, non-ASCII terminal byte sequences (mouse codes, special keys)
    # can trigger UnicodeDecodeError in the linux_driver input loop.
    for _var, _val in (
        ("PYTHONIOENCODING", "utf-8"),
        ("LANG",             "en_US.UTF-8"),
        ("LC_ALL",           "en_US.UTF-8"),
    ):
        _os.environ.setdefault(_var, _val)

    from birdclaw.config import settings
    from birdclaw.tui.app import BirdClawApp
    settings.ensure_dirs()
    BirdClawApp().run()


def cmd_memorise(args: argparse.Namespace) -> None:
    from birdclaw.config import settings
    from birdclaw.memory.memorise import run_memorise

    settings.ensure_dirs()
    session_id = getattr(args, "session_id", None) or None
    with console.status("[dim]memorising…[/]", spinner="dots"):
        count = run_memorise(session_id=session_id)
    console.print(f"[green]Memorised[/] {count} content unit(s).")


def cmd_graph_server(_args: argparse.Namespace) -> None:
    import logging
    from birdclaw.config import settings
    from birdclaw.tools.mcp.graph_server import serve

    settings.ensure_dirs()
    # Keep stdout clean for the JSON-RPC transport
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    serve()


def cmd_daemon(_args: argparse.Namespace) -> None:
    """Start the gateway daemon — persistent process, no TUI."""
    import signal
    from birdclaw.config import settings
    from birdclaw.gateway.gateway import gateway
    from birdclaw.tools.mcp.manager import mcp_manager, mcp_bridge

    settings.ensure_dirs()
    mcp_manager.load_from_config()
    mcp_bridge.register_all()

    gateway.start()
    from birdclaw.skills.cron import cron_service
    cron_service.start()
    from birdclaw.memory.dream import run_dream_cycle
    cron_service.register_system_job(
        "dream",
        lambda: run_dream_cycle(quiet=True),
        "0 3 * * *",
        "Nightly dream cycle — memorise + graph reflection + inner life + cleanup",
    )
    from birdclaw.gateway.tui_socket import start as start_tui_socket
    start_tui_socket()
    from birdclaw.gateway.test_socket import start as start_test_socket
    start_test_socket()
    console.print(
        "[bold cyan]BirdClaw daemon started.[/] "
        f"[dim]model: {settings.llm_model}[/]\n"
        "[dim]Ctrl+C to stop.[/]"
    )

    stop_event = threading.Event()

    def _handle_signal(sig, frame):  # noqa: ANN001
        console.print("\n[dim]Shutting down…[/]")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    stop_event.wait()  # block until signal


def cmd_dream(_args: argparse.Namespace) -> None:
    """Run a full dreaming cycle: memorise + graph + inner life + user knowledge + cleanup."""
    from birdclaw.memory.dream import run_dream_cycle
    from birdclaw.gateway.notify import push_notification
    results = run_dream_cycle(quiet=False)
    errors = [k for k, v in results.items() if "error" in v.lower() or "fail" in v.lower()]
    ok = [k for k, v in results.items() if k not in errors]
    if errors:
        msg = "Dreaming done with errors — " + ", ".join(errors) + " failed"
        push_notification(msg, title="Dream", severity="warning")
    else:
        msg = ("Dreaming complete — " + ", ".join(ok)) if ok else "Dreaming complete."
        push_notification(msg, title="Dream", severity="information")


def cmd_self_update(_args: argparse.Namespace) -> None:
    """Run one self-update cycle: hypothesis → patch → test → accept/revert."""
    from birdclaw.agent.self_update import run_self_update_cycle
    from birdclaw.gateway.notify import push_notification
    result = run_self_update_cycle()
    if result.get("success"):
        push_notification("Self-update accepted — patch applied.", title="Self-update", severity="information")
        console.print(f"[green]Self-update accepted.[/] {result.get('summary', '')}")
    else:
        push_notification("Self-update reverted — no improvement.", title="Self-update", severity="warning")
        console.print(f"[yellow]Self-update reverted.[/] {result.get('summary', '')}")


def cmd_cleanup(_args: argparse.Namespace) -> None:
    """Prune stale sessions, tasks, pages, and self-update backups."""
    from birdclaw.memory.cleanup import run_cleanup
    from birdclaw.gateway.notify import push_notification
    results = run_cleanup()
    total = sum(results.values())
    if total:
        console.print(f"[green]Cleanup complete:[/] {results}")
        push_notification(f"Cleanup complete — {total} item(s) removed.", title="Cleanup")
    else:
        console.print("[dim]Cleanup: nothing to remove.[/]")
        push_notification("Cleanup: nothing to remove.", title="Cleanup")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdclaw",
        description="BirdClaw — long-term autonomous AI agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug output")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cli", help="Interactive REPL (rich terminal)")
    sub.add_parser("tui", help="Three-pane multi-agent TUI")

    p_prompt = sub.add_parser("prompt", help="One-shot prompt")
    p_prompt.add_argument("text", nargs="+", help="The prompt text")

    p_memorise = sub.add_parser("memorise", help="Drain session logs into knowledge graph")
    p_memorise.add_argument("session_id", nargs="?", help="Process only this session (default: all)")

    sub.add_parser("graph-server", help="Expose knowledge graph as MCP stdio server")

    sub.add_parser("daemon", help="Start gateway daemon")
    sub.add_parser("dream", help="Run memory consolidation")
    sub.add_parser("cleanup", help="Prune stale sessions, tasks, pages, and backups")
    sub.add_parser("self-update", help="Run one self-update cycle (hypothesis → patch → test → accept/revert)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "cli": cmd_cli,
        "tui": cmd_tui,
        "prompt": cmd_prompt,
        "memorise": cmd_memorise,
        "graph-server": cmd_graph_server,
        "daemon": cmd_daemon,
        "dream": cmd_dream,
        "cleanup": cmd_cleanup,
        "self-update": cmd_self_update,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
