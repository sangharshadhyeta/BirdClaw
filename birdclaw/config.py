"""BirdClaw configuration.

All settings can be overridden via environment variables with the BC_ prefix
or via ~/.birdclaw/config.toml (not yet implemented — env only for now).

LLM backend is any OpenAI-compatible API. llama.cpp server is the default;
the user can point it at any provider:

    # llama.cpp server (default)
    BC_LLM_BASE_URL=http://localhost:8080/v1
    BC_LLM_MODEL=gemma-4-4b

    # OpenAI
    BC_LLM_BASE_URL=https://api.openai.com/v1
    BC_LLM_MODEL=gpt-4o
    BC_LLM_API_KEY=sk-...

    # Groq, Together, LM Studio, vLLM, etc. — same pattern
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _detect_container() -> str:
    """Return the container runtime name, or 'host' if running on bare metal.

    Checks (in order): Docker cgroup marker, .dockerenv sentinel, systemd-nspawn,
    Kubernetes downward-API env, generic CONTAINER env var.
    """
    import os

    if os.path.exists("/.dockerenv"):
        return "docker"
    try:
        cgroup = Path("/proc/1/cgroup").read_text(errors="replace")
        if "docker" in cgroup:
            return "docker"
        if "lxc" in cgroup:
            return "lxc"
        if "kubepods" in cgroup:
            return "kubernetes"
    except OSError:
        pass
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if os.environ.get("container") == "systemd-nspawn":
        return "systemd-nspawn"
    container_env = os.environ.get("CONTAINER", "").lower()
    if container_env:
        return container_env
    return "host"


def _default_workspace_roots() -> list[Path]:
    """Find workspace roots dynamically.

    Result is always: explicit_roots + cwd (if cwd looks like a real project dir).

    explicit_roots come from:
      1. BC_WORKSPACE shell env var (comma-separated paths)
      2. BC_WORKSPACE from ~/.birdclaw/.env (same format)

    cwd is ALWAYS added on top of explicit roots when it is not shallow, not root,
    and not already covered by an explicit root — so that `birdclaw tui` launched
    from /home/user/myproject automatically operates in /home/user/myproject
    even if BC_WORKSPACE points elsewhere.

    NOTE: the package install directory is intentionally NOT used as a fallback.
    When running an installed copy (e.g. from /opt), config.py.__file__ points back
    to the source tree (editable install), which would leak the dev project's files
    (CLAUDE.md, birdclaw source) into the user's workspace snapshot.
    """
    import os

    # Priority 1: shell env var
    env_roots_str = os.environ.get("BC_WORKSPACE", "").strip()

    # Priority 2: ~/.birdclaw/.env (pydantic-settings reads it internally but
    # doesn't expose it to os.environ; parse it directly for the BC_WORKSPACE key)
    if not env_roots_str:
        user_env = Path.home() / ".birdclaw" / ".env"
        if user_env.exists():
            for line in user_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("BC_WORKSPACE=") and "ROOTS" not in line:
                    env_roots_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    explicit: list[Path] = []
    if env_roots_str:
        explicit = [Path(p).resolve() for p in env_roots_str.split(",") if p.strip()]

    # Always add cwd if it's a real project directory and not already covered
    cwd = Path.cwd().resolve()
    cwd_is_shallow = len(cwd.parts) <= 2
    cwd_is_root    = cwd == Path("/")
    cwd_already_covered = any(
        cwd == r or r in cwd.parents or cwd in r.parents
        for r in explicit
    )

    roots = list(explicit)  # start with explicit roots (preserves order/priority)

    if not cwd_is_shallow and not cwd_is_root and not cwd_already_covered:
        roots.insert(0, cwd)  # launch cwd is first — highest priority for default writes

    return roots if roots else [cwd]


def _env_files() -> tuple[str, ...]:
    """Return env file search path in priority order (last wins).

    ~/.birdclaw/.env  — written by install.sh with the user's chosen ports/paths
    .env              — local project override (dev use)
    """
    user_env = Path.home() / ".birdclaw" / ".env"
    return (str(user_env), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BC_",
        env_file=_env_files(),
        extra="ignore",
    )

    # ── LLM backend (any OpenAI-compatible API) ───────────────────────────────
    llm_base_url: str = Field(
        "http://localhost:8080/v1",
        description="OpenAI-compatible chat completions endpoint (llama.cpp default)",
    )
    llm_model: str = Field(
        "gemma-4-4b",
        description="Primary model — must match the model loaded in llama.cpp server",
    )
    llm_action_model: str = Field(
        "",
        description=(
            "Optional action model override for write_file/bash/edit_file steps. "
            "Leave empty to use llm_model for all steps."
        ),
    )
    llm_api_key: str = Field(
        "",
        description="API key — leave empty for local backends (llama.cpp, LM Studio)",
    )

    # ── Hands model — small specialist for all format-mode calls ─────────────
    # Leave empty to run single-model (hands falls back to llm_model).
    # Intended for a 270M function-call fine-tune on a second llama.cpp port.
    llm_hands_base_url: str = Field(
        "",
        description="Endpoint for the hands (format-mode) model. Empty = use llm_base_url.",
    )
    llm_hands_model: str = Field(
        "",
        description="Model name for hands calls (plan, gate, edit_file, soul routing). Empty = use llm_model.",
    )

    # ── llama.cpp server ──────────────────────────────────────────────────────
    llamacpp_parallel: int = Field(
        4,
        description="Parallel inference slots — set via --parallel when starting the server",
    )
    llamacpp_gpu_layers: int = Field(
        -1,
        description="GPU layers to offload — -1 = all (full GPU), 0 = CPU only",
    )
    temperature: float = Field(0.0, description="Deterministic — small models need this for tool calling")
    max_tokens: int = Field(4096, description="Output token budget")
    n_ctx: int = Field(32768, description="Context window per slot; total server ctx = n_ctx × llamacpp_parallel")

    llm_scheduler_enabled: bool = Field(
        True,
        description="Route LLM calls through the priority scheduler (set false to bypass in tests)",
    )
    parallel_tasks: bool = Field(
        False,
        description="Allow multiple agent tasks to run concurrently (BC_PARALLEL_TASKS=true to enable)",
    )

    # ── Agent loop ────────────────────────────────────────────────────────────
    max_tools_per_turn: int = Field(2, description="Max domain tools shown per turn (plus answer = 3 total)")
    max_agent_steps: int = Field(500, description="Absolute runaway safety cap — per-stage budgets are the primary governor (BC_MAX_AGENT_STEPS to override)")

    # Per-stage-type default step budgets (used when planner doesn't specify and
    # no historical data exists yet). Historical P75 from stage_history.jsonl
    # replaces these automatically after a few runs.
    stage_budgets: dict = Field(
        default_factory=lambda: {
            "research":   12,
            "write_doc":  10,
            "write_code": 12,
            "edit_file":  8,
            "verify":     8,
            "reflect":    5,
        },
        description="Default step budget per stage type — overridden by planner or historical P75",
    )
    stage_budget_max_grant: int = Field(
        100,
        description="Maximum additional steps a single request_budget call can grant",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        Path.home() / ".birdclaw",
        description="Root directory for all persistent data",
    )

    # ── Workspace ─────────────────────────────────────────────────────────────
    workspace_dir: Path = Field(
        Path.home(),
        description="Default directory for task output files when no launch_cwd is provided",
    )
    workspace_roots: list[Path] = Field(
        default_factory=_default_workspace_roots,
        description="Directories the agent may read/write and scan for workspace state",
    )
    permission_mode: str = Field(
        "workspace_write",
        description="Permission mode: read_only | workspace_write | danger_full_access | prompt | allow",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    searxng_url: str = Field("http://localhost:8888", description="SearXNG instance URL")
    max_fetch_chars: int = Field(6000, description="Max chars kept from a fetched URL")

    # ── Gateway ───────────────────────────────────────────────────────────────
    gateway_host: str = Field("127.0.0.1")
    gateway_port: int = Field(7823)

    # ── TUI ───────────────────────────────────────────────────────────────────
    theme: str = Field(
        "dark",
        description="TUI color theme: dark | light | solarized | catppuccin",
    )

    # ── Sandbox ───────────────────────────────────────────────────────────────
    sandbox_enabled: bool = Field(
        False,
        description="Enable Linux namespace sandbox for bash commands (BC_SANDBOX_ENABLED=1)",
    )
    sandbox_network_isolation: bool = Field(
        False,
        description="Isolate network in sandbox (BC_SANDBOX_NETWORK_ISOLATION=1)",
    )

    # ── Self-modification ─────────────────────────────────────────────────────
    self_modify: bool = Field(
        False,
        description="Allow agent to modify its own source code (BC_SELF_MODIFY=1)",
    )

    # ── Derived paths (not configurable directly) ─────────────────────────────
    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def plans_dir(self) -> Path:
        return self.data_dir / "plans"

    @property
    def src_dir(self) -> Path:
        """The birdclaw package source directory — always readable, writable only when
        self_modify=True (self-update cycle)."""
        return Path(__file__).resolve().parent

    @property
    def self_update_todo_path(self) -> Path:
        """Backlog of improvements noted by the agent during normal tasks."""
        return self.data_dir / "self_update_todo.jsonl"

    @property
    def container_runtime(self) -> str:
        """Detected container runtime: 'docker' | 'kubernetes' | 'lxc' | 'host' | ..."""
        return _detect_container()

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.sessions_dir, self.skills_dir, self.plans_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("[config] runtime=%s  workspace=%s", self.container_runtime, self.workspace_roots)


settings = Settings()
