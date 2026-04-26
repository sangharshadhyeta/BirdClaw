"""Sandbox — Linux namespace isolation for bash commands.

Python port of claw-code-parity/rust/crates/runtime/src/sandbox.rs.

Three isolation layers (all via `unshare`):
  namespace  — user/mount/IPC/PID/UTS namespaces  (--user --mount --ipc --pid --uts)
  network    — network namespace                   (--net)
  filesystem — WorkspaceOnly or AllowList          (env vars consumed by the shell wrapper)

Usage:
    from birdclaw.tools.sandbox import resolve_sandbox_status, build_sandbox_command
    from birdclaw.config import settings

    status = resolve_sandbox_status(sandbox_config_from_settings(), cwd=Path.cwd())
    sc = build_sandbox_command("ls /tmp", Path.cwd(), status)
    if sc:
        proc = subprocess.run([sc.program] + sc.args, env=dict(sc.env))
    else:
        proc = subprocess.run(["sh", "-lc", "ls /tmp"])

Returned SandboxStatus is attached to every BashCommandOutput so the agent
can see what level of isolation was actually active.

Reference: claw-code-parity/rust/crates/runtime/src/sandbox.rs
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FilesystemIsolationMode(str, Enum):
    off            = "off"
    workspace_only = "workspace-only"
    allow_list     = "allow-list"


@dataclass
class SandboxConfig:
    """User-supplied sandbox preferences (all optional — missing = use default)."""
    enabled:                 Optional[bool]                        = None
    namespace_restrictions:  Optional[bool]                        = None
    network_isolation:       Optional[bool]                        = None
    filesystem_mode:         Optional[FilesystemIsolationMode]     = None
    allowed_mounts:          list[str]                             = field(default_factory=list)

    def resolve_request(
        self,
        *,
        enabled_override:           Optional[bool]                    = None,
        namespace_override:         Optional[bool]                    = None,
        network_override:           Optional[bool]                    = None,
        filesystem_mode_override:   Optional[FilesystemIsolationMode] = None,
        allowed_mounts_override:    Optional[list[str]]               = None,
    ) -> "SandboxRequest":
        return SandboxRequest(
            enabled=_first(enabled_override, self.enabled, True),
            namespace_restrictions=_first(namespace_override, self.namespace_restrictions, True),
            network_isolation=_first(network_override, self.network_isolation, False),
            filesystem_mode=_first(
                filesystem_mode_override,
                self.filesystem_mode,
                FilesystemIsolationMode.workspace_only,
            ),
            allowed_mounts=allowed_mounts_override if allowed_mounts_override is not None
                           else list(self.allowed_mounts),
        )


@dataclass
class SandboxRequest:
    """Fully resolved sandbox request — no Optional fields."""
    enabled:                bool
    namespace_restrictions: bool
    network_isolation:      bool
    filesystem_mode:        FilesystemIsolationMode
    allowed_mounts:         list[str] = field(default_factory=list)


@dataclass
class ContainerEnvironment:
    in_container: bool
    markers:      list[str] = field(default_factory=list)


@dataclass
class SandboxStatus:
    enabled:              bool
    requested:            SandboxRequest
    supported:            bool
    active:               bool
    namespace_supported:  bool
    namespace_active:     bool
    network_supported:    bool
    network_active:       bool
    filesystem_mode:      FilesystemIsolationMode
    filesystem_active:    bool
    allowed_mounts:       list[str]
    in_container:         bool
    container_markers:    list[str]
    fallback_reason:      Optional[str]

    def to_dict(self) -> dict:
        return {
            "enabled":             self.enabled,
            "supported":           self.supported,
            "active":              self.active,
            "namespace_active":    self.namespace_active,
            "network_active":      self.network_active,
            "filesystem_mode":     self.filesystem_mode.value,
            "filesystem_active":   self.filesystem_active,
            "in_container":        self.in_container,
            "fallback_reason":     self.fallback_reason,
        }


@dataclass
class LinuxSandboxCommand:
    """The `unshare` invocation that wraps a shell command."""
    program: str                        # "unshare"
    args:    list[str]                  # flags + ["sh", "-lc", command]
    env:     list[tuple[str, str]]      # overrides to merge into subprocess env


# ---------------------------------------------------------------------------
# Container detection (mirrors detect_container_environment_from)
# ---------------------------------------------------------------------------

def detect_container_environment() -> ContainerEnvironment:
    proc_1_cgroup: Optional[str] = None
    try:
        proc_1_cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return _detect_from(
        env_pairs=list(os.environ.items()),
        dockerenv_exists=Path("/.dockerenv").exists(),
        containerenv_exists=Path("/run/.containerenv").exists(),
        proc_1_cgroup=proc_1_cgroup,
    )


def _detect_from(
    env_pairs: list[tuple[str, str]],
    dockerenv_exists: bool,
    containerenv_exists: bool,
    proc_1_cgroup: Optional[str],
) -> ContainerEnvironment:
    markers: list[str] = []
    if dockerenv_exists:
        markers.append("/.dockerenv")
    if containerenv_exists:
        markers.append("/run/.containerenv")
    for key, value in env_pairs:
        if key.lower() in ("container", "docker", "podman", "kubernetes_service_host") and value:
            markers.append(f"env:{key}={value}")
    if proc_1_cgroup:
        for needle in ("docker", "containerd", "kubepods", "podman", "libpod"):
            if needle in proc_1_cgroup:
                markers.append(f"/proc/1/cgroup:{needle}")
    markers = sorted(set(markers))
    return ContainerEnvironment(in_container=bool(markers), markers=markers)


# ---------------------------------------------------------------------------
# Sandbox capability detection
# ---------------------------------------------------------------------------

def _unshare_works() -> bool:
    """Check once whether `unshare --user --map-root-user` actually succeeds.

    Some CI environments have the binary but kernel user-namespaces disabled.
    Cached after first call (mirrors Rust OnceLock).
    """
    global _unshare_works_cached
    if _unshare_works_cached is None:
        if shutil.which("unshare") is None:
            _unshare_works_cached = False
        else:
            try:
                result = subprocess.run(
                    ["unshare", "--user", "--map-root-user", "true"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                _unshare_works_cached = result.returncode == 0
            except Exception:
                _unshare_works_cached = False
    return _unshare_works_cached


_unshare_works_cached: Optional[bool] = None


# ---------------------------------------------------------------------------
# Sandbox status resolution
# ---------------------------------------------------------------------------

def sandbox_config_from_settings() -> SandboxConfig:
    """Build a SandboxConfig from flat settings fields (avoids circular import)."""
    from birdclaw.config import settings
    return SandboxConfig(
        enabled=settings.sandbox_enabled,
        network_isolation=settings.sandbox_network_isolation,
    )


def resolve_sandbox_status(config: SandboxConfig, cwd: Path) -> SandboxStatus:
    request = config.resolve_request()
    return _resolve_for_request(request, cwd)


def _resolve_for_request(request: SandboxRequest, cwd: Path) -> SandboxStatus:
    container = detect_container_environment()
    on_linux  = sys.platform == "linux"
    ns_supported = on_linux and _unshare_works()
    net_supported = ns_supported

    fallback_reasons: list[str] = []
    if request.enabled and request.namespace_restrictions and not ns_supported:
        fallback_reasons.append(
            "namespace isolation unavailable (requires Linux with `unshare`)"
        )
    if request.enabled and request.network_isolation and not net_supported:
        fallback_reasons.append(
            "network isolation unavailable (requires Linux with `unshare`)"
        )
    if (request.enabled
            and request.filesystem_mode == FilesystemIsolationMode.allow_list
            and not request.allowed_mounts):
        fallback_reasons.append(
            "filesystem allow-list requested without configured mounts"
        )

    active = (
        request.enabled
        and (not request.namespace_restrictions or ns_supported)
        and (not request.network_isolation or net_supported)
    )
    filesystem_active = request.enabled and request.filesystem_mode != FilesystemIsolationMode.off
    allowed_mounts = _normalize_mounts(request.allowed_mounts, cwd)

    return SandboxStatus(
        enabled=request.enabled,
        requested=request,
        supported=ns_supported,
        active=active,
        namespace_supported=ns_supported,
        namespace_active=request.enabled and request.namespace_restrictions and ns_supported,
        network_supported=net_supported,
        network_active=request.enabled and request.network_isolation and net_supported,
        filesystem_mode=request.filesystem_mode,
        filesystem_active=filesystem_active,
        allowed_mounts=allowed_mounts,
        in_container=container.in_container,
        container_markers=container.markers,
        fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
    )


# ---------------------------------------------------------------------------
# Sandbox command builder
# ---------------------------------------------------------------------------

def build_sandbox_command(
    command: str,
    cwd: Path,
    status: SandboxStatus,
) -> Optional[LinuxSandboxCommand]:
    """Build the `unshare` invocation for a command, or None if sandbox is inactive.

    Returns None when:
    - not on Linux
    - sandbox disabled
    - neither namespace nor network isolation is active
    """
    if sys.platform != "linux":
        return None
    if not status.enabled:
        return None
    if not status.namespace_active and not status.network_active:
        return None

    args = [
        "--user",
        "--map-root-user",
        "--mount",
        "--ipc",
        "--pid",
        "--uts",
        "--fork",
    ]
    if status.network_active:
        args.append("--net")
    args += ["sh", "-lc", command]

    sandbox_home = cwd / ".sandbox-home"
    sandbox_tmp  = cwd / ".sandbox-tmp"
    env: list[tuple[str, str]] = [
        ("HOME",   str(sandbox_home)),
        ("TMPDIR", str(sandbox_tmp)),
        ("BC_SANDBOX_FILESYSTEM_MODE",   status.filesystem_mode.value),
        ("BC_SANDBOX_ALLOWED_MOUNTS",    ":".join(status.allowed_mounts)),
    ]
    if path := os.environ.get("PATH"):
        env.append(("PATH", path))

    return LinuxSandboxCommand(program="unshare", args=args, env=env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_mounts(mounts: list[str], cwd: Path) -> list[str]:
    result = []
    for m in mounts:
        p = Path(m)
        result.append(str(p if p.is_absolute() else cwd / p))
    return result


def _first(*values):
    """Return the first non-None value."""
    for v in values:
        if v is not None:
            return v
    return None
