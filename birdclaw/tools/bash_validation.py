"""Bash command validation pipeline.

Full port of claw-code-parity/rust/crates/runtime/src/bash_validation.rs.

Five validation stages run in sequence:
    1. validate_mode        — mode-level check (includes read-only)
    2. validate_sed         — sed -i in read-only
    3. check_destructive    — rm -rf /, fork bombs, shred, etc.
    4. validate_paths       — ../traversal, ~/escape warnings
    (destructive is checked after sed so sed -i in read-only gets the
     clearer Block rather than a Warn)

Public entry point:
    validate_command(command, mode, workspace) → ValidationResult

Also exposes:
    classify_command(command) → CommandIntent
    (used by tool_cache to decide cacheability without running full pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


# ---------------------------------------------------------------------------
# Result + intent types
# ---------------------------------------------------------------------------

class ValidationResult:
    """Discriminated union: Allow | Block | Warn."""

    @staticmethod
    def allow() -> "Allow":
        return Allow()

    @staticmethod
    def block(reason: str) -> "Block":
        return Block(reason)

    @staticmethod
    def warn(message: str) -> "Warn":
        return Warn(message)


@dataclass(frozen=True)
class Allow(ValidationResult):
    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Allow)


@dataclass(frozen=True)
class Block(ValidationResult):
    reason: str

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class Warn(ValidationResult):
    message: str

    def __bool__(self) -> bool:
        return True  # warns still allow execution — caller decides


class CommandIntent(Enum):
    ReadOnly          = auto()
    Write             = auto()
    Destructive       = auto()
    Network           = auto()
    ProcessManagement = auto()
    PackageManagement = auto()
    SystemAdmin       = auto()
    Unknown           = auto()


# ---------------------------------------------------------------------------
# Command tables
# ---------------------------------------------------------------------------

_WRITE_COMMANDS = frozenset({
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "chmod", "chown", "chgrp",
    "ln", "install", "tee", "truncate", "shred", "mkfifo", "mknod", "dd",
})

_STATE_MODIFYING_COMMANDS = frozenset({
    "apt", "apt-get", "yum", "dnf", "pacman", "brew",
    "pip", "pip3", "npm", "yarn", "pnpm", "bun", "cargo", "gem", "go", "rustup",
    "docker", "systemctl", "service", "mount", "umount",
    "kill", "pkill", "killall",
    "reboot", "shutdown", "halt", "poweroff",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "crontab", "at",
})

_WRITE_REDIRECTIONS = (">", ">>", ">&")

_GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "tag", "stash", "remote",
    "fetch", "ls-files", "ls-tree", "cat-file", "rev-parse", "describe",
    "shortlog", "blame", "bisect", "reflog", "config",
})

_DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rm -rf /",   "Recursive forced deletion at root — this will destroy the system"),
    ("rm -rf ~",   "Recursive forced deletion of home directory"),
    ("rm -rf *",   "Recursive forced deletion of all files in current directory"),
    ("rm -rf .",   "Recursive forced deletion of current directory"),
    ("mkfs",       "Filesystem creation will destroy existing data on the device"),
    ("dd if=",     "Direct disk write — can overwrite partitions or devices"),
    ("> /dev/sd",  "Writing to raw disk device"),
    ("chmod -R 777",  "Recursively setting world-writable permissions"),
    ("chmod -R 000",  "Recursively removing all permissions"),
    (":(){ :|:& };:", "Fork bomb — will crash the system"),
)

_ALWAYS_DESTRUCTIVE_COMMANDS = frozenset({"shred", "wipefs"})

_SEMANTIC_READ_ONLY_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "less", "more", "wc", "sort", "uniq",
    "grep", "egrep", "fgrep", "find", "which", "whereis", "whatis", "man",
    "info", "file", "stat", "du", "df", "free", "uptime", "uname", "hostname",
    "whoami", "id", "groups", "env", "printenv", "echo", "printf", "date",
    "cal", "bc", "expr", "test", "true", "false", "pwd", "tree", "diff",
    "cmp", "md5sum", "sha256sum", "sha1sum", "xxd", "od", "hexdump", "strings",
    "readlink", "realpath", "basename", "dirname", "seq", "yes", "tput",
    "column", "jq", "yq", "xargs", "tr", "cut", "paste", "awk", "sed",
})

_NETWORK_COMMANDS = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "ftp", "sftp", "nc", "ncat",
    "telnet", "ping", "traceroute", "dig", "nslookup", "host", "whois",
    "ifconfig", "ip", "netstat", "ss", "nmap",
})

_PROCESS_COMMANDS = frozenset({
    "kill", "pkill", "killall", "ps", "top", "htop", "bg", "fg", "jobs",
    "nohup", "disown", "wait", "nice", "renice",
})

_PACKAGE_COMMANDS = frozenset({
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "pip", "pip3",
    "npm", "yarn", "pnpm", "bun", "cargo", "gem", "go", "rustup", "snap", "flatpak",
})

_SYSTEM_ADMIN_COMMANDS = frozenset({
    "sudo", "su", "chroot", "mount", "umount", "fdisk", "parted", "lsblk",
    "blkid", "systemctl", "service", "journalctl", "dmesg", "modprobe",
    "insmod", "rmmod", "iptables", "ufw", "firewall-cmd", "sysctl",
    "crontab", "at", "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "passwd", "visudo",
})

_SYSTEM_PATHS = (
    "/etc/", "/usr/", "/var/", "/boot/", "/sys/",
    "/proc/", "/dev/", "/sbin/", "/lib/", "/opt/",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_first_command(command: str) -> str:
    """Extract the first bare command, skipping env var assignments."""
    remaining = command.strip()

    # Skip leading KEY=value pairs
    import re
    while remaining:
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=\S*\s+', remaining)
        if m:
            remaining = remaining[m.end():]
        else:
            break

    parts = remaining.split()
    if not parts:
        return ""
    return Path(parts[0]).name  # strip path prefix (/usr/bin/grep → grep)


def _extract_sudo_inner(command: str) -> str:
    """Return the command after 'sudo', skipping sudo flags."""
    parts = command.split()
    try:
        idx = parts.index("sudo")
    except ValueError:
        return ""
    rest = parts[idx + 1:]
    for i, part in enumerate(rest):
        if not part.startswith("-"):
            return " ".join(rest[i:])
    return ""


# ---------------------------------------------------------------------------
# Stage 1 — read-only validation
# ---------------------------------------------------------------------------

def _validate_git_read_only(command: str) -> ValidationResult:
    parts = command.split()
    subcommand = next((p for p in parts[1:] if not p.startswith("-")), None)
    if subcommand is None:
        return Allow()
    if subcommand in _GIT_READ_ONLY_SUBCOMMANDS:
        return Allow()
    return Block(
        f"Git subcommand '{subcommand}' modifies repository state "
        f"and is not allowed in read-only mode"
    )


def validate_read_only(command: str, mode: str) -> ValidationResult:
    """Block write-like commands when mode is read_only."""
    if mode != "read_only":
        return Allow()

    first = _extract_first_command(command)

    if first in _WRITE_COMMANDS:
        return Block(
            f"Command '{first}' modifies the filesystem "
            f"and is not allowed in read-only mode"
        )

    if first in _STATE_MODIFYING_COMMANDS:
        return Block(
            f"Command '{first}' modifies system state "
            f"and is not allowed in read-only mode"
        )

    # sudo wrapping a write command
    if first == "sudo":
        inner = _extract_sudo_inner(command)
        if inner:
            inner_result = validate_read_only(inner, mode)
            if not isinstance(inner_result, Allow):
                return inner_result

    # Shell redirect operators
    for redir in _WRITE_REDIRECTIONS:
        if redir in command:
            return Block(
                f"Command contains write redirection '{redir}' "
                f"which is not allowed in read-only mode"
            )

    if first == "git":
        return _validate_git_read_only(command)

    return Allow()


# ---------------------------------------------------------------------------
# Stage 2 — sed validation
# ---------------------------------------------------------------------------

def validate_sed(command: str, mode: str) -> ValidationResult:
    """Block sed -i (in-place editing) in read-only mode."""
    first = _extract_first_command(command)
    if first != "sed":
        return Allow()
    if mode == "read_only" and " -i" in command:
        return Block("sed -i (in-place editing) is not allowed in read-only mode")
    return Allow()


# ---------------------------------------------------------------------------
# Stage 3 — destructive command warnings
# ---------------------------------------------------------------------------

def check_destructive(command: str) -> ValidationResult:
    """Warn on known destructive patterns regardless of mode."""
    for pattern, warning in _DESTRUCTIVE_PATTERNS:
        if pattern in command:
            return Warn(f"Destructive command detected: {warning}")

    first = _extract_first_command(command)
    if first in _ALWAYS_DESTRUCTIVE_COMMANDS:
        return Warn(
            f"Command '{first}' is inherently destructive and may cause data loss"
        )

    # Any remaining rm -rf not caught by specific patterns
    if "rm " in command and "-r" in command and "-f" in command:
        return Warn(
            "Recursive forced deletion detected — verify the target path is correct"
        )

    return Allow()


# ---------------------------------------------------------------------------
# Stage 4 — mode validation
# ---------------------------------------------------------------------------

def _command_targets_outside_workspace(command: str) -> bool:
    first = _extract_first_command(command)
    is_write = first in _WRITE_COMMANDS or first in _STATE_MODIFYING_COMMANDS
    if not is_write:
        return False
    return any(sys_path in command for sys_path in _SYSTEM_PATHS)


def validate_mode(command: str, mode: str) -> ValidationResult:
    """Enforce permission mode constraints."""
    if mode == "read_only":
        return validate_read_only(command, mode)
    if mode == "workspace_write":
        if _command_targets_outside_workspace(command):
            return Warn(
                "Command appears to target files outside the workspace "
                "— requires elevated permission"
            )
    return Allow()


# ---------------------------------------------------------------------------
# Stage 5 — path validation
# ---------------------------------------------------------------------------

def validate_paths(command: str, workspace: Path) -> ValidationResult:
    """Warn on directory traversal and home directory escapes."""
    if "../" in command:
        workspace_str = str(workspace)
        if workspace_str not in command:
            return Warn(
                "Command contains directory traversal pattern '../' "
                "— verify the target path resolves within the workspace"
            )
    if "~/" in command or "$HOME" in command:
        return Warn(
            "Command references home directory "
            "— verify it stays within the workspace scope"
        )
    return Allow()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def validate_command(command: str, mode: str, workspace: Path) -> ValidationResult:
    """Run all validation stages. Returns first non-Allow result, else Allow."""
    for stage in (
        lambda: validate_mode(command, mode),
        lambda: validate_sed(command, mode),
        lambda: check_destructive(command),
        lambda: validate_paths(command, workspace),
    ):
        result = stage()
        if not isinstance(result, Allow):
            return result
    return Allow()


# ---------------------------------------------------------------------------
# Command semantic classification
# ---------------------------------------------------------------------------

def _classify_git(command: str) -> CommandIntent:
    parts = command.split()
    subcommand = next((p for p in parts[1:] if not p.startswith("-")), None)
    if subcommand in _GIT_READ_ONLY_SUBCOMMANDS:
        return CommandIntent.ReadOnly
    return CommandIntent.Write


def classify_command(command: str) -> CommandIntent:
    """Classify the semantic intent of a bash command into 8 categories."""
    first = _extract_first_command(command)

    if first in _SEMANTIC_READ_ONLY_COMMANDS:
        # sed -i is actually a write
        if first == "sed" and " -i" in command:
            return CommandIntent.Write
        return CommandIntent.ReadOnly

    if first in _ALWAYS_DESTRUCTIVE_COMMANDS or first == "rm":
        return CommandIntent.Destructive

    if first in _WRITE_COMMANDS:
        return CommandIntent.Write

    if first in _NETWORK_COMMANDS:
        return CommandIntent.Network

    if first in _PROCESS_COMMANDS:
        return CommandIntent.ProcessManagement

    if first in _PACKAGE_COMMANDS:
        return CommandIntent.PackageManagement

    if first in _SYSTEM_ADMIN_COMMANDS:
        return CommandIntent.SystemAdmin

    if first == "git":
        return _classify_git(command)

    return CommandIntent.Unknown
