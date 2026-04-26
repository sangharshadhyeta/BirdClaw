#!/usr/bin/env bash
# BirdClaw installer — install | uninstall | update
#
# Usage:
#   sudo ./install.sh install    — full fresh install
#   sudo ./install.sh uninstall  — remove everything (prompts for data)
#   sudo ./install.sh update     — pull latest code + restart services
#
# What gets installed:
#   /opt/birdclaw (or chosen dir)   — BirdClaw source + conda env
#   /opt/llama.cpp (or chosen dir)  — llama.cpp binary (built from source for CUDA)
#   /opt/birdclaw/models/           — GGUF model file
#   /opt/searxng/                   — SearXNG local search backend
#   /usr/local/bin/birdclaw         — CLI wrapper
#   /etc/systemd/system/llama-server.service
#   /etc/systemd/system/birdclaw-daemon.service
#   /etc/systemd/system/searxng.service
#   ~/.birdclaw/.env                — runtime config (BC_* vars, ports)

echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source conda.sh if conda is not already in PATH (common when running via sudo).
if ! command -v conda &>/dev/null; then
    for _conda_sh in \
        /root/miniconda3/etc/profile.d/conda.sh \
        /opt/miniconda3/etc/profile.d/conda.sh \
        /usr/local/miniconda3/etc/profile.d/conda.sh \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$_conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$_conda_sh"
            break
        fi
    done
fi

# Non-interactive mode: set BC_YES=1 or pass --yes to accept all defaults silently.
NONINTERACTIVE="${BC_YES:-0}"
[[ "${2:-}" == "--yes" || "${2:-}" == "-y" ]] && NONINTERACTIVE=1
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && NONINTERACTIVE=1

# Wrapper around read -rp: skips prompt in non-interactive mode and echoes default.
_ask() {
    local prompt=$1 default=$2 var=$3
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        printf "  %s [%s]: %s\n" "$prompt" "$default" "$default" >&2
        printf -v "$var" '%s' "$default"
    else
        local _input
        read -rp "  ${prompt} [${default}]: " _input
        printf -v "$var" '%s' "${_input:-$default}"
    fi
}

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*" >&2; }
ok()      { echo -e "${GREEN}[ OK ]${NC} $*" >&2; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
die()     { echo -e "${RED}[ERR ]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}── $* ──${NC}" >&2; }

# ── Require root ───────────────────────────────────────────────────────────────
require_root() {
    [[ $EUID -eq 0 ]] || die "Run as root:  sudo ./install.sh $*"
}

# ── Port utilities ─────────────────────────────────────────────────────────────
port_in_use() {
    ss -tuln 2>/dev/null | grep -qE ":${1}[[:space:]]"
}

port_status_line() {
    # Print a one-line summary of a port's availability.
    local port=$1
    if port_in_use "$port"; then
        local proc
        proc=$(ss -tulnp 2>/dev/null | grep ":${port}[[:space:]]" | awk '{print $NF}' | head -1)
        echo -e "  ${RED}${port}${NC}  in use  (${proc:-unknown process})"
    else
        echo -e "  ${GREEN}${port}${NC}  free"
    fi
}

pick_port() {
    # Prompt user to confirm or replace a port.  Echoes the chosen port.
    local name=$1 default=$2
    if [[ "$NONINTERACTIVE" == "1" ]]; then
        if port_in_use "$default"; then
            warn "Port ${default} in use — using it anyway (non-interactive)."
        fi
        echo "$default"; return
    fi
    while true; do
        local input
        read -rp "  Port for ${name} [${default}]: " input
        local port="${input:-${default}}"
        if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
            warn "Enter a number between 1024 and 65535."
            continue
        fi
        if port_in_use "$port"; then
            warn "Port ${port} is already in use."
            read -rp "  Use it anyway? [y/N]: " yn
            [[ "$yn" =~ ^[Yy]$ ]] && { echo "$port"; return; }
            continue
        fi
        echo "$port"
        return
    done
}

show_port_check() {
    local llama_port=$1 gw_port=$2
    header "Port availability"
    echo "  llama.cpp server:"
    port_status_line "$llama_port"
    echo "  BirdClaw gateway:"
    port_status_line "$gw_port"
}

# ── Hardware detection ─────────────────────────────────────────────────────────
detect_gpu() {
    # Returns: cuda | vulkan | rocm | cpu
    #
    # cuda   — NVIDIA GPU + CUDA toolkit (nvcc available) → build from source
    # vulkan — NVIDIA GPU with only driver (no nvcc), or any Vulkan-capable GPU → prebuilt
    # rocm   — AMD GPU with ROCm → prebuilt
    # cpu    — no GPU detected → prebuilt CPU build
    if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        # Prefer CUDA source build only if the full toolkit is installed
        if command -v nvcc &>/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]]; then
            echo "cuda"
        else
            echo "vulkan"  # driver present but no toolkit — use Vulkan prebuilt
        fi
    elif command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -q "HSA Agent"; then
        echo "rocm"
    else
        echo "cpu"
    fi
}

detect_vram_mb() {
    # Returns VRAM in MB, or 0 if no GPU.
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' ' || echo 0
    else
        echo 0
    fi
}

suggest_quant() {
    # Suggest quantisation based on VRAM.
    local vram_mb=$1
    if (( vram_mb >= 9000 )); then
        echo "Q8_0"
    elif (( vram_mb >= 6000 )); then
        echo "Q4_K_M"
    else
        echo "Q4_K_M"  # CPU or low VRAM — Q4 is always safest
    fi
}

suggest_gpu_layers() {
    # Suggest --n-gpu-layers based on VRAM and chosen quant.
    local vram_mb=$1 quant=$2
    if (( vram_mb == 0 )); then
        echo 0
        return
    fi
    local model_mb
    case "$quant" in
        Q8_0)   model_mb=8200 ;;
        Q4_K_M) model_mb=5500 ;;
        *)      model_mb=5500 ;;
    esac
    if (( vram_mb >= model_mb + 1500 )); then
        echo -1  # all layers on GPU
    else
        # Rough estimate: each layer ~150 MB for 4B model (18 transformer layers)
        local available=$(( vram_mb - 1500 ))  # reserve 1.5 GB for OS + KV cache
        local layers=$(( available / 150 ))
        (( layers < 0 )) && layers=0
        (( layers > 18 )) && layers=18
        echo "$layers"
    fi
}

# ── llama.cpp binary download / build ─────────────────────────────────────────
_llamacpp_get_tag() {
    curl -fsSL "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])"
}

_llamacpp_build_from_source() {
    # Build llama.cpp with CUDA support from source.
    # Used for Linux+CUDA — prebuilt Linux CUDA binaries no longer shipped upstream.
    local install_dir=$1 tag=$2

    info "No Linux CUDA prebuilt available for ${tag} — building from source."
    info "This takes 5–15 minutes depending on CPU speed."

    for cmd in cmake gcc g++ git; do
        command -v "$cmd" &>/dev/null \
            || die "${cmd} not found. Install build tools:  dnf install cmake gcc gcc-c++ git"
    done

    local src="/tmp/llama_cpp_src_$$"
    info "Cloning llama.cpp ${tag}..."
    git clone -q --depth=1 --branch "$tag" \
        "https://github.com/ggerganov/llama.cpp.git" "$src" \
        || die "git clone failed."

    info "Configuring CMake (CUDA enabled)..."
    cmake -B "${src}/build" -S "$src" \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_AVXVNNI=OFF \
        || die "CMake configuration failed."

    info "Building llama-server (using $(nproc) cores)..."
    cmake --build "${src}/build" --target llama-server -j"$(nproc)" \
        || die "CMake build failed. Check CUDA toolkit is installed: nvidia-smi"

    # Binary location varies by llama.cpp version — search rather than hardcode
    local server_bin
    server_bin=$(find "${src}/build" -name "llama-server" -type f 2>/dev/null | head -1)
    [[ -n "$server_bin" ]] || die "llama-server binary not found after build."

    mkdir -p "${install_dir}/bin"
    cp "$server_bin" "${install_dir}/bin/llama-server"
    chmod +x "${install_dir}/bin/llama-server"
    find "${src}/build/bin" -name "lib*.so*" -type f 2>/dev/null \
        | xargs -I{} cp {} "${install_dir}/bin/" 2>/dev/null || true
    rm -rf "$src"

    ok "llama.cpp built and installed: ${install_dir}/bin/llama-server"
    echo "${install_dir}/bin/llama-server"
}

_llamacpp_download_prebuilt() {
    # Download a prebuilt tar.gz from GitHub releases (CPU / Vulkan / ROCm).
    local install_dir=$1 gpu_type=$2 tag=$3
    local arch
    arch=$(uname -m)

    local release_json
    release_json=$(curl -fsSL \
        "https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/${tag}")

    local asset_url
    asset_url=$(echo "$release_json" | python3 -c "
import json, sys, re
data = json.load(sys.stdin)
gpu = '${gpu_type}'
arch = '${arch}'
arch_tag = 'x64' if arch == 'x86_64' else arch  # releases use x64 not x86_64

# Priority: ROCm → Vulkan (GPU-ish) → plain CPU
patterns = []
if gpu == 'rocm':
    patterns.append(re.compile(r'ubuntu.*rocm.*' + arch_tag + r'.*\.tar\.gz$', re.I))
# Vulkan works on any GPU via driver; skip for CPU-only systems
if gpu != 'cpu':
    patterns.append(re.compile(r'ubuntu.*vulkan.*' + arch_tag + r'.*\.tar\.gz$', re.I))
patterns.append(re.compile(r'ubuntu-' + arch_tag + r'\.tar\.gz$', re.I))
patterns.append(re.compile(r'ubuntu.*' + arch_tag + r'\.tar\.gz$', re.I))

for pat in patterns:
    for a in data.get('assets', []):
        if pat.search(a['name']):
            print(a['browser_download_url'])
            sys.exit(0)
print('')
")

    [[ -n "$asset_url" ]] \
        || die "No prebuilt binary found for arch=${arch} gpu=${gpu_type} tag=${tag}. See https://github.com/ggerganov/llama.cpp/releases"

    local tgz="/tmp/llama_cpp_${tag}.tar.gz"
    info "Downloading ${asset_url##*/}..."
    curl -fL --progress-bar -o "$tgz" "$asset_url"

    info "Extracting to ${install_dir}..."
    mkdir -p "$install_dir"
    tar -xzf "$tgz" -C "$install_dir"
    rm -f "$tgz"

    local server_bin
    server_bin=$(find "$install_dir" -name "llama-server" -type f 2>/dev/null | head -1)
    [[ -n "$server_bin" ]] || die "llama-server not found after extraction."
    chmod +x "$server_bin"
    ok "llama.cpp installed: ${server_bin}"
    echo "$server_bin"
}

_llamacpp_build_no_cuda() {
    # Build llama.cpp from source — with Vulkan if available, else CPU-only.
    # Used when: no CUDA toolkit, OR prebuilt binary is incompatible (GLIBCXX mismatch).
    local install_dir=$1 tag=$2

    for cmd in cmake gcc g++ git; do
        command -v "$cmd" &>/dev/null \
            || die "${cmd} not found. Install build tools:  dnf install cmake gcc gcc-c++ git"
    done

    # Try to enable Vulkan — install system packages if missing.
    # Vulkan + glslc are available in RHEL9/AlmaLinux/Rocky appstream repos.
    local vulkan_flag="-DGGML_VULKAN=OFF"
    if command -v glslc &>/dev/null && pkg-config --exists vulkan 2>/dev/null; then
        vulkan_flag="-DGGML_VULKAN=ON"
        info "Vulkan headers and glslc found — building with GPU (Vulkan) support."
    elif dnf list installed vulkan-headers glslc &>/dev/null 2>&1; then
        vulkan_flag="-DGGML_VULKAN=ON"
        info "Building with GPU (Vulkan) support."
    else
        info "Attempting to install Vulkan build deps (vulkan-headers, vulkan-loader-devel, glslc)..."
        if dnf install -y vulkan-headers vulkan-loader-devel glslc &>/dev/null 2>&1; then
            vulkan_flag="-DGGML_VULKAN=ON"
            info "Vulkan deps installed — building with GPU support."
        else
            warn "Vulkan packages unavailable — building CPU-only."
        fi
    fi

    [[ "$vulkan_flag" == "-DGGML_VULKAN=ON" ]] \
        && info "Building llama.cpp ${tag} from source (Vulkan GPU + RHEL9 compatible)." \
        || info "Building llama.cpp ${tag} from source (CPU-only, RHEL9 compatible)."
    info "This takes 5–15 minutes."

    local src="/tmp/llama_cpp_src_$$"
    info "Cloning llama.cpp ${tag}..."
    git clone -q --depth=1 --branch "$tag" \
        "https://github.com/ggerganov/llama.cpp.git" "$src" \
        || die "git clone failed."

    info "Configuring and building..."
    # GGML_NATIVE=OFF avoids generating instructions (e.g. vpdpbusd/AVX-VNNI)
    # that GCC detects but older binutils on RHEL 9 cannot assemble.
    cmake -B "${src}/build" -S "$src" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_AVXVNNI=OFF \
        "$vulkan_flag" \
        || die "CMake configuration failed."
    cmake --build "${src}/build" --target llama-server -j"$(nproc)" \
        || die "CMake build failed."

    local server_bin
    server_bin=$(find "${src}/build" -name "llama-server" -type f 2>/dev/null | head -1)
    [[ -n "$server_bin" ]] || die "llama-server binary not found after build."

    mkdir -p "${install_dir}/bin"
    cp "$server_bin" "${install_dir}/bin/llama-server"
    chmod +x "${install_dir}/bin/llama-server"

    # Copy shared libraries (.so files) alongside the binary so they are found
    # at runtime regardless of LD_LIBRARY_PATH. The service file also sets
    # LD_LIBRARY_PATH=${install_dir}/bin for safety.
    find "${src}/build/bin" -name "lib*.so*" -type f 2>/dev/null \
        | xargs -I{} cp {} "${install_dir}/bin/" 2>/dev/null || true

    rm -rf "$src"

    ok "llama.cpp built and installed: ${install_dir}/bin/llama-server"
    echo "${install_dir}/bin/llama-server"
}

_llamacpp_check_compatible() {
    # Return 0 (true) if the binary runs on this system, 1 if not.
    local bin=$1
    "$bin" --version &>/dev/null && return 0
    # Check specifically for GLIBCXX mismatch
    "$bin" --version 2>&1 | grep -q "GLIBCXX" && return 1
    return 1
}

download_llamacpp() {
    local install_dir=$1 gpu_type=$2

    info "Fetching latest llama.cpp release tag..."
    local tag
    tag=$(_llamacpp_get_tag) \
        || die "Could not reach GitHub API. Check internet connection."
    info "Latest release: ${tag}"

    # CUDA requires full toolkit → build with CUDA support from source
    if [[ "$gpu_type" == "cuda" ]]; then
        _llamacpp_build_from_source "$install_dir" "$tag"
        return
    fi

    # Try prebuilt first (fast), fall back to source build if GLIBCXX mismatch
    if [[ "$gpu_type" == "vulkan" ]]; then
        info "NVIDIA GPU (no CUDA toolkit) — trying Vulkan prebuilt."
    fi

    local tmp_dir="/tmp/llama_prebuilt_test_$$"
    local prebuilt_bin
    prebuilt_bin=$(_llamacpp_download_prebuilt "$tmp_dir" "$gpu_type" "$tag" 2>/dev/null) || true

    if [[ -n "$prebuilt_bin" ]] && _llamacpp_check_compatible "$prebuilt_bin"; then
        # Prebuilt works — move to final location
        mkdir -p "${install_dir}/bin"
        mv "$prebuilt_bin" "${install_dir}/bin/llama-server"
        rm -rf "$tmp_dir"
        ok "llama.cpp prebuilt installed: ${install_dir}/bin/llama-server"
        echo "${install_dir}/bin/llama-server"
    else
        rm -rf "$tmp_dir" 2>/dev/null || true
        warn "Prebuilt binary incompatible (likely GLIBCXX mismatch) — building from source."
        _llamacpp_build_no_cuda "$install_dir" "$tag"
    fi
}

# ── Model download ─────────────────────────────────────────────────────────────
download_model() {
    local models_dir=$1 quant=$2
    mkdir -p "$models_dir"

    local filename="gemma-4-E2B-it-${quant}.gguf"
    local url="https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/${filename}"
    local dest="${models_dir}/${filename}"

    if [[ -f "$dest" ]]; then
        ok "Model already present: ${dest}"
        echo "$dest"
        return
    fi

    info "Downloading ${filename} (~$([ "$quant" = "Q8_0" ] && echo "2.5 GB" || echo "1.5 GB"))..."
    info "Source: ${url}"
    curl -fL --progress-bar -o "$dest" "$url" \
        || die "Model download failed. If the URL has changed, check https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF"

    ok "Model saved: ${dest}"
    echo "$dest"
}


# ── Conda env + pip install ────────────────────────────────────────────────────
setup_python_env() {
    local install_dir=$1 src_dir=$2
    local env_dir="${install_dir}/env"

    if [[ -x "${env_dir}/bin/python" ]]; then
        info "Python env already exists at ${env_dir}, updating packages..."
    else
        # Remove partial env directory before recreating (avoids conda errors)
        [[ -d "$env_dir" ]] && { info "Removing incomplete env at ${env_dir}..."; rm -rf "$env_dir"; }
        header "Creating Python environment"
        info "Creating Python 3.10 conda environment at ${env_dir}..."
        conda create -y -p "$env_dir" python=3.10 \
            || die "conda create failed. Ensure miniconda/anaconda is installed."
    fi

    header "Installing Python dependencies"

    # Ensure pip is sane
    info "Upgrading pip, setuptools, wheel..."
    conda run -p "$env_dir" python -m pip install --upgrade pip setuptools wheel \
        || die "pip bootstrap failed."

    # Install requirements first (if present)
    if [[ -f "${src_dir}/requirements.txt" ]]; then
        info "Installing requirements.txt..."
        conda run -p "$env_dir" pip install --no-cache-dir -r "${src_dir}/requirements.txt" \
            || die "requirements.txt installation failed."
    fi

    # Install project (editable mode)
    info "Installing BirdClaw (editable mode)..."
    conda run -p "$env_dir" pip install --no-cache-dir -e "$src_dir" \
        || die "BirdClaw install failed."

    # Patch Textual's Linux input driver to tolerate non-UTF-8 terminal bytes
    # (legacy X10 mouse encoding sends bytes > 0x7f which crash the strict decoder).
    local textual_driver
    textual_driver=$(conda run -p "$env_dir" python3 -c \
        "import textual.drivers.linux_driver; print(textual.drivers.linux_driver.__file__)" 2>/dev/null || true)
    if [[ -f "$textual_driver" ]]; then
        sed -i 's/getincrementaldecoder("utf-8")()\.decode/getincrementaldecoder("utf-8")(errors="replace").decode/' \
            "$textual_driver" && info "Textual UTF-8 decoder patched." || warn "Textual patch failed (non-fatal)."
    fi

    ok "Python environment ready."
}

# ── Systemd units ──────────────────────────────────────────────────────────────
# ── SELinux context fix ────────────────────────────────────────────────────────
fix_selinux_contexts() {
    # On SELinux-enforcing systems (RHEL, CentOS, Fedora, Rocky, Alma),
    # conda env binaries inherit admin_home_t from /root/miniconda3/ and
    # systemd's init_t domain cannot execute admin_home_t files.
    # semanage fcontext adds permanent labelling rules; restorecon applies them.
    local install_dir=${1:-}
    local searxng_dir=${2:-}

    [[ "$(getenforce 2>/dev/null)" == "Enforcing" || "$(getenforce 2>/dev/null)" == "Permissive" ]] \
        || return 0  # SELinux not active — nothing to do

    command -v semanage &>/dev/null \
        || { warn "semanage not found — skipping SELinux relabelling (install policycoreutils-python-utils if services fail)"; return 0; }

    info "Applying SELinux context rules for conda envs and miniconda..."

    # miniconda installation — label bin/, envs/, and pkgs/ under each root
    for conda_root in /root/miniconda3 /opt/miniconda3 /usr/local/miniconda3; do
        [[ -d "$conda_root" ]] || continue
        for subdir in bin envs pkgs; do
            [[ -d "${conda_root}/${subdir}" ]] || continue
            semanage fcontext -a -t bin_t "${conda_root}/${subdir}(/.*)?" 2>/dev/null \
                || semanage fcontext -m -t bin_t "${conda_root}/${subdir}(/.*)?" 2>/dev/null || true
        done
        semanage fcontext -a -t lib_t "${conda_root}/lib(/.*)?" 2>/dev/null \
            || semanage fcontext -m -t lib_t "${conda_root}/lib(/.*)?" 2>/dev/null || true
        restorecon -R "$conda_root" 2>/dev/null || true
    done

    # Conda prefix envs at the install locations
    local env_dirs=()
    [[ -n "$install_dir" ]] && env_dirs+=("${install_dir}/env")
    [[ -n "$searxng_dir"  ]] && env_dirs+=("${searxng_dir}/env")

    for env_dir in "${env_dirs[@]}"; do
        [[ -d "$env_dir" ]] || continue
        semanage fcontext -a -t bin_t "${env_dir}/bin(/.*)?" 2>/dev/null \
            || semanage fcontext -m -t bin_t "${env_dir}/bin(/.*)?" 2>/dev/null || true
        semanage fcontext -a -t lib_t "${env_dir}/lib(/.*)?" 2>/dev/null \
            || semanage fcontext -m -t lib_t "${env_dir}/lib(/.*)?" 2>/dev/null || true
        restorecon -R "$env_dir" 2>/dev/null || true
    done

    ok "SELinux contexts updated."
}

write_llamacpp_service() {
    local server_bin=$1 model_path=$2 port=$3 parallel=$4 gpu_layers=$5 run_user=$6
    local ctx=${7:-32768}

    local bin_dir
    bin_dir=$(dirname "$server_bin")

    cat > /etc/systemd/system/llama-server.service <<EOF
[Unit]
Description=llama.cpp server (BirdClaw LLM backend)
After=network.target
Documentation=https://github.com/ggerganov/llama.cpp

[Service]
Type=simple
User=${run_user}
# LD_LIBRARY_PATH ensures shared libs (libggml-vulkan.so etc.) are found when
# built from source — they live alongside the binary in the same directory.
Environment=LD_LIBRARY_PATH=${bin_dir}
ExecStart=${server_bin} \\
    --model ${model_path} \\
    --host 127.0.0.1 \\
    --port ${port} \\
    --parallel ${parallel} \\
    --ctx-size ${ctx} \\
    --n-gpu-layers ${gpu_layers} \\
    --log-verbosity 3
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    ok "Wrote /etc/systemd/system/llama-server.service"
}


write_birdclaw_service() {
    local install_dir=$1 llama_port=$2 gw_port=$3 run_user=$4 searxng_port=$5
    local env_python="${install_dir}/env/bin/python"

    cat > /etc/systemd/system/birdclaw-daemon.service <<EOF
[Unit]
Description=BirdClaw AI Agent Daemon
After=network.target llama-server.service searxng.service
Requires=llama-server.service

[Service]
Type=simple
User=${run_user}
WorkingDirectory=${install_dir}
ExecStart=${env_python} ${install_dir}/main.py daemon
Restart=on-failure
RestartSec=10
Environment=BC_LLM_BASE_URL=http://127.0.0.1:${llama_port}/v1
Environment=BC_GATEWAY_PORT=${gw_port}
Environment=BC_SEARXNG_URL=http://127.0.0.1:${searxng_port}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    ok "Wrote /etc/systemd/system/birdclaw-daemon.service"
}

write_cli_wrapper() {
    local install_dir=$1
    local env_python="${install_dir}/env/bin/python"

    cat > /usr/local/bin/birdclaw <<EOF
#!/usr/bin/env bash
# BirdClaw CLI wrapper — generated by install.sh
exec "${env_python}" "${install_dir}/main.py" "\$@"
EOF
    chmod +x /usr/local/bin/birdclaw
    ok "CLI wrapper installed: /usr/local/bin/birdclaw"
}

# ── SearXNG install ────────────────────────────────────────────────────────────
install_searxng() {
    local searxng_dir=$1 port=$2 run_user=$3
    local env_dir="${searxng_dir}/env"

    if [[ -d "$searxng_dir/.git" ]]; then
        info "SearXNG repo already present — pulling latest..."
        git -C "$searxng_dir" pull -q --ff-only || warn "git pull failed, using existing code."
    else
        info "Cloning SearXNG..."
        git clone -q --depth=1 "https://github.com/searxng/searxng.git" "$searxng_dir" \
            || die "SearXNG clone failed."
    fi

    if [[ -d "$env_dir" ]]; then
        info "SearXNG Python env already exists, updating..."
    else
        info "Creating Python env for SearXNG..."
        conda create -y -p "$env_dir" python=3.10 \
            || die "conda create failed for SearXNG env."
    fi

    info "Installing SearXNG dependencies..."
    # Install requirements.txt first — SearXNG's __init__.py imports msgspec at
    # module level, which makes pip's editable-install build step fail if it isn't
    # already present. Installing requirements separately avoids this.
    if [[ -f "${searxng_dir}/requirements.txt" ]]; then
        conda run -p "$env_dir" pip install -q -r "${searxng_dir}/requirements.txt"
    fi
    conda run -p "$env_dir" pip install -q --no-build-isolation -e "$searxng_dir"

    # Generate a random secret key
    local secret
    secret=$(python3 -c "import secrets; print(secrets.token_hex(32))")

    # Write minimal settings.yml
    mkdir -p "${searxng_dir}/searx"
    cat > "${searxng_dir}/searxng-settings.yml" <<EOF
# SearXNG settings — generated by BirdClaw install.sh
use_default_settings: true

general:
  debug: false
  instance_name: "BirdClaw Search"

search:
  safe_search: 0
  default_lang: "en"
  formats:
    - html
    - json
    - csv
    - rss

server:
  port: ${port}
  bind_address: "127.0.0.1"
  secret_key: "${secret}"
  limiter: false
  image_proxy: false
  http_protocol_version: "1.0"

ui:
  static_use_hash: true
  default_theme: simple

# Engines — web search sources
engines:
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg

  - name: google
    engine: google
    shortcut: g

  - name: bing
    engine: bing
    shortcut: bi

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
EOF

    # Write limiter.toml — bot detection is always active even with limiter: false.
    # Pass-listing localhost lets BirdClaw call the JSON API without 403s.
    # SearXNG resolves limiter.toml from the same folder as searxng-settings.yml
    # when SEARXNG_SETTINGS_PATH points to a file (rule 2 of get_user_cfg_folder).
    cat > "${searxng_dir}/limiter.toml" <<'LIMITEREOF'
[real_ip]
x_for = 1
depth = 1
check_tor = false

[botdetection.ip_lists]
pass_ip = ["127.0.0.1", "::1"]
LIMITEREOF

    # Install tomllib shim — Python 3.10 lacks tomllib (added in 3.11).
    # SearXNG's limiter reads TOML via tomllib; the shim wraps tomli backport.
    local tomllib_shim="${env_dir}/lib/python3.10/site-packages/tomllib.py"
    if [[ ! -f "$tomllib_shim" ]]; then
        conda run -p "$env_dir" pip install -q tomli
        cat > "$tomllib_shim" <<'SHIMEOF'
# tomllib shim: Python 3.10 backport via tomli
from tomli import loads, load
__all__ = ["loads", "load"]
SHIMEOF
    fi

    ok "SearXNG installed at ${searxng_dir}"
}

write_searxng_service() {
    local searxng_dir=$1 port=$2 run_user=$3
    local env_python="${searxng_dir}/env/bin/python"

    cat > /etc/systemd/system/searxng.service <<EOF
[Unit]
Description=SearXNG local search (BirdClaw)
After=network.target

[Service]
Type=simple
User=${run_user}
WorkingDirectory=${searxng_dir}
ExecStart=${env_python} -m searx.webapp
Environment=SEARXNG_SETTINGS_PATH=${searxng_dir}/searxng-settings.yml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    ok "Wrote /etc/systemd/system/searxng.service"
}

write_config() {
    # Write ~/.birdclaw/.env with BC_* vars.
    # pydantic-settings reads this automatically — no hardcoded ports at runtime.
    local config_dir=$1 llama_port=$2 gw_port=$3 model_path=$4 parallel=$5 searxng_port=$6 install_dir=${7:-}
    mkdir -p "$config_dir"

    # Prompt for TUI theme
    local theme="dark"
    echo ""
    echo "  Available TUI themes: dark | light | solarized | catppuccin"
    _ask "TUI theme" "dark" input
    theme="${input:-dark}"

    # Prompt for parallel task mode
    local parallel_tasks="false"
    echo ""
    echo "  Parallel tasks: allow multiple agent tasks to run at the same time."
    echo "  Recommended: no (serial — one task at a time, more reliable on small models)."
    _ask "Enable parallel tasks? (yes/no)" "no" input
    [[ "${input:-no}" =~ ^[Yy] ]] && parallel_tasks="true" || parallel_tasks="false"

    cat > "${config_dir}/.env" <<EOF
# BirdClaw runtime configuration — generated by install.sh
# pydantic-settings reads this at startup (BC_ prefix stripped).
# Edit here to change ports/paths without reinstalling.

# Main model (4B thinker) — tool-call stages, content generation, final answer
BC_LLM_BASE_URL=http://127.0.0.1:${llama_port}/v1
BC_LLM_MODEL=$(basename "${model_path}")

BC_GATEWAY_PORT=${gw_port}
BC_LLAMACPP_PARALLEL=${parallel}
BC_SEARXNG_URL=http://127.0.0.1:${searxng_port}

# TUI colour theme: dark | light | solarized | catppuccin
BC_THEME=${theme}

# LLM request scheduler — gates parallel requests to match llama.cpp slot count.
# Set false only for development/testing.
BC_LLM_SCHEDULER_ENABLED=true

# Parallel tasks — false = serial (one at a time), true = concurrent.
# Serial is recommended for small models (less GPU contention, more reliable output).
BC_PARALLEL_TASKS=${parallel_tasks}

# Workspace roots — directories the agent may read/write (comma-separated).
# Add your project directories here so the agent can access them.
BC_WORKSPACE=${install_dir}
EOF
    ok "Config written: ${config_dir}/.env"

    # ── Buddy GIF ─────────────────────────────────────────────────────────────
    local buddy_dir="${config_dir}/buddy"
    mkdir -p "$buddy_dir"
    local default_gif="${buddy_dir}/alizee.gif"
    if [[ ! -f "$default_gif" ]]; then
        info "Downloading default buddy GIF…"
        if curl -fsSL \
            "https://media.tenor.com/syT6A2VuGiUAAAAM/aliz%C3%A9e-hot-pants.gif" \
            -o "$default_gif" 2>/dev/null; then
            ok "Buddy GIF saved: ${default_gif}"
        else
            warn "Could not download buddy GIF (no internet?). Drop any *.gif into ${buddy_dir}/ later."
        fi
    fi
}

# ── INSTALL ────────────────────────────────────────────────────────────────────
cmd_install() {
    require_root install

    header "BirdClaw installer"

    # ── Source directory ──────────────────────────────────────────────────────
    local src_dir="$SCRIPT_DIR"
    [[ -f "${src_dir}/main.py" ]] || die "Run install.sh from the BirdClaw source directory."

    # ── Install locations ─────────────────────────────────────────────────────
    header "Install locations"
    local install_dir llamacpp_dir
    _ask "BirdClaw install directory" "/opt/birdclaw" input
    install_dir="$input"
    _ask "llama.cpp install directory" "/opt/llama.cpp" input
    llamacpp_dir="$input"

    # ── Ports ─────────────────────────────────────────────────────────────────
    header "Port configuration"
    echo "  Checking default ports..."
    echo ""
    echo "  llama.cpp server  (default 8081):"
    port_status_line 8081
    echo "  BirdClaw gateway  (default 7823):"
    port_status_line 7823
    echo "  SearXNG search    (default 8888):"
    port_status_line 8888
    echo ""
    echo "  Press Enter to accept a default, or type a new port number."
    echo ""

    local llama_port gw_port searxng_port
    llama_port=$(pick_port    "llama.cpp server (4B main)"  8081)
    gw_port=$(pick_port       "BirdClaw gateway"            7823)
    searxng_port=$(pick_port  "SearXNG search"              8888)

    # ── GPU / model ───────────────────────────────────────────────────────────
    header "Hardware detection"
    local gpu_type vram_mb suggested_quant suggested_layers
    gpu_type=$(detect_gpu)
    vram_mb=$(detect_vram_mb)

    if [[ "$gpu_type" == "cuda" ]]; then
        ok "NVIDIA GPU detected  (${vram_mb} MB VRAM, CUDA toolkit present)"
    elif [[ "$gpu_type" == "vulkan" ]]; then
        ok "NVIDIA GPU detected  (${vram_mb} MB VRAM, Vulkan — no CUDA toolkit)"
        info "Using Vulkan prebuilt binary. To enable CUDA: install toolkit + re-run."
    elif [[ "$gpu_type" == "rocm" ]]; then
        ok "AMD GPU detected (ROCm)"
    else
        info "No GPU detected — CPU-only inference"
    fi

    suggested_quant=$(suggest_quant "$vram_mb")
    suggested_layers=$(suggest_gpu_layers "$vram_mb" "$suggested_quant")

    header "Model selection"
    echo "  Available quantisations for gemma-4-E2B-it:"
    echo "    Q4_K_M  — 1.5 GB   fast, good quality"
    echo "    Q8_0    — 2.5 GB   near full precision"
    echo ""
    echo -e "  Suggested based on your hardware: ${BOLD}${suggested_quant}${NC}"
    echo ""
    local quant
    _ask "Quantisation" "$suggested_quant" input
    quant="$input"
    [[ "$quant" =~ ^(Q4_K_M|Q8_0)$ ]] || { warn "Unknown quantisation '${quant}', using ${suggested_quant}."; quant="$suggested_quant"; }

    header "GPU offload"
    echo -e "  Suggested GPU layers: ${BOLD}${suggested_layers}${NC}  (-1 = all layers on GPU, 0 = CPU only)"
    local gpu_layers
    _ask "GPU layers" "$suggested_layers" input
    gpu_layers="$input"

    header "Parallel inference slots"
    echo "  llama.cpp handles multiple requests simultaneously via parallel slots."
    echo "  More slots = more concurrent tasks, but more VRAM/RAM required."
    local parallel
    _ask "Parallel slots" "4" input
    parallel="$input"

    # ── Run user ──────────────────────────────────────────────────────────────
    local run_user="${SUDO_USER:-$USER}"
    [[ "$run_user" == "root" ]] && run_user="root"
    info "Services will run as user: ${run_user}"

    # ── Summary ───────────────────────────────────────────────────────────────
    header "Summary"
    echo "  BirdClaw dir  : ${install_dir}"
    echo "  llama.cpp dir : ${llamacpp_dir}"
    echo "  Main model    : gemma-4-E2B-it-${quant}.gguf  (2B thinker, port ${llama_port})"
    echo "  Gateway port  : ${gw_port}"
    echo "  SearXNG port  : ${searxng_port}"
    echo "  GPU layers    : ${gpu_layers}"
    echo "  Parallel slots: ${parallel}"
    echo "  Run as        : ${run_user}"
    echo ""
    if [[ "$NONINTERACTIVE" != "1" ]]; then
        read -rp "  Proceed? [Y/n]: " confirm
        [[ "${confirm:-Y}" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
    fi

    # ── Check system dependencies ─────────────────────────────────────────────
    header "Checking dependencies"
    for cmd in curl python3 unzip conda ss; do
        command -v "$cmd" &>/dev/null && ok "$cmd" || die "$cmd not found. Install it and re-run."
    done

    # ── Copy source if installing to a different location ─────────────────────
    if [[ "$install_dir" != "$src_dir" ]]; then
        info "Copying source to ${install_dir}..."
        mkdir -p "$install_dir"
        rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='tests/' --exclude='repo/' --exclude='*.egg-info' \
              --exclude='outputs/' \
              "${src_dir}/" "${install_dir}/"
        ok "Source copied."
    fi

    # ── Python environment ────────────────────────────────────────────────────
    header "Python environment"
    setup_python_env "$install_dir" "$install_dir"

    # ── llama.cpp binary ──────────────────────────────────────────────────────
    header "llama.cpp binary"
    local server_bin=""

    # SAFE find (critical fix)
    server_bin=$(find "$llamacpp_dir" -name "llama-server" -type f 2>/dev/null | head -1 || true)

    if [[ -n "$server_bin" ]]; then
        ok "llama-server already installed: ${server_bin}"
    elif server_bin=$(find /tmp -name "llama-server" -type f 2>/dev/null | head -1) && [[ -n "$server_bin" ]] && _llamacpp_check_compatible "$server_bin"; then
        info "Reusing existing build: ${server_bin}"
        mkdir -p "${llamacpp_dir}/bin"
        cp "$server_bin" "${llamacpp_dir}/bin/llama-server"
        # Copy any shared libs alongside it
        find "$(dirname "$server_bin")" -name "lib*.so*" -type f 2>/dev/null \
            | xargs -I{} cp {} "${llamacpp_dir}/bin/" 2>/dev/null || true
        server_bin="${llamacpp_dir}/bin/llama-server"
        ok "llama-server installed from existing build: ${server_bin}"
    else
        info "Installing llama.cpp (GPU type: ${gpu_type})..."

        set +e
        download_llamacpp "$llamacpp_dir" "$gpu_type" > /tmp/llama_install.log 2>&1
        status=$?
        set -e

        cat /tmp/llama_install.log

        server_bin=$(find "$llamacpp_dir" -name "llama-server" -type f 2>/dev/null | head -1 || true)

        if [[ $status -ne 0 || -z "$server_bin" ]]; then
            warn "Primary install failed — retrying CPU fallback..."

            set +e
            download_llamacpp "$llamacpp_dir" "cpu" >> /tmp/llama_install.log 2>&1
            status=$?
            set -e

            cat /tmp/llama_install.log

            server_bin=$(find "$llamacpp_dir" -name "llama-server" -type f 2>/dev/null | head -1 || true)

            if [[ $status -ne 0 || -z "$server_bin" ]]; then
                die "llama.cpp installation failed completely."
            fi
        fi

        ok "llama.cpp ready: ${server_bin}"
    fi

    # ── Model ─────────────────────────────────────────────────────────────────
    header "Model download"
    local models_dir="${install_dir}/models"
    local model_path
    model_path=$(download_model "$models_dir" "$quant")

    # ── SearXNG ───────────────────────────────────────────────────────────────
    header "SearXNG"
    _ask "SearXNG install directory" "/opt/searxng" input
    local searxng_dir="$input"
    install_searxng "$searxng_dir" "$searxng_port" "$run_user"

    # ── Systemd units ─────────────────────────────────────────────────────────
    header "Systemd services"
    write_llamacpp_service "$server_bin" "$model_path" "$llama_port" \
                           "$parallel" "$gpu_layers" "$run_user"
    write_birdclaw_service "$install_dir" "$llama_port" "$gw_port" "$run_user" "$searxng_port"
    write_searxng_service  "$searxng_dir" "$searxng_port" "$run_user"

    systemctl daemon-reload
    systemctl enable llama-server.service birdclaw-daemon.service searxng.service
    ok "Services enabled (will auto-start on boot)."

    # ── CLI wrapper ───────────────────────────────────────────────────────────
    header "CLI wrapper"
    write_cli_wrapper "$install_dir"

    # ── Config ────────────────────────────────────────────────────────────────
    header "Configuration"
    local config_dir
    if [[ "$run_user" == "root" ]]; then
        config_dir="/root/.birdclaw"
    else
        config_dir="$(getent passwd "$run_user" | cut -d: -f6)/.birdclaw"
    fi
    write_config "$config_dir" "$llama_port" "$gw_port" "$model_path" "$parallel" "$searxng_port" "$install_dir"

    # ── SELinux contexts ──────────────────────────────────────────────────────
    header "SELinux"
    fix_selinux_contexts "$install_dir" "$searxng_dir"

    # ── Start services ────────────────────────────────────────────────────────
    header "Starting services"
    info "Starting llama-server..."
    systemctl start llama-server.service
    sleep 3
    if systemctl is-active --quiet llama-server.service; then
        ok "llama-server is running."
    else
        warn "llama-server failed to start. Check:  journalctl -u llama-server -n 40"
    fi

    info "Starting searxng..."
    systemctl start searxng.service
    sleep 2
    if systemctl is-active --quiet searxng.service; then
        ok "searxng is running."
    else
        warn "searxng failed to start. Check:  journalctl -u searxng -n 40"
    fi

    info "Starting birdclaw-daemon..."
    systemctl start birdclaw-daemon.service
    sleep 2
    if systemctl is-active --quiet birdclaw-daemon.service; then
        ok "birdclaw-daemon is running."
    else
        warn "birdclaw-daemon failed to start. Check:  journalctl -u birdclaw-daemon -n 40"
    fi

    # ── Done ──────────────────────────────────────────────────────────────────
    header "Installation complete"
    echo ""
    echo -e "  Run the CLI:          ${BOLD}birdclaw cli${NC}"
    echo -e "  One-shot prompt:      ${BOLD}birdclaw prompt \"your task\"${NC}"
    echo -e "  Dream cycle:          ${BOLD}birdclaw dream${NC}"
    echo -e "  Cleanup stale data:   ${BOLD}birdclaw cleanup${NC}"
    echo -e "  Service status:       ${BOLD}systemctl status llama-server birdclaw-daemon${NC}"
    echo -e "  llama.cpp logs:       ${BOLD}journalctl -u llama-server -f${NC}"
    echo -e "  BirdClaw logs:        ${BOLD}journalctl -u birdclaw-daemon -f${NC}"
    echo ""
    echo -e "  Theme:                edit ${BOLD}~/.birdclaw/.env${NC}  →  BC_THEME=dark|light|solarized|catppuccin"
    echo -e "  Buddy GIF:            drop *.gif files into ${BOLD}~/.birdclaw/buddy/${NC}"
    echo ""
}

# ── UNINSTALL ──────────────────────────────────────────────────────────────────
cmd_uninstall() {
    require_root uninstall

    header "BirdClaw uninstaller"
    warn "This will stop and remove all BirdClaw services and binaries."
    read -rp "  Continue? [y/N]: " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }

    # Stop and disable services
    for svc in birdclaw-daemon searxng llama-server llama-server-hands; do
        if systemctl is-active --quiet "${svc}.service" 2>/dev/null; then
            systemctl stop "${svc}.service"
            ok "Stopped ${svc}"
        fi
        if systemctl is-enabled --quiet "${svc}.service" 2>/dev/null; then
            systemctl disable "${svc}.service"
            ok "Disabled ${svc}"
        fi
        rm -f "/etc/systemd/system/${svc}.service"
    done
    systemctl daemon-reload

    # Remove CLI wrapper
    if [[ -L /usr/local/bin/birdclaw ]] || [[ -f /usr/local/bin/birdclaw ]]; then
        rm -f /usr/local/bin/birdclaw
        ok "Removed /usr/local/bin/birdclaw"
    fi

    # Detect install dir from systemd service (set at install time), fall back to default
    local install_dir
    install_dir=$(systemctl show birdclaw-daemon.service -p WorkingDirectory 2>/dev/null \
                  | cut -d= -f2)
    [[ -n "$install_dir" && -d "$install_dir" ]] || install_dir="/opt/birdclaw"

    # Detect llama.cpp dir from service ExecStart
    local llamacpp_dir
    llamacpp_dir=$(systemctl show llama-server.service -p ExecStart 2>/dev/null \
                   | grep -oP '[^ ]+llama[^ ]+server' | head -1 | xargs dirname 2>/dev/null || true)
    [[ -n "$llamacpp_dir" && -d "$llamacpp_dir" ]] || llamacpp_dir="/opt/llama.cpp"

    # Offer to remove install directories
    for dir in "$install_dir" "$llamacpp_dir" "/opt/searxng"; do
        [[ -n "$dir" ]] || continue
        if [[ -d "$dir" ]]; then
            read -rp "  Remove ${dir}? [y/N]: " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                rm -rf "$dir"
                ok "Removed ${dir}"
            else
                info "Kept ${dir}"
            fi
        fi
    done

    # Offer to remove data/config
    local run_user="${SUDO_USER:-$USER}"
    local config_dir
    if [[ "$run_user" == "root" ]]; then
        config_dir="/root/.birdclaw"
    else
        config_dir="$(getent passwd "$run_user" | cut -d: -f6)/.birdclaw"
    fi

    if [[ -d "$config_dir" ]]; then
        read -rp "  Remove ${config_dir} (memory, config, sessions)? [y/N]: " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            rm -rf "$config_dir"
            ok "Removed ${config_dir}"
        else
            info "Kept ${config_dir} (your memory and config are preserved)"
        fi
    fi

    ok "Uninstall complete."
}

# ── UPDATE ─────────────────────────────────────────────────────────────────────
cmd_update() {
    require_root update

    header "BirdClaw update"

    local src_dir="$SCRIPT_DIR"
    [[ -f "${src_dir}/main.py" ]] || die "Run install.sh from the BirdClaw source directory."

    # Determine install dir from the running service or default
    local install_dir
    install_dir=$(systemctl show birdclaw-daemon.service -p WorkingDirectory 2>/dev/null \
                  | cut -d= -f2)
    [[ -n "$install_dir" && -d "$install_dir" ]] || install_dir="/opt/birdclaw"

    info "Source:  ${src_dir}"
    info "Install: ${install_dir}"

    # Stop services before touching files
    header "Stopping services"
    for svc in birdclaw-daemon; do
        if systemctl is-active --quiet "${svc}.service" 2>/dev/null; then
            systemctl stop "${svc}.service"
            ok "Stopped ${svc}"
        fi
    done

    # Sync source → install dir
    if [[ "$src_dir" != "$install_dir" ]]; then
        info "Syncing source to ${install_dir}..."
        rsync -a --delete \
              --exclude='.git' \
              --exclude='__pycache__' \
              --exclude='*.pyc' \
              --exclude='*.egg-info' \
              --exclude='tests/' \
              --exclude='repo/' \
              --exclude='models/' \
              --exclude='env/' \
              --exclude='outputs/' \
              --exclude='BIRDCLAW.md' \
              "${src_dir}/" "${install_dir}/"
        ok "Source synced."
    else
        # Same directory — optionally git pull
        if [[ -d "${install_dir}/.git" ]]; then
            info "Pulling latest code..."
            git -C "$install_dir" pull --ff-only || warn "git pull failed — continuing."
            ok "Code updated via git."
        fi
    fi

    # Update (or create) Python environment
    local env_dir="${install_dir}/env"
    if [[ -x "${env_dir}/bin/python" ]]; then
        info "Updating Python dependencies..."
        conda run -p "$env_dir" pip install -q --no-cache-dir -e "$install_dir"
        ok "Dependencies updated."
    else
        info "Python env missing or incomplete — creating it now..."
        setup_python_env "$install_dir" "$install_dir"
    fi

    # Recreate systemd services if missing (env path may have changed)
    local server_bin
    server_bin=$(find /opt/llama.cpp -name "llama-server" -type f 2>/dev/null | head -1 || true)
    local model_path
    model_path=$(find "${install_dir}/models" -name "*.gguf" 2>/dev/null | head -1 || true)

    local llama_port gw_port searxng_port
    llama_port=$(grep BC_LLM_BASE_URL ~/.birdclaw/.env 2>/dev/null | grep -oP ':\K[0-9]+(?=/v1)' || echo 8081)
    gw_port=$(grep BC_GATEWAY_PORT ~/.birdclaw/.env 2>/dev/null | cut -d= -f2 || echo 7823)
    searxng_port=$(grep BC_SEARXNG_URL ~/.birdclaw/.env 2>/dev/null | grep -oP ':\K[0-9]+$' || echo 8888)
    local run_user="${SUDO_USER:-root}"

    if [[ -n "$server_bin" && -n "$model_path" ]]; then
        if ! systemctl cat llama-server.service &>/dev/null; then
            info "Recreating llama-server.service..."
            local gpu_layers
            gpu_layers=$(suggest_gpu_layers "$(detect_vram_mb)" "$(suggest_quant "$(detect_vram_mb)")")
            write_llamacpp_service "$server_bin" "$model_path" "$llama_port" 4 "$gpu_layers" "$run_user"
        fi
    fi

    if ! systemctl cat birdclaw-daemon.service &>/dev/null; then
        info "Recreating birdclaw-daemon.service..."
        write_birdclaw_service "$install_dir" "$llama_port" "$gw_port" "$run_user" "$searxng_port"
    fi

    # Write CLI wrapper (env path may have changed)
    write_cli_wrapper "$install_dir"

    # Reload systemd and restart services
    systemctl daemon-reload
    local svcs_to_enable=""
    systemctl cat llama-server.service    &>/dev/null && svcs_to_enable+=" llama-server.service"
    systemctl cat birdclaw-daemon.service &>/dev/null && svcs_to_enable+=" birdclaw-daemon.service"
    # shellcheck disable=SC2086
    [[ -n "$svcs_to_enable" ]] && systemctl enable $svcs_to_enable 2>/dev/null || true

    header "Restarting services"
    for svc in llama-server birdclaw-daemon; do
        systemctl cat "${svc}.service" &>/dev/null || continue
        systemctl restart "${svc}.service" && ok "Restarted ${svc}" \
            || warn "${svc} failed to restart — check: journalctl -u ${svc} -n 20"
    done

    ok "Update complete."
}

# ── Entry point ────────────────────────────────────────────────────────────────
case "${1:-}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    update)    cmd_update ;;
    *)
        echo "Usage: sudo ./install.sh [install|uninstall|update]"
        echo ""
        echo "  install   — fresh install: llama.cpp + model + BirdClaw + services"
        echo "  uninstall — stop services, remove binaries (prompts for data)"
        echo "  update    — sync source → /opt/birdclaw, update deps, restart services"
        exit 1
        ;;
esac
