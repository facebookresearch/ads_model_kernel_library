#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Setup script for the GDPA OSS benchmark environment.
#
# This installs all dependencies needed to benchmark 4 attention kernels:
#   1. CuteDSL GDPA  - CuTe DSL kernel with gelu activation
#   2. Triton GDPA   - Triton kernel with gelu activation (requires TLX)
#   3. CUTLASS FMHA  - NVIDIA CUTLASS Blackwell FMHA via fbgemm-gpu-genai
#   4. CuteDSL FA4   - Tri Dao's Flash Attention v4
#
# Prerequisites:
#   - NVIDIA Blackwell GPU (B200/B100/GB200)
#   - CUDA 12.8+
#   - conda or miniconda
#
# Usage:
#   bash ads_mkl/ops/cute_dsl/gdpa/scripts/setup_benchmark_env.sh
#   conda activate gdpa-bench
#   python ads_mkl/ops/cute_dsl/gdpa/scripts/benchmark.py bench-fwd

set -euo pipefail

ENV_NAME="${GDPA_BENCH_ENV:-gdpa-bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
TRITON_VERSION="${TRITON_VERSION:-3.6.0}"
FBGEMM_GENAI_VERSION="${FBGEMM_GENAI_VERSION:-1.5.0}"

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Detect conda ---
if ! command -v conda &>/dev/null; then
    # Try sourcing conda from common locations
    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/data/users/$USER/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$conda_sh" ]; then
            # shellcheck source=/dev/null
            source "$conda_sh"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    error "conda not found. Install miniconda first:"
    echo "  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh"
    echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
    exit 1
fi

# --- Create conda environment ---
# Unset proxy vars during conda operations — conda on Meta devservers
# has its own proxy config that conflicts with http_proxy/https_proxy.
_saved_http_proxy="${http_proxy:-}"
_saved_https_proxy="${https_proxy:-}"
unset http_proxy https_proxy

if conda env list | grep -q "^${ENV_NAME} "; then
    info "Conda environment '${ENV_NAME}' already exists. Activating..."
else
    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

# Restore proxy vars for pip
export http_proxy="${_saved_http_proxy}"
export https_proxy="${_saved_https_proxy}"

# --- Install dependencies ---
# Install fbgemm-gpu-genai and other deps BEFORE torch, because pip's
# dependency resolver may downgrade torch if installed first.
info "Installing fbgemm-gpu-genai ${FBGEMM_GENAI_VERSION}..."
pip install fbgemm-gpu-genai=="${FBGEMM_GENAI_VERSION}" --quiet

info "Installing benchmark dependencies..."
pip install click matplotlib pandas nvidia-cutlass-dsl apache-tvm-ffi quack-kernels --quiet

# --- Install PyTorch (pinned version, installed last to prevent downgrades) ---
info "Installing PyTorch ${PYTORCH_VERSION} and Triton ${TRITON_VERSION}..."
pip install torch=="${PYTORCH_VERSION}" triton=="${TRITON_VERSION}" --quiet

# --- Install Flash Attention v4 (CuTe DSL) ---
# The CuTe DSL FA4 is a pure Python module (JIT-compiled via cutlass-dsl).
# Install from GitHub main branch; PyPI release may lag behind cutlass-dsl.
info "Installing flash-attn from GitHub (skipping CUDA build)..."
if FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install \
    git+https://github.com/Dao-AILab/flash-attention.git \
    --no-build-isolation --quiet 2>/dev/null; then
    info "flash-attn installed successfully"
else
    warn "Could not install flash-attn from GitHub."
    warn "CuteDSL FA4 will be unavailable. Other kernels will still work."
fi

# --- Create cutlass.utils.ampere_helpers shim ---
# flash-attn imports cutlass.utils.ampere_helpers which was removed in
# cutlass-dsl 4.4.0. Create a minimal shim with the SMEM_CAPACITY dict.
CUTLASS_UTILS_DIR="$(python -c 'import cutlass.utils, os; print(os.path.dirname(cutlass.utils.__file__))')"
if [ -n "$CUTLASS_UTILS_DIR" ] && [ ! -f "$CUTLASS_UTILS_DIR/ampere_helpers.py" ]; then
    info "Creating cutlass.utils.ampere_helpers shim..."
    cat > "$CUTLASS_UTILS_DIR/ampere_helpers.py" << 'PYEOF'
# Compatibility shim for cutlass.utils.ampere_helpers
# This module was removed in cutlass-dsl 4.4.0 but is used by flash-attn
# Only SMEM_CAPACITY is needed

SMEM_CAPACITY = {
    "sm80": 163840,   # 160 KB
    "sm86": 102400,   # 100 KB
    "sm89": 102400,   # 100 KB
    "sm90": 232448,   # 227 KB
    "sm100": 229376,  # 224 KB
}
PYEOF
    info "Created $CUTLASS_UTILS_DIR/ampere_helpers.py"
else
    info "cutlass.utils.ampere_helpers already exists"
fi

# --- Setup Triton TLX (tl.async_task for warp specialization) ---
# The Triton GDPA kernel uses tl.async_task for warp specialization.
# As of triton 3.6.0, async_task is available natively.
# For older versions, we patch the installed triton by symlinking the TLX
# module from the facebookexperimental fork.
if python -c "import triton.language as tl; assert hasattr(tl, 'async_task')" 2>/dev/null; then
    info "tl.async_task already available in triton (native support)"
else
    info "Setting up Triton TLX (tl.async_task)..."

    TRITON_SP="$(python -c 'import triton, os; print(os.path.dirname(triton.__file__))')"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Go up to fbcode (5 levels from scripts/)
    FBCODE_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
    # Go up one more to fbsource
    FBSOURCE_ROOT="$(cd "$FBCODE_ROOT/.." && pwd)"

    # Option 1: Use internal fbsource triton (if available)
    INTERNAL_TLX="${FBSOURCE_ROOT}/third-party/triton/stable/triton/third_party/tlx/language/tlx"

    # Option 2: Clone facebookexperimental/triton for TLX
    EXTERNAL_TLX="${FBCODE_ROOT}/third_party_triton/third_party/tlx/language/tlx"

    TLX_SOURCE=""
    if [ -d "$INTERNAL_TLX" ]; then
        TLX_SOURCE="$INTERNAL_TLX"
        info "Using internal TLX source: $INTERNAL_TLX"
    elif [ -d "$EXTERNAL_TLX" ]; then
        TLX_SOURCE="$EXTERNAL_TLX"
        info "Using previously cloned TLX source: $EXTERNAL_TLX"
    else
        info "Cloning facebookexperimental/triton for TLX..."
        CLONE_DIR="${FBCODE_ROOT}/third_party_triton"
        if git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/facebookexperimental/triton.git "$CLONE_DIR" 2>/dev/null; then
            cd "$CLONE_DIR"
            git sparse-checkout set third_party/tlx/language/tlx
            cd -
            TLX_SOURCE="$EXTERNAL_TLX"
        else
            warn "Could not clone facebookexperimental/triton."
            warn "Triton GDPA will be unavailable. Other kernels will still work."
        fi
    fi

    if [ -n "$TLX_SOURCE" ] && [ -d "$TLX_SOURCE" ]; then
        # Ensure extra/ directory and __init__.py exist
        mkdir -p "$TRITON_SP/language/extra"
        [ -f "$TRITON_SP/language/extra/__init__.py" ] || echo "" > "$TRITON_SP/language/extra/__init__.py"

        # Symlink TLX into triton
        ln -sf "$TLX_SOURCE" "$TRITON_SP/language/extra/tlx"

        # Patch triton/language/__init__.py to export async_task
        if ! grep -q "async_task" "$TRITON_SP/language/__init__.py"; then
            sed -i '/^from \. import extra$/a # Import TLX features (async_task, async_tasks)\nfrom .extra.tlx import async_task, async_tasks' \
                "$TRITON_SP/language/__init__.py"
            info "Patched triton/language/__init__.py to export tl.async_task"
        else
            info "tl.async_task already available"
        fi
    fi
fi

# --- Verify installation ---
echo ""
info "=== Verification ==="
python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')

import triton
print(f'  Triton:     {triton.__version__}')

import triton.language as tl
has_async = hasattr(tl, 'async_task')
print(f'  tl.async_task: {\"available\" if has_async else \"NOT available (Triton GDPA disabled)\"}')

try:
    import tvm_ffi
    print('  tvm_ffi:       available (apache-tvm-ffi)')
except ImportError:
    print('  tvm_ffi:       NOT available (pip install apache-tvm-ffi)')

try:
    import cutlass.utils.ampere_helpers
    print('  ampere_helpers: available')
except ImportError:
    print('  ampere_helpers: NOT available (shim creation failed)')

try:
    import ctypes, glob, os
    for lib_name in ['libcublasLt.so.*[0-9]', 'libcublas.so.*[0-9]']:
        for cuda_dir in ['/usr/local/cuda-12.9/lib64', '/usr/local/cuda-12.8/lib64', '/usr/local/cuda/lib64',
                         '/usr/local/fbcode/platform010/lib']:
            matches = sorted(glob.glob(os.path.join(cuda_dir, lib_name)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                break
    from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import cutlass_blackwell_fmha_func
    print('  CUTLASS FMHA:  available (fbgemm-gpu-genai)')
except Exception as e:
    print(f'  CUTLASS FMHA:  NOT available ({e.__class__.__name__})')

try:
    import sys, types, subprocess, os
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError) as e:
        if 'flash_attn_2_cuda' in str(e):
            for key in list(sys.modules.keys()):
                if key.startswith('flash_attn'):
                    del sys.modules[key]
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', 'flash-attn'],
                capture_output=True, text=True,
            )
            location = None
            for line in result.stdout.split('\n'):
                if line.startswith('Location:'):
                    location = line.split(':', 1)[1].strip()
                    break
            if location:
                fa_path = os.path.join(location, 'flash_attn')
                dummy = types.ModuleType('flash_attn')
                dummy.__path__ = [fa_path]
                dummy.__package__ = 'flash_attn'
                sys.modules['flash_attn'] = dummy
                from flash_attn.cute import flash_attn_varlen_func
        else:
            raise
    print('  CuteDSL FA4:   available (flash-attn)')
except Exception as e:
    print(f'  CuteDSL FA4:   NOT available ({e.__class__.__name__}: {e})')
"

echo ""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "Setup complete! Run the benchmark with:"
echo "  conda activate ${ENV_NAME}"
echo "  cd $(cd "$SCRIPT_DIR/../../../.." && pwd)"
echo "  python ads_mkl/ops/cute_dsl/gdpa/scripts/benchmark.py bench-fwd"
