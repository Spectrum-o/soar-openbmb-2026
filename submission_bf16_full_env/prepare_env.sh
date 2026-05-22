#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] submission dir: ${SUBMISSION_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[prepare_env] fatal: uv is required by the SOAR base environment" >&2
    return 1 2>/dev/null || exit 1
fi

python_pkg_present() {
    python3 - "$1" <<'PY'
import importlib.metadata as metadata
import sys

try:
    metadata.version(sys.argv[1])
except metadata.PackageNotFoundError:
    raise SystemExit(1)
PY
}

python_pkg_exact_version() {
    python3 - "$1" "$2" <<'PY'
import importlib.metadata as metadata
import sys

pkg, expected = sys.argv[1], sys.argv[2]
try:
    version = metadata.version(pkg)
except metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if version == expected else 1)
PY
}

flash_attn_importable() {
    python3 - <<'PY'
import importlib.util

raise SystemExit(0 if importlib.util.find_spec("flash_attn") is not None else 1)
PY
}

print_versions() {
    python3 - <<'PY'
import importlib.metadata as metadata
import sys

print(f"[versions] python={sys.version.split()[0]}", flush=True)
for package in [
    "torch",
    "transformers",
    "flash-attn",
    "flash-linear-attention",
    "tokenizers",
    "huggingface-hub",
    "accelerate",
    "sglang",
]:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "NOT_INSTALLED"
    print(f"[versions] {package}={version}", flush=True)
PY
}

install_with_cn_fallbacks() {
    local index
    local -a indexes=(
        "${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
        "https://mirrors.aliyun.com/pypi/simple"
        "https://pypi.org/simple"
    )

    for index in "${indexes[@]}"; do
        echo "[prepare_env] uv pip install via ${index}: $*"
        if uv pip install -v --index-url "${index}" "$@"; then
            return 0
        fi
        echo "[prepare_env] install failed via ${index}, trying next index" >&2
    done

    return 1
}

# ============================================================================
# v5_bf16_full_env — BF16 submission with FULL 1849-style env bundle
# ============================================================================
#
# Why this variant exists:
#   v3 (no-bundle BF16 safetynet) crashed in 11s on 2026-05-23 00:44 with the
#   transformers model_type list dump. Same failure mode as v2 (with bundle).
#   The 2026-05-20 hypothesis ("bundled SGLang vs base env SGLang") was
#   FALSIFIED — both bundled (v2) and not-bundled (v3) BF16 attempts crashed
#   the same way.
#
# Working hypothesis (verified by 1849 W4A16 success on same platform):
#   Platform's base env transformers 5.9.0 + base env SGLang has incomplete
#   MiniCPM-SALA support. To run ANY SALA inference, the submission must:
#     1. Pin transformers to 4.57.1 (cascades hub<1.0, tokenizers 0.22.x)
#     2. Install flash_attn (SALA hard-asserts flash_attention_2)
#     3. Bundle a known-working SGLang version (with our minicpm.py model)
#
# v5 mirrors 1849's prepare_env EXCEPT:
#   - No gptqmodel install (BF16 needs no quant tool)
#   - No fp16 sed-patch (BF16 runs native bf16, not Marlin fp16)
#   - SGLANG_SERVER_ARGS has no --quantization / --dtype flags
#
# Expected: final_score 25-30 (chunked-prefill 32K boost on identity BF16).

if [ -d "${SUBMISSION_DIR}/sglang/python" ]; then
    echo "[prepare_env] installing submitted sglang/python in editable mode"
    uv pip install --no-deps -e "${SUBMISSION_DIR}/sglang/python"
fi

TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"

# Step 1: ensure accelerate present (transformers uses it for device placement)
if ! python_pkg_present accelerate; then
    echo "[prepare_env] installing missing accelerate"
    install_with_cn_fallbacks "accelerate"
fi

# Step 2: force-reinstall transformers to 4.57.1 and enforce dep window.
# DO NOT pass --no-deps — let uv cascade hub down to <1.0.
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed_transformers=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] forcing transformers==${TRANSFORMERS_PIN} (currently: ${installed_transformers})"
    install_with_cn_fallbacks --force-reinstall "transformers==${TRANSFORMERS_PIN}"
fi

echo "[prepare_env] enforcing transformers dependency window: huggingface-hub>=0.34.0,<1.0 tokenizers>=0.22.0,<0.23.0"
install_with_cn_fallbacks \
    "huggingface-hub>=0.34.0,<1.0" \
    "tokenizers>=0.22.0,<0.23.0"

# Step 3: HARD GUARD on transformers + accelerate.
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed_transformers=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: transformers==${TRANSFORMERS_PIN} required after install step, got ${installed_transformers}" >&2
    exit 1
fi
if ! python_pkg_present accelerate; then
    echo "[prepare_env] FATAL: accelerate not installed after install step" >&2
    exit 1
fi

# Step 4: import smoke test — catches dep-skew that metadata can't see
# (the 2026-05-22 18:33 hub=1.16.0 failure mode).
echo "[prepare_env] import smoke test: transformers"
if ! python3 - <<'PY'
import importlib.metadata as metadata
import transformers

print(
    "[smoke] "
    f"transformers={metadata.version('transformers')} "
    f"huggingface-hub={metadata.version('huggingface-hub')} "
    f"tokenizers={metadata.version('tokenizers')} "
    f"accelerate={metadata.version('accelerate')}",
    flush=True,
)
PY
then
    echo "[prepare_env] FATAL: import smoke test failed; deps incompatible with transformers==${TRANSFORMERS_PIN}" >&2
    exit 1
fi

echo "[prepare_env] all required packages present: transformers==${TRANSFORMERS_PIN} accelerate (+ hub/tokenizers cascade)"

# Step 5: install flash-attn (SALA's modeling code requires flash_attention_2)
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
LOCAL_FLASH_ATTN_WHEEL="$(
    find "${SUBMISSION_DIR}" -maxdepth 1 -type f -name 'flash_attn-*.whl' | sort | tail -n 1
)"

if flash_attn_importable; then
    echo "[prepare_env] flash-attn already importable; skipping install"
else
    if [ -n "${LOCAL_FLASH_ATTN_WHEEL}" ]; then
        echo "[prepare_env] installing bundled flash-attn wheel: ${LOCAL_FLASH_ATTN_WHEEL}"
        uv pip install --no-deps --no-build-isolation "${LOCAL_FLASH_ATTN_WHEEL}"
    else
        echo "[prepare_env] bundled flash-attn wheel missing; trying direct prebuilt wheel URL" >&2
        if ! uv pip install --no-deps --no-build-isolation "${FLASH_ATTN_WHEEL}"; then
            echo "[prepare_env] FATAL: flash-attn wheel install failed" >&2
            exit 1
        fi
    fi
fi

if ! python3 -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} import OK')"; then
    echo "[prepare_env] FATAL: flash_attn imports failed after install" >&2
    exit 1
fi

print_versions

# ============================================================================
# SGLang server args — BF16 mode (no quantization, native bf16 dtype)
# ============================================================================
#
# Includes chunked-prefill 32K + mixed-chunk boost (per local 5/14 bench:
# +59% throughput on SALA). Since we BUNDLE our own SGLang (with verified
# MiniCPM-SALA + mixed_chunk forward-mode support in minicpm_backend.py),
# these flags are guaranteed to be recognized by the running SGLang.
#
# Differences vs 1849 (W4A16):
#   - NO --quantization gptq_marlin (BF16 path)
#   - NO --dtype float16 (let SALA run native bf16)
#   - chunked-prefill bumped 8192 -> 32768 (acc unchanged, throughput +59%)
#   - --max-prefill-tokens 32768 (must match chunked size, default 16384 caps)
#   - --enable-mixed-chunk (SARATHI piggyback for decode under prefill)
#
# CRITICAL: do NOT add --kv-cache-dtype fp8_*  (verified incompatible with
# MiniCPM sparse backend in earlier submissions).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --enable-mixed-chunk --skip-server-warmup --dense-as-sparse"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
