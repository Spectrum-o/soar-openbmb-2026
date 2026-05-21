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

python_pkg_major_ok() {
    python3 - "$1" "$2" "$3" <<'PY'
import importlib.metadata as metadata
import re
import sys

pkg, min_major, max_major = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    version = metadata.version(pkg)
except metadata.PackageNotFoundError:
    raise SystemExit(1)

match = re.match(r"(\d+)", version)
if not match:
    raise SystemExit(1)
major = int(match.group(1))
raise SystemExit(0 if min_major <= major < max_major else 1)
PY
}

python_pkg_min_version() {
    python3 - "$1" "$2" <<'PY'
import importlib.metadata as metadata
import re
import sys

pkg, min_version = sys.argv[1], sys.argv[2]
try:
    version = metadata.version(pkg)
except metadata.PackageNotFoundError:
    raise SystemExit(1)

def parts(value: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", value)
    return tuple(int(x) for x in nums[:3])

current = parts(version)
minimum = parts(min_version)
width = max(len(current), len(minimum))
current = current + (0,) * (width - len(current))
minimum = minimum + (0,) * (width - len(minimum))
raise SystemExit(0 if current >= minimum else 1)
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
    "gptqmodel",
    "flash-attn",
    "flash-linear-attention",
    "tokenizers",
    "huggingface-hub",
    "accelerate",
    "ninja",
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

if [ -d "${SUBMISSION_DIR}/sglang/python" ]; then
    echo "[prepare_env] installing submitted sglang/python in editable mode"
    uv pip install --no-deps -e "${SUBMISSION_DIR}/sglang/python"
fi

# GPTQModel for onsite Hessian-based W4A16 quantization. PINNED to the
# EXACT version verified on AutoDL (96 qzeros tensors patched 7->8 by
# fix_qzeros_for_marlin in the v21 quant run on 2026-05-20). The previous
# major-range gate `>=7.0,<8.0` accepted any 7.x; when the platform base
# env already had a different 7.x, the install was skipped and we ran
# with an untested binary. v21 (local acc=49) and v22 (full-attn) both
# scored 0 on the platform with bench timings identical to v17 — the
# best-fit hypothesis is platform's gptqmodel writing qzeros in a layout
# that the pre-2026-05-21 fix_qzeros silently no-op'd on. Pinning here
# makes the install side deterministic; the hardened fix_qzeros also
# now hard-exits on any layout mismatch, so if pin and hardening together
# still fail, the platform log will tell us *why* instead of just acc=0.
GPTQMODEL_PIN="${GPTQMODEL_PIN:-7.0.0}"

if python_pkg_exact_version gptqmodel "${GPTQMODEL_PIN}" && \
   python_pkg_min_version transformers 4.45 && \
   python_pkg_present accelerate && \
   python_pkg_present ninja; then
    echo "[prepare_env] gptqmodel ${GPTQMODEL_PIN} already installed; skipping PyPI download"
else
    installed_gptqmodel=$(python3 -c "import importlib.metadata as m; print(m.version('gptqmodel'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] forcing gptqmodel==${GPTQMODEL_PIN} (currently: ${installed_gptqmodel})"
    install_with_cn_fallbacks "gptqmodel==${GPTQMODEL_PIN}" "transformers>=4.45" accelerate ninja
fi

# Install flash-attn. SALA's HF modeling code (loaded via trust_remote_code)
# hard-asserts `_attn_implementation == "flash_attention_2"` at __init__
# (modeling_minicpm_sala.py:1328), and transformers >= 5.0 ADDITIONALLY
# does a strict import-check that requires the flash_attn package to be
# importable at model instantiation time (modeling_utils.py:1714).
#
# The 2026-05-19 21:16 submission crashed in 17s here:
#   ImportError: FlashAttention2 has been toggled on, but it cannot be used
#   due to the following error: the package for FlashAttention2 doesn't
#   seem to be installed.
#
# Platform env (confirmed from prior submission logs): torch 2.9.1+cu128,
# python 3.10. The exact-match prebuilt wheel from mjun0812's repo is
# `flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl` — about
# 253 MB, installs in seconds, no source compile. Using the direct URL
# avoids the `+cu128` local-version bug that breaks pip's wheel auto-
# resolution from `pip install flash-attn`.
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
            if [ "${ALLOW_FLASH_ATTN_SOURCE_BUILD:-0}" = "1" ]; then
                echo "[prepare_env] direct wheel failed; ALLOW_FLASH_ATTN_SOURCE_BUILD=1 so trying source build" >&2
                uv pip install --no-build-isolation flash-attn
            else
                echo "[prepare_env] FATAL: flash-attn wheel install failed and source build is disabled" >&2
                echo "[prepare_env]        include flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl in the submission package" >&2
                exit 1
            fi
        fi
    fi
fi

# Verify the import works before proceeding — fail fast if anything's wrong.
if ! python3 -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} import OK')"; then
    echo "[prepare_env] FATAL: flash_attn imports failed after install" >&2
    exit 1
fi

print_versions

# GPTQ/Marlin runtime expects fp16-compatible MiniCPM sparse attention helpers.
# This patch was verified necessary in earlier submissions — the sparse
# backend hardcodes bf16, but Marlin GEMM emits fp16.
BACKEND_DIR="${SUBMISSION_DIR}/sglang/python/sglang/srt/layers/attention"
for pyfile in "${BACKEND_DIR}/minicpm_backend.py" "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
    if [ -f "${pyfile}" ]; then
        sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
        sed -i 's/"bfloat16"/"float16"/g' "${pyfile}"
        echo "[prepare_env] patched $(basename "${pyfile}") for fp16 quantized run"
    fi
done

# Marlin GEMM env. Use FP32 accumulation for higher numerical stability
# during decode (small perf cost; safer for correctness gate).
export GPTQMODEL_MARLIN_USE_FP32="${GPTQMODEL_MARLIN_USE_FP32:-1}"

# SGLang server args. NOTE: NO --kv-cache-dtype fp8_* (verified incompatible
# with MiniCPM sparse backend in earlier submissions).
# Chunked-prefill tuning (cherry-picked 2026-05-21 from config/chunked-prefill-tuned):
#   chunk=8192   -> 267.67 tok/s,  TTFT 34049ms  (old baseline)
#   chunk=32768  -> 425.58 tok/s,  TTFT 13545ms  (+59% throughput / -60% TTFT)
#   chunk=65536  -> 488.84 tok/s,  TTFT 12529ms  (+83% throughput / -63% TTFT)
# RTX PRO 6000 Blackwell, 64 prompts x 4096-in x 512-out random-ids.
# Two gotchas the source branch documented:
#   1. sglang default --max-prefill-tokens=16384 silently caps actual prefill
#      regardless of --chunked-prefill-size. Must set it to match.
#   2. With large chunks, sglang's auto-calc of mem_fraction_static can go
#      negative. Must explicitly set --mem-fraction-static (0.80 verified).
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --mem-fraction-static 0.80 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --dtype float16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
