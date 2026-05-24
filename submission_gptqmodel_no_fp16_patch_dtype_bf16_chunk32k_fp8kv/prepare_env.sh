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

TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"

# v24_pin_transformers — TWO concerns, handled separately:
#   1. gptqmodel must be EXACTLY pinned, and accelerate must be PRESENT
#      (they were missing on the
#      2026-05-22 18:18 platform run when this was a single bundled
#      `uv pip install --force-reinstall gptqmodel transformers accelerate ninja`
#      call. prepare_env reached prepare_model with gptqmodel/accelerate absent,
#      so the install result must be verified explicitly.)
#   2. transformers must be EXACTLY 4.57.1 (the actual hypothesis under
#      test, because platform's transformers 5.9.0 vs local 4.57.1 likely
#      explains the v23b shape-mismatch crash).
#
# Fix: do each install separately, then HARD-VERIFY every package is present
# before continuing. Never trust uv's exit code alone.

# Step 1: ensure required packages are installed. GPTQModel is pinned exactly;
# accelerate/ninja only need to be importable/present.
if ! python_pkg_exact_version gptqmodel "${GPTQMODEL_PIN}"; then
    installed_gptqmodel=$(python3 -c "import importlib.metadata as m; print(m.version('gptqmodel'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] installing gptqmodel==${GPTQMODEL_PIN} (currently: ${installed_gptqmodel})"
    install_with_cn_fallbacks "gptqmodel==${GPTQMODEL_PIN}"
fi

for pkg_spec in "accelerate" "ninja"; do
    if ! python_pkg_present "${pkg_spec}"; then
        echo "[prepare_env] installing missing ${pkg_spec}"
        install_with_cn_fallbacks "${pkg_spec}"
    fi
done

# Step 2: force-reinstall transformers (the hypothesis variable) and enforce
# the dependency window that transformers 4.57.1 requires.
#
# DO NOT pass --no-deps here. transformers 4.57.1 declares
# huggingface-hub>=0.34.0,<1.0 and the platform's base env had hub==1.16.0
# (paired with transformers 5.9.0). The 2026-05-22 18:33 platform run used
# --no-deps and crashed with `ImportError: huggingface-hub==1.16.0 ...
# required <1.0`.
#
# Even if transformers is already exactly 4.57.1, still enforce the
# transitive ranges below: the failed run reached prepare_model with
# transformers==4.57.1 but hub==1.16.0, so a metadata-only transformers gate is
# insufficient.
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed_transformers=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] forcing transformers==${TRANSFORMERS_PIN} (currently: ${installed_transformers})"
    install_with_cn_fallbacks --force-reinstall "transformers==${TRANSFORMERS_PIN}"
fi

echo "[prepare_env] enforcing transformers dependency window: huggingface-hub>=0.34.0,<1.0 tokenizers>=0.22.0,<0.23.0"
install_with_cn_fallbacks \
    "huggingface-hub>=0.34.0,<1.0" \
    "tokenizers>=0.22.0,<0.23.0"

# Step 3: HARD GUARD. v24_pin_transformers's first platform attempt failed
# because uv reported success while gptqmodel + accelerate were missing.
# Refuse to proceed if any required package is absent, with a loud diagnostic
# so the platform log unambiguously surfaces the cause.
if ! python_pkg_exact_version gptqmodel "${GPTQMODEL_PIN}"; then
    installed_gptqmodel=$(python3 -c "import importlib.metadata as m; print(m.version('gptqmodel'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: gptqmodel==${GPTQMODEL_PIN} required after install step, got ${installed_gptqmodel}" >&2
    exit 1
fi

if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed_transformers=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: transformers==${TRANSFORMERS_PIN} required after install step, got ${installed_transformers}" >&2
    exit 1
fi

for pkg in accelerate ninja; do
    if ! python_pkg_present "${pkg}"; then
        echo "[prepare_env] FATAL: ${pkg} not installed after install step, refusing to proceed" >&2
        echo "[prepare_env] FATAL: this usually means the active --index-url did not have the package's cp$(python3 -c 'import sys; print(f\"{sys.version_info.major}{sys.version_info.minor}\")') wheel" >&2
        exit 1
    fi
done

# IMPORT SMOKE TEST — the 2026-05-22 18:33 run passed metadata-only HARD GUARD
# (transformers==4.57.1 was correctly registered in dist-info) but failed at
# `import transformers` because huggingface-hub was on an incompatible 1.16.0
# from the platform's base env. importlib.metadata can't detect broken-by-dep
# states; an actual import can.
#
# Do not import gptqmodel here: quantize_gptqmodel_w4a16.py applies a small
# transformers-4.57 compatibility shim before importing gptqmodel. A bare
# prepare_env import would be a different code path and can create a false
# failure. Metadata exact-version checks above are enough for gptqmodel.
echo "[prepare_env] import smoke test: transformers"
if ! python3 - <<'PY'
import importlib.metadata as metadata
import transformers

print(
    "[smoke] "
    f"transformers={metadata.version('transformers')} "
    f"huggingface-hub={metadata.version('huggingface-hub')} "
    f"tokenizers={metadata.version('tokenizers')} "
    f"gptqmodel={metadata.version('gptqmodel')}",
    flush=True,
)
PY
then
    echo "[prepare_env] FATAL: import smoke test failed; metadata says packages are installed but they don't actually load" >&2
    echo "[prepare_env] FATAL: see traceback above. Most likely cause: a transitive dep (huggingface-hub, tokenizers) is on a version incompatible with transformers==${TRANSFORMERS_PIN}" >&2
    exit 1
fi

echo "[prepare_env] all required packages present: gptqmodel==${GPTQMODEL_PIN} transformers==${TRANSFORMERS_PIN} accelerate ninja"

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

# v5j: SKIPPING the fp16 sed-patch on sparse backend.
#
# WHY: this patch hard-converts torch.bfloat16 to torch.float16 throughout
# minicpm_backend.py + minicpm_sparse_utils.py. That includes lightning
# attention's recurrent state h_t. fp16 has range ±65504; bf16 has ±3.4e38.
# In lightning recurrence h_t = g_t × h_{t-1} + ..., the state can grow
# large across long contexts → fp16 loses precision / overflows.
#
# 1849 (with this patch) hit acc_ori=46.89. v5d (BF16 + this patch) hit
# acc_ori=46.64 → patch is the prime suspect, NOT --dtype float16 alone.
#
# Official + champion's GPTQ + Marlin recipe does NOT include this patch.
# SGLang's auto-cast at Linear boundaries should keep sparse backend bf16
# while Marlin GEMM internally uses fp16. We are testing whether SGLang's
# auto-cast handles this correctly.
#
# v5j is BYTE-IDENTICAL to 1849 except for this skip — single-variable
# diagnostic. If acc jumps, the sed-patch is the bug; the official Marlin
# path is fine and we should never re-enable this patch.
echo "[prepare_env] v5j: SKIPPING the fp16 sed-patch (mirrors official/champion recipe)"

# Marlin GEMM env. Use FP32 accumulation for higher numerical stability
# during decode (small perf cost; safer for correctness gate).
export GPTQMODEL_MARLIN_USE_FP32="${GPTQMODEL_MARLIN_USE_FP32:-1}"


# ----------------------------------------------------------------------
# FP8 KV CACHE — PATH D PATCH (per SOAR official toolkit guidance)
# ----------------------------------------------------------------------
# Official recommendation: "路径一：量化加速 — GPTQ W4A16 + Marlin Kernel + FP8 KV Cache"
# Specifies: --kv-cache-dtype fp8_e5m2 (NOT fp8_e4m3 — e5m2 has 5 exp bits
# giving ±57344 range, safer for SALA's scale_emb=12 KV magnitudes).
#
# Why we need a patch: SGLang's current GPTQMarlinConfig.get_quant_method
# only handles LinearBase / FusedMoE, not RadixAttention. So
# --quantization gptq_marlin + --kv-cache-dtype fp8_* leaves
# layer.k_scale = None → FlashAttention rejects fp8 query without scales.
#
# Path D fix (3 lines): extend GPTQMarlinConfig.get_quant_method to also
# return BaseKVCacheMethod for RadixAttention. SGLang's standard
# process_weights_after_loading then defaults k_scale to 1.0 when no
# scales are in the checkpoint. Mirrors fp8.py:185-186 pattern.
#
# Compressed_k dtype issue (only triggers without --dense-as-sparse):
# Our SGLANG_SERVER_ARGS includes --dense-as-sparse, which sets
# dense_len=0 (minicpm_backend.py:230) → all sequences go through sparse
# top-k path, never through allocate_and_compress_keys, so the
# bf16↔fp8 compressed_k mismatch doesn't trigger.
TARGET_GPTQ="${SUBMISSION_DIR}/sglang/python/sglang/srt/layers/quantization/gptq.py"
PATCH_TOOL="${SUBMISSION_DIR}/apply_gptq_marlin_kv_method_patch.py"
if [ -f "${TARGET_GPTQ}" ] && [ -f "${PATCH_TOOL}" ]; then
    echo "[prepare_env] applying GPTQMarlin KV cache method patch (Path D)"
    if ! python3 "${PATCH_TOOL}" "${TARGET_GPTQ}"; then
        echo "[prepare_env] FATAL: gptq_marlin KV patch failed" >&2
        exit 1
    fi
else
    echo "[prepare_env] WARN: cannot apply gptq_marlin KV patch — target or tool missing" >&2
fi

# SGLang server args. NOTE: NO --kv-cache-dtype fp8_* (verified incompatible
# with MiniCPM sparse backend in earlier submissions).
#
# CHUNKED-PREFILL: kept at 8192 (the historical v17/v21/v22 value) for v23.
# The +83% throughput config (chunked-prefill 65536 + max-prefill 65536 +
# mem-fraction-static 0.80) measured on config/chunked-prefill-tuned is REAL
# and is in `run_sala.sh` for local benchmarking, but was DELIBERATELY left
# OUT of v23's submission tarball on 2026-05-22 to isolate variables: v23
# differs from v22 in EXACTLY ONE thing — the hardened fix_qzeros_for_marlin.
# If v23 passes, we add chunked-prefill 65K in v24 to compound the perf win.
# Reverting this back to 65K (and the two companion flags) is a one-line
# change once v23's acc is known.
# MEM-FRACTION EXPLICIT (2026-05-23 added):
# Setting 0.80 explicitly to match 1849's verified-safe config on 84GB platform.
# W4A16 model (5GB) + 0.80 × 84 KV (67GB) + 0.7GB buffer (8K chunked-prefill)
# = 72.7GB ≤ 84GB ✓ (11GB headroom). Same budget as 1849 which ran successfully
# on platform (acc_ori=46.89), so OOM risk is essentially zero.
#
# DTYPE BFLOAT16 (v5j_dtype_bf16 variant, NOT v5j):
# v5j tests removing only the sed-patch, keeping --dtype float16.
# THIS variant tests removing BOTH: the sed-patch AND switching dtype to bfloat16.
# Marlin GEMM internally still outputs fp16; SGLang must cast that to bf16
# for the sparse-backend boundary. If v5j gives partial result (50-70 acc),
# this tests whether explicit --dtype bfloat16 fixes the remaining gap.
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashattn --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.70 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --kv-cache-dtype fp8_e5m2 --dtype bfloat16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
