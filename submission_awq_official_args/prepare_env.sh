#!/usr/bin/env bash
# submission_awq_official_args/prepare_env.sh
#
# AWQ via llm-compressor (compressed-tensors format) + OFFICIAL run_sala_w4a16.sh
# serving args (65K chunked-prefill + max-prefill 65K + max-running 32 +
# mem-fraction 0.80).
#
# Purpose: PRIMARY W4A16 CANDIDATE for tonight's migration validation.
# Diff vs submission_awq_llmcompressor/prepare_env.sh:
#   SGLANG_SERVER_ARGS: chunked-prefill 8192 -> 65536
#                       + --max-prefill-tokens 65536
#                       + --max-running-requests 32
#                       + --mem-fraction-static 0.80
#
# All v5g + 1849 baseline ingredients preserved:
#   - llmcompressor (>=0.7) + AWQModifier
#   - transformers 4.57.1 + hub<1.0 cascade
#   - bundled flash_attn wheel
#   - bundled SGLang
#   - NO fp16 sed-patch (compressed-tensors works in bf16)
#   - --quantization compressed-tensors --dtype bfloat16
#   - quantize_llmcompressor_awq.py applies H4 tokenizer overwrite + auto_map strip
#
# Reference: run_sala_w4a16.sh (official launch script):
#   --quantization compressed-tensors \
#   --chunked-prefill-size 65536 \
#   --max-prefill-tokens 65536 \
#   --max-running-requests 32 \
#   --mem-fraction-static 0.80 \
#   --dense-as-sparse

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

def parts(value):
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
    "torch", "transformers", "llmcompressor", "compressed-tensors",
    "datasets", "flash-attn", "tokenizers", "huggingface-hub",
    "accelerate", "sglang",
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

# ----------------------------------------------------------------------------
# Install bundled SGLang (MiniCPM-SALA support)
# ----------------------------------------------------------------------------
if [ -d "${SUBMISSION_DIR}/sglang/python" ]; then
    echo "[prepare_env] installing submitted sglang/python in editable mode"
    uv pip install --no-deps -e "${SUBMISSION_DIR}/sglang/python"
fi

# ----------------------------------------------------------------------------
# llm-compressor install (the key tool for AWQ)
# ----------------------------------------------------------------------------
LLMCOMPRESSOR_MIN="${LLMCOMPRESSOR_MIN:-0.7}"   # AWQModifier requires >= 0.7
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"

if ! python_pkg_min_version llmcompressor "${LLMCOMPRESSOR_MIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('llmcompressor'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] installing llmcompressor>=${LLMCOMPRESSOR_MIN} (currently: ${installed})"
    install_with_cn_fallbacks "llmcompressor>=${LLMCOMPRESSOR_MIN}"
fi

for pkg_spec in "accelerate" "ninja" "datasets"; do
    if ! python_pkg_present "${pkg_spec}"; then
        echo "[prepare_env] installing missing ${pkg_spec}"
        install_with_cn_fallbacks "${pkg_spec}"
    fi
done

# ----------------------------------------------------------------------------
# transformers pin (H5 fix — verified critical by 1849 success vs v17/v21/v22)
# ----------------------------------------------------------------------------
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] forcing transformers==${TRANSFORMERS_PIN} (currently: ${installed})"
    install_with_cn_fallbacks --force-reinstall "transformers==${TRANSFORMERS_PIN}"
fi

echo "[prepare_env] enforcing transformers dependency window: huggingface-hub>=0.34.0,<1.0 tokenizers>=0.22.0,<0.23.0"
install_with_cn_fallbacks \
    "huggingface-hub>=0.34.0,<1.0" \
    "tokenizers>=0.22.0,<0.23.0"

# ----------------------------------------------------------------------------
# HARD GUARD
# ----------------------------------------------------------------------------
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: transformers==${TRANSFORMERS_PIN} required, got ${installed}" >&2
    exit 1
fi
if ! python_pkg_min_version llmcompressor "${LLMCOMPRESSOR_MIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('llmcompressor'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: llmcompressor>=${LLMCOMPRESSOR_MIN} required, got ${installed}" >&2
    exit 1
fi
for pkg in accelerate datasets; do
    if ! python_pkg_present "${pkg}"; then
        echo "[prepare_env] FATAL: ${pkg} not installed after install step" >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Import smoke test
# ----------------------------------------------------------------------------
echo "[prepare_env] import smoke test: transformers + llmcompressor"
if ! python3 - <<'PY'
import importlib.metadata as metadata
import transformers
import llmcompressor

print(
    "[smoke] "
    f"transformers={metadata.version('transformers')} "
    f"llmcompressor={metadata.version('llmcompressor')} "
    f"huggingface-hub={metadata.version('huggingface-hub')} "
    f"tokenizers={metadata.version('tokenizers')} "
    f"accelerate={metadata.version('accelerate')}",
    flush=True,
)

try:
    from llmcompressor.modifiers.awq import AWQModifier
    print("[smoke] AWQModifier import OK", flush=True)
except ImportError as e:
    print(f"[smoke] WARN: AWQModifier import failed: {e}", flush=True)

from llmcompressor.modifiers.quantization import QuantizationModifier
print("[smoke] QuantizationModifier import OK", flush=True)
PY
then
    echo "[prepare_env] FATAL: import smoke test failed" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# flash_attn (SALA requires flash_attention_2)
# ----------------------------------------------------------------------------
LOCAL_FLASH_ATTN_WHEEL="$(
    find "${SUBMISSION_DIR}" -maxdepth 1 -type f -name 'flash_attn-*.whl' | sort | tail -n 1
)"

if flash_attn_importable; then
    echo "[prepare_env] flash-attn already importable; skipping install"
elif [ -n "${LOCAL_FLASH_ATTN_WHEEL}" ]; then
    echo "[prepare_env] installing bundled flash-attn wheel: ${LOCAL_FLASH_ATTN_WHEEL}"
    uv pip install --no-deps --no-build-isolation "${LOCAL_FLASH_ATTN_WHEEL}"
else
    echo "[prepare_env] FATAL: flash-attn bundled wheel missing" >&2
    exit 1
fi

if ! python3 -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} import OK')"; then
    echo "[prepare_env] FATAL: flash_attn imports failed after install" >&2
    exit 1
fi

print_versions

# ----------------------------------------------------------------------------
# SGLang server args — OFFICIAL run_sala_w4a16.sh recipe
#
# CHANGES FROM submission_awq_llmcompressor/prepare_env.sh:
#   - chunked-prefill-size: 8192 -> 65536
#   - + --max-prefill-tokens 65536
#   - + --max-running-requests 32
#   - + --mem-fraction-static 0.80
#
# Source: run_sala_w4a16.sh lines 39-51 (verbatim official W4A16 launch)
#
# NO fp16 sed-patch: compressed-tensors loader is bf16-native (same as v5g)
# NO --dtype float16: --dtype bfloat16 explicit (matches v5g success)
# ----------------------------------------------------------------------------
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80 --skip-server-warmup --dense-as-sparse --quantization compressed-tensors --dtype bfloat16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
