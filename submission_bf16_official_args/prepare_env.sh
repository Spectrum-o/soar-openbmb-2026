#!/usr/bin/env bash
# submission_bf16_official_args/prepare_env.sh
#
# v5g (BF16 native; acc_ori=83.04 on 2026-05-23) + official run_sala.sh
# serving args (65K chunked-prefill, 0.80 mem-fraction, max-prefill 65K,
# max-running-requests 32).
#
# Purpose: CONTROL EXPERIMENT for tonight's W4A16 migration validation.
# Tests whether the official BF16 serving args preserve acc=83 on SALA at
# platform-scale context. If yes -> the args are safe to layer on top of
# W4A16; if acc drops -> args themselves are the problem.
#
# Diff vs submission_bf16_native_bf16/prepare_env.sh (v5g):
#   SGLANG_SERVER_ARGS: chunked-prefill 8192 -> 65536
#                       + --max-prefill-tokens 65536
#                       + --max-running-requests 32
#                       + --mem-fraction-static 0.80
#   Everything else byte-identical to v5g (no fp16 sed-patch, --dtype bfloat16,
#   transformers 4.57.1 pin, hub<1.0 cascade, bundled flash_attn).

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
for package in ["torch", "transformers", "flash-attn", "tokenizers",
                "huggingface-hub", "accelerate", "sglang"]:
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
# transformers pin (H5 fix — verified critical by 1849 success vs v17/v21/v22)
# DO NOT pass --no-deps. transformers 4.57.1 declares hub>=0.34,<1.0 and the
# platform's base env has hub>=1.0. The cascade-downgrade is required.
# ----------------------------------------------------------------------------
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"

if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] forcing transformers==${TRANSFORMERS_PIN} (currently: ${installed})"
    install_with_cn_fallbacks --force-reinstall "transformers==${TRANSFORMERS_PIN}"
fi

echo "[prepare_env] enforcing transformers dependency window: huggingface-hub>=0.34.0,<1.0 tokenizers>=0.22.0,<0.23.0"
install_with_cn_fallbacks \
    "huggingface-hub>=0.34.0,<1.0" \
    "tokenizers>=0.22.0,<0.23.0"

for pkg in accelerate; do
    if ! python_pkg_present "${pkg}"; then
        echo "[prepare_env] installing missing ${pkg}"
        install_with_cn_fallbacks "${pkg}"
    fi
done

# ----------------------------------------------------------------------------
# HARD GUARD on package versions
# ----------------------------------------------------------------------------
if ! python_pkg_exact_version transformers "${TRANSFORMERS_PIN}"; then
    installed=$(python3 -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null || echo "(not installed)")
    echo "[prepare_env] FATAL: transformers==${TRANSFORMERS_PIN} required, got ${installed}" >&2
    exit 1
fi
for pkg in accelerate; do
    if ! python_pkg_present "${pkg}"; then
        echo "[prepare_env] FATAL: ${pkg} not installed after install step" >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Import smoke test (catches dep-skew that metadata can't see)
# ----------------------------------------------------------------------------
echo "[prepare_env] import smoke test: transformers"
if ! python3 - <<'PY'
import importlib.metadata as metadata
import transformers

print(
    "[smoke] "
    f"transformers={metadata.version('transformers')} "
    f"huggingface-hub={metadata.version('huggingface-hub')} "
    f"tokenizers={metadata.version('tokenizers')}",
    flush=True,
)
PY
then
    echo "[prepare_env] FATAL: import smoke test failed" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# flash_attn (SALA requires flash_attention_2; bundled wheel for offline install)
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
# BF16 native — NO fp16 sed-patch (the v5g breakthrough)
# ----------------------------------------------------------------------------
echo "[prepare_env] BF16 native + official serving args — NO fp16 sed-patch"

# ----------------------------------------------------------------------------
# SGLang server args — official run_sala.sh BF16 baseline
#
# CHANGES FROM v5g (acc_ori=83.04):
#   - chunked-prefill-size: 8192 -> 65536 (official tuned value)
#   - + --max-prefill-tokens 65536
#   - + --max-running-requests 32
#   - + --mem-fraction-static 0.80
#
# Source: run_sala.sh / run_sala_w4a16.sh (the official launch scripts)
# All other args byte-identical to v5g.
# ----------------------------------------------------------------------------
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --max-running-requests 32 --mem-fraction-static 0.80 --skip-server-warmup --dense-as-sparse --dtype bfloat16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
