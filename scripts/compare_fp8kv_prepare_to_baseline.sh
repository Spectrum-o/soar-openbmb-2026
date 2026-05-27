#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/compare_fp8kv_prepare_to_baseline.sh \
    --baseline-variant submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_safe \
    --fp8kv-tarball /path/to/soar_fp8kv_flashinfer_no_jit_cache_*.tar.gz

Compares the fp8kv final tarball against the platform-proven chunk32k_safe
prepare path. This is not an accuracy test; it is a prepare/startup risk audit.
EOF
}

BASELINE_VARIANT=""
FP8KV_TARBALL=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --baseline-variant)
            BASELINE_VARIANT="${2:-}"
            shift 2
            ;;
        --fp8kv-tarball)
            FP8KV_TARBALL="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "${BASELINE_VARIANT}" ] || [ -z "${FP8KV_TARBALL}" ]; then
    usage >&2
    exit 2
fi

if [ ! -f "${BASELINE_VARIANT}/prepare_env.sh" ]; then
    echo "FAIL: baseline prepare_env.sh not found: ${BASELINE_VARIANT}/prepare_env.sh" >&2
    exit 2
fi
if [ ! -f "${FP8KV_TARBALL}" ]; then
    echo "FAIL: fp8kv tarball not found: ${FP8KV_TARBALL}" >&2
    exit 2
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

baseline_env="${BASELINE_VARIANT}/prepare_env.sh"
fp8kv_env="${tmp_dir}/fp8kv_prepare_env.sh"
tar -xOf "${FP8KV_TARBALL}" ./prepare_env.sh > "${fp8kv_env}"

extract_server_args() {
    local file="$1"
    grep -E '^export SGLANG_SERVER_ARGS=' "${file}" | tail -n 1
}

baseline_args="$(extract_server_args "${baseline_env}")"
fp8kv_args="$(extract_server_args "${fp8kv_env}")"

check_contains() {
    local text="$1"
    local needle="$2"
    local description="$3"
    if [[ "${text}" != *"${needle}"* ]]; then
        echo "FAIL: missing ${description}: ${needle}" >&2
        exit 1
    fi
}

check_not_contains_file() {
    local file="$1"
    local needle="$2"
    local description="$3"
    if grep -Fq "${needle}" "${file}"; then
        echo "FAIL: found ${description}: ${needle}" >&2
        exit 1
    fi
}

check_contains "${baseline_args}" "--chunked-prefill-size 32768" "baseline chunk32k"
check_contains "${baseline_args}" "--max-prefill-tokens 32768" "baseline max prefill"
check_contains "${baseline_args}" "--mem-fraction-static 0.70" "baseline mem fraction"
check_contains "${baseline_args}" "--quantization gptq_marlin" "baseline quantization"
check_contains "${baseline_args}" "--dtype bfloat16" "baseline dtype"

check_contains "${fp8kv_args}" "--chunked-prefill-size 32768" "fp8kv chunk32k"
check_contains "${fp8kv_args}" "--max-prefill-tokens 32768" "fp8kv max prefill"
check_contains "${fp8kv_args}" "--mem-fraction-static 0.70" "fp8kv mem fraction"
check_contains "${fp8kv_args}" "--quantization gptq_marlin" "fp8kv quantization"
check_contains "${fp8kv_args}" "--dtype bfloat16" "fp8kv dtype"
check_contains "${fp8kv_args}" "--kv-cache-dtype fp8_e4m3" "fp8kv KV dtype"
check_contains "${fp8kv_args}" "--cuda-graph-bs 1 2 4 8 12 16 24 32" "fp8kv CUDA graph batch list"
check_contains "${fp8kv_args}" "--max-running-requests 32" "fp8kv max-running-requests"

check_contains "$(cat "${fp8kv_env}")" "ALLOW_FLASH_ATTN_DOWNLOAD" "offline flash-attn guard"
check_contains "$(cat "${fp8kv_env}")" "FATAL: bundled flash-attn wheel missing" "missing-wheel fast failure"
check_contains "$(cat "${fp8kv_env}")" "FORCE_FLASHINFER_CACHE_REBUILD" "opt-in FlashInfer rebuild"
check_contains "$(cat "${fp8kv_env}")" "no reusable FlashInfer cache target found" "platform-local FlashInfer JIT fallback"
check_not_contains_file "${fp8kv_env}" "nuking ~/.cache/flashinfer" "old unconditional cache deletion"
check_not_contains_file "${fp8kv_env}" "bundled flash-attn wheel missing; trying direct prebuilt wheel URL" "old default GitHub fallback"
check_not_contains_file "${fp8kv_env}" "restoring bundled FlashInfer JIT cache" "bundled FlashInfer cache restore"
check_not_contains_file "${fp8kv_env}" "flashinfer_cache_0.5.3_120f.tar.gz" "bundled FlashInfer cache reference"

listing="${tmp_dir}/listing.txt"
tar -tzf "${FP8KV_TARBALL}" > "${listing}"
grep -Eq '^\./flash_attn-.*-cp310-cp310-.*\.whl$' "${listing}"
if grep -Eq '^\./flashinfer_cache_0\.5\.3_120f\.tar\.gz$' "${listing}"; then
    echo "FAIL: fp8kv tarball contains a bundled FlashInfer JIT cache" >&2
    exit 1
fi

cat <<EOF
OK: fp8kv prepare path is aligned with the platform-proven baseline where expected.

baseline=${BASELINE_VARIANT}
fp8kv_tarball=${FP8KV_TARBALL}

shared server args:
- --chunked-prefill-size 32768
- --max-prefill-tokens 32768
- --mem-fraction-static 0.70
- --quantization gptq_marlin
- --dtype bfloat16

fp8kv-only server args:
- --kv-cache-dtype fp8_e4m3
- --cuda-graph-bs 1 2 4 8 12 16 24 32
- --max-running-requests 32

prepare-timeout guards verified:
- bundled cp310 flash-attn wheel present in final tarball
- no bundled FlashInfer JIT cache is present
- flash-attn GitHub download is opt-in only
- FlashInfer cache rebuild is opt-in only
EOF
