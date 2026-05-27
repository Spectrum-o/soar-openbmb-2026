#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/audit_fp8kv_submission_tarball.sh /path/to/soar_fp8kv_*.tar.gz

Checks the final SOAR fp8kv tarball, not the git variant directory. This is
intended to catch the prepare-timeout failure modes before the last platform
submission:

- accidentally uploading the tiny runtime snapshot instead of a full package
- missing bundled flash-attn wheel, which would route prepare to GitHub
- wrong Python ABI wheel (platform is cp310)
- missing FlashInfer cache bundle
- accidental default FlashInfer cache deletion
- fp8kv/cudagraph args drifting from the locally verified path
EOF
}

if [ "$#" -ne 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 2
fi

TARBALL="$1"
if [ ! -f "${TARBALL}" ]; then
    echo "FAIL: tarball not found: ${TARBALL}" >&2
    exit 2
fi

size_bytes="$(stat -c '%s' "${TARBALL}")"
if [ "${size_bytes}" -lt $((120 * 1024 * 1024)) ]; then
    echo "FAIL: tarball is too small (${size_bytes} bytes); likely not a full SOAR package" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

listing="${tmp_dir}/listing.txt"
tar -tzf "${TARBALL}" > "${listing}"

require_entry() {
    local pattern="$1"
    local description="$2"
    if ! grep -Eq "${pattern}" "${listing}"; then
        echo "FAIL: missing ${description}" >&2
        exit 1
    fi
}

reject_entry() {
    local pattern="$1"
    local description="$2"
    if grep -Eq "${pattern}" "${listing}"; then
        echo "FAIL: unexpected ${description}" >&2
        exit 1
    fi
}

require_prepare_text() {
    local pattern="$1"
    local description="$2"
    if ! grep -Eq -- "${pattern}" "${tmp_dir}/prepare_env.sh"; then
        echo "FAIL: prepare_env.sh missing ${description}" >&2
        exit 1
    fi
}

reject_prepare_text() {
    local pattern="$1"
    local description="$2"
    if grep -Eq -- "${pattern}" "${tmp_dir}/prepare_env.sh"; then
        echo "FAIL: prepare_env.sh contains ${description}" >&2
        exit 1
    fi
}

require_entry '^\./prepare_env\.sh$' 'prepare_env.sh'
require_entry '^\./prepare_model\.sh$' 'prepare_model.sh'
require_entry '^\./quantize_gptqmodel_w4a16\.py$' 'quantize_gptqmodel_w4a16.py'
require_entry '^\./perf_public_set\.jsonl$' 'perf_public_set.jsonl'
require_entry '^\./sglang/python/sglang/srt/layers/attention/minicpm_backend\.py$' 'bundled SGLang minicpm_backend.py'
require_entry '^\./sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels\.py$' 'bundled SGLang minicpm_attention_kernels.py'
require_entry '^\./sglang/python/sglang/srt/layers/quantization/gptq\.py$' 'bundled SGLang gptq.py'
require_entry '^\./flash_attn-.*-cp310-cp310-.*\.whl$' 'cp310 flash-attn wheel'
reject_entry '^\./flash_attn-.*-cp312-cp312-.*\.whl$' 'cp312 flash-attn wheel'
require_entry '^\./flashinfer_cache_0\.5\.3_120f\.tar\.gz$' 'FlashInfer 0.5.3 120f cache bundle'

tar -xOf "${TARBALL}" ./prepare_env.sh > "${tmp_dir}/prepare_env.sh"
require_prepare_text 'ALLOW_FLASH_ATTN_DOWNLOAD' 'offline flash-attn download guard'
require_prepare_text 'FATAL: bundled flash-attn wheel missing' 'missing-wheel fast failure'
require_prepare_text 'cp310-cp310' 'cp310 wheel guard'
require_prepare_text 'FORCE_FLASHINFER_CACHE_REBUILD' 'opt-in FlashInfer cache rebuild guard'
require_prepare_text '--kv-cache-dtype fp8_e4m3' 'fp8_e4m3 KV cache arg'
require_prepare_text '--cuda-graph-bs 1 2 4 8 12 16 24 32' 'explicit small CUDA graph batch list'
require_prepare_text '--max-running-requests 32' 'max-running-requests 32'
require_prepare_text 'ENABLE_SM120=.*1' 'ENABLE_SM120 default'
require_prepare_text 'FLASHINFER_CUDA_ARCH_LIST=.*12\.0f' 'FlashInfer SM120 arch default'
reject_prepare_text 'nuking ~/.cache/flashinfer' 'old unconditional FlashInfer cache deletion message'
reject_prepare_text 'bundled flash-attn wheel missing; trying direct prebuilt wheel URL' 'old default GitHub download fallback'

tar -xzf "${TARBALL}" -C "${tmp_dir}" ./flashinfer_cache_0.5.3_120f.tar.gz
cache_listing="${tmp_dir}/flashinfer_cache_listing.txt"
tar -tzf "${tmp_dir}/flashinfer_cache_0.5.3_120f.tar.gz" > "${cache_listing}"
if ! grep -Eq '0\.5\.3/120f/cached_ops/.+dtype_kv_e4m3.+\.so$' "${cache_listing}"; then
    echo "FAIL: FlashInfer cache bundle does not contain an e4m3 cached op .so" >&2
    exit 1
fi

echo "OK: ${TARBALL}"
echo "size_bytes=${size_bytes}"
md5sum "${TARBALL}"
