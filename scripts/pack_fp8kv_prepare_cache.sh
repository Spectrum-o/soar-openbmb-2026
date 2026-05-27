#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="submission_gptqmodel_no_fp16_patch_dtype_bf16_chunk32k_fp8kv_flashinfer"
VARIANT_DIR="${REPO_ROOT}/${VARIANT}"

WHEEL_PATH="${FLASH_ATTN_WHEEL:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
OUTPUT=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/pack_fp8kv_prepare_cache.sh --wheel /path/to/flash_attn-*.whl [--out-dir DIR]
  bash scripts/pack_fp8kv_prepare_cache.sh --wheel /path/to/flash_attn-*.whl --output FILE.tar.gz

Environment:
  FLASH_ATTN_WHEEL  Alternative to --wheel.
  OUT_DIR           Output directory when --output is not set. Default: ./dist

The flash-attn wheel is intentionally not committed to git because common
builds are larger than GitHub's 100 MB single-file limit. This script
temporarily symlinks the local wheel into the variant directory, runs
tools/pack_submission.py, verifies the tarball, then restores the workspace.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --wheel)
            WHEEL_PATH="${2:-}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:-}"
            shift 2
            ;;
        --output)
            OUTPUT="${2:-}"
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

if [ -z "${WHEEL_PATH}" ]; then
    echo "error: --wheel or FLASH_ATTN_WHEEL is required" >&2
    usage >&2
    exit 2
fi

if [ ! -f "${WHEEL_PATH}" ]; then
    echo "error: flash-attn wheel not found: ${WHEEL_PATH}" >&2
    exit 2
fi

wheel_name="$(basename "${WHEEL_PATH}")"
case "${wheel_name}" in
    flash_attn-*.whl) ;;
    *)
        echo "error: wheel filename must match flash_attn-*.whl: ${wheel_name}" >&2
        exit 2
        ;;
esac
case "${wheel_name}" in
    *-cp310-cp310-*) ;;
    *)
        echo "error: wheel must target platform Python cp310-cp310: ${wheel_name}" >&2
        exit 2
        ;;
esac

if [ ! -d "${VARIANT_DIR}" ]; then
    echo "error: variant directory not found: ${VARIANT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
if [ -z "${OUTPUT}" ]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    OUTPUT="${OUT_DIR}/soar_fp8kv_flashinfer_prepare_cache_${stamp}.tar.gz"
fi

tmp_link="${VARIANT_DIR}/${wheel_name}"
backup_dir="$(mktemp -d)"
created_tmp_link=0
cleanup() {
    if [ "${created_tmp_link}" = "1" ]; then
        rm -f "${tmp_link}"
    fi
    if [ -d "${backup_dir}" ]; then
        find "${backup_dir}" -mindepth 1 -maxdepth 1 -exec mv {} "${VARIANT_DIR}/" \;
        rmdir "${backup_dir}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

find "${VARIANT_DIR}" -maxdepth 1 \( -type f -o -type l \) -name 'flash_attn-*.whl' \
    -exec mv {} "${backup_dir}/" \;
ln -s "$(realpath "${WHEEL_PATH}")" "${tmp_link}"
created_tmp_link=1

echo "[pack_fp8kv] repo:    ${REPO_ROOT}"
echo "[pack_fp8kv] variant: ${VARIANT}"
echo "[pack_fp8kv] wheel:   ${WHEEL_PATH}"
echo "[pack_fp8kv] output:  ${OUTPUT}"

cd "${REPO_ROOT}"

bash -n "${VARIANT_DIR}/prepare_env.sh"
python3 tools/pack_submission.py --variant "${VARIANT}" --check-only
python3 tools/pack_submission.py --variant "${VARIANT}" --output "${OUTPUT}"

echo "[pack_fp8kv] verifying tarball contents"
listing="$(mktemp)"
tar -tzf "${OUTPUT}" > "${listing}"
grep -q "^./${wheel_name}$" "${listing}"
grep -q '^./flashinfer_cache_0.5.3_120f.tar.gz$' "${listing}"
tar -xOf "${OUTPUT}" ./prepare_env.sh | grep -q -- '--kv-cache-dtype fp8_e4m3'
tar -xOf "${OUTPUT}" ./prepare_env.sh | grep -q -- '--cuda-graph-bs 1 2 4 8 12 16 24 32'
tar -xOf "${OUTPUT}" ./prepare_env.sh | grep -q -- '--max-running-requests 32'
tar -xOf "${OUTPUT}" ./prepare_env.sh | grep -q 'ALLOW_FLASH_ATTN_DOWNLOAD'

tmp_dir="$(mktemp -d)"
trap 'cleanup; rm -rf "${tmp_dir}" "${listing}"' EXIT
tar -xzf "${OUTPUT}" -C "${tmp_dir}" ./flashinfer_cache_0.5.3_120f.tar.gz
cache_listing="${tmp_dir}/flashinfer_cache.list"
tar -tzf "${tmp_dir}/flashinfer_cache_0.5.3_120f.tar.gz" > "${cache_listing}"
grep -q 'dtype_kv_e4m3.*\.so$' "${cache_listing}"

echo "[pack_fp8kv] OK"
md5sum "${OUTPUT}"
ls -lh "${OUTPUT}"
