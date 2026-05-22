#!/usr/bin/env bash
# v5_bf16_full_env prepare_model.sh
# Pure symlink — no quantization, BF16 identity passthrough.
# Math equivalence to baseline is guaranteed (no weight changes).
set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)  INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: bash prepare_model.sh --input <original> --output <processed>" >&2
    exit 2
fi

echo "[prepare_model] BF16 full-env: symlinking ${INPUT_DIR} -> ${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

if ln -snf "${INPUT_DIR}" "${OUTPUT_DIR}" 2>/dev/null; then
    echo "[prepare_model] symlink created"
else
    echo "[prepare_model] symlink failed; falling back to cp -r"
    rm -rf "${OUTPUT_DIR}"
    cp -r "${INPUT_DIR}" "${OUTPUT_DIR}"
fi

echo "[prepare_model] done"
