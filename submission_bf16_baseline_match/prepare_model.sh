#!/usr/bin/env bash
# Identity prepare_model: symlink/copy BF16 model through unchanged.
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

echo "[prepare_model] passthrough: ${INPUT_DIR} -> ${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"
rm -rf "${OUTPUT_DIR}"

if ln -s "${INPUT_DIR}" "${OUTPUT_DIR}" 2>/dev/null; then
    echo "[prepare_model] symlink created"
else
    echo "[prepare_model] symlink failed; falling back to cp -r"
    cp -r "${INPUT_DIR}" "${OUTPUT_DIR}"
fi

echo "[prepare_model] done"
