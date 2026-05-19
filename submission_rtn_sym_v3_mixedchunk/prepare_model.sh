#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: bash prepare_model.sh --input <original model path> --output <processed model path> [quantize args...]" >&2
    exit 2
fi

python3 "${SCRIPT_DIR}/quantize_gptq_rtn_sym.py" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
