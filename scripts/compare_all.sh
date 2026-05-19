#!/usr/bin/env bash
# Run smoke_bench.sh against every submission_*/ variant in the repo,
# then print a pretty summary of scripts/bench_results.csv.
#
# Use after `source sglang_minicpm_sala_env/bin/activate`.
#
# Defaults are aimed at smoke-testing the BF16 stock model:
#   - --bf16 strips --quantization/--dtype, so we can compare flag changes
#     without re-running prepare_model.sh for each variant.
#   - 16 prompts, conc=4 — small enough to finish each run in ~60s.
#
# To test against a quantized model instead, drop --bf16 and pass
# --model-path /path/to/quantized/model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${REPO_ROOT}/scripts/bench_results.csv"

# Pass-through args
PASSTHRU=("--bf16")
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-bf16)
            PASSTHRU=()
            shift
            ;;
        *)
            PASSTHRU+=("$1")
            shift
            ;;
    esac
done

# Discover variants by listing submission_*/ folders. Order: v1, v2, v3, v4.
mapfile -t VARIANTS < <(ls -d "${REPO_ROOT}"/submission_rtn_sym_*/ 2>/dev/null | sort)

if [ "${#VARIANTS[@]}" -eq 0 ]; then
    echo "no submission_rtn_sym_*/ folders found in ${REPO_ROOT}" >&2
    exit 1
fi

echo "found ${#VARIANTS[@]} variants:"
for d in "${VARIANTS[@]}"; do echo "  - $(basename "${d%/}")"; done
echo

for d in "${VARIANTS[@]}"; do
    name="$(basename "${d%/}")"
    echo "=========================================="
    echo " RUNNING ${name}"
    echo "=========================================="
    if ! bash "${REPO_ROOT}/scripts/smoke_bench.sh" --variant "${name}" "${PASSTHRU[@]}"; then
        echo "[compare_all] ${name} failed; continuing to next variant"
    fi
    # Give the GPU a beat to release memory before next launch.
    sleep 5
done

echo
echo "=========================================="
echo " SUMMARY (${RESULTS_CSV})"
echo "=========================================="
if [ -f "${RESULTS_CSV}" ]; then
    if command -v column >/dev/null 2>&1; then
        column -ts, < "${RESULTS_CSV}" | cut -c1-180
    else
        cat "${RESULTS_CSV}"
    fi
fi
