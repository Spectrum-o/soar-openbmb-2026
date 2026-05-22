#!/usr/bin/env bash
# sanity_check.sh — verify all infrastructure is healthy.
# Runs in <5 seconds. No GPU needed.
#
# Use after any commit / pull / rebase to confirm nothing regressed.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "===================================================================="
echo " sanity_check — local-infra health verifier"
echo "===================================================================="

# 1. Test suite
echo
echo "[1/4] Unit test suite (106 tests expected)..."
if test_output=$(python3 -m unittest discover tests 2>&1 | tail -3); then
    echo "$test_output" | sed 's/^/    /'
else
    echo "    FAIL — test suite did not pass cleanly" >&2
    echo "$test_output" | sed 's/^/    /' >&2
    exit 1
fi

# 2. Preflight on every submission_* variant dir
echo
echo "[2/4] Variant dir preflights..."
fail_count=0
total=0
for d in submission_gptqmodel_calib_w4a16 \
         submission_gptq_v17_minconfig \
         submission_gptqmodel_calib_w4a16_v24 \
         submission_gptqmodel_calib_w4a16_v24_no_dtype_key \
         submission_gptqmodel_calib_w4a16_v24_pin_transformers \
         submission_gptqmodel_calib_w4a16_v24_bits8 \
         submission_gptqmodel_calib_w4a16_v25_calib_multi_adaptive \
         submission_gptqmodel_calib_w4a16_v25_g64; do
    [ -d "$d" ] || continue
    total=$((total + 1))
    result=$(python3 tools/pack_submission.py --variant "$d" --check-only 2>&1 | tail -1)
    if echo "$result" | grep -q "OK (no problems)"; then
        echo "    ✓ $d"
    else
        echo "    ✗ $d : $result" >&2
        fail_count=$((fail_count + 1))
    fi
done
echo
echo "    $((total - fail_count))/$total variants preflight clean"
[ "$fail_count" -eq 0 ] || exit 2

# 3. CLI smoke tests on key tools
echo
echo "[3/4] Tool --help smoke tests..."
for tool in tools/parse_quant_diagnostic.py tools/check_tokenizer_compat.py \
            tools/compute_corrected_acc.py tools/pack_submission.py \
            scripts/decide_next_variant.py; do
    if python3 "$tool" --help >/dev/null 2>&1; then
        echo "    ✓ $tool --help"
    else
        echo "    ✗ $tool --help" >&2
        exit 3
    fi
done

# 4. Shell scripts: syntax check
echo
echo "[4/4] Shell script syntax checks..."
for sh in scripts/local_eval.sh scripts/peek.sh scripts/v24_auto.sh \
          scripts/overwrite_tokenizer_with_base.sh \
          submission_gptqmodel_calib_w4a16/prepare_env.sh \
          submission_gptqmodel_calib_w4a16/prepare_model.sh; do
    if bash -n "$sh"; then
        echo "    ✓ $sh"
    else
        echo "    ✗ $sh" >&2
        exit 4
    fi
done

echo
echo "===================================================================="
echo " ALL GREEN. 106 tests + variant preflights + CLI + shell syntax OK."
echo "===================================================================="
