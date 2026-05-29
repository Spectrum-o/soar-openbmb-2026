#!/usr/bin/env bash
# Run the complete pre-submit check suite for the full-W4A16 selective-BF16 queue.
#
# This intentionally checks only the primary queue candidates from
# tools/full_selective_decide_next.py, not every historical submission_* dir.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mapfile -t VARIANTS < <(python3 tools/full_selective_decide_next.py --variants)
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  echo "FATAL: no full-selective variants returned by helper" >&2
  exit 1
fi

echo "== full-selective unit tests =="
python3 -m unittest \
  tests.test_full_selective_decide_next \
  tests.test_apply_lightning_skip_overlay

echo
echo "== full-selective tarball audit =="
python3 tools/full_selective_decide_next.py --audit-tarballs

echo
echo "== full-selective variant preflight =="
for variant in "${VARIANTS[@]}"; do
  bash scripts/full_preflight.sh --variant "$variant"
done

echo
echo "== cleanup check =="
find tools tests scripts "${VARIANTS[@]}" -type d -name __pycache__ -print -exec rm -rf {} +
leftovers="$(find tools tests scripts "${VARIANTS[@]}" -type d -name __pycache__ -print)"
if [ -n "$leftovers" ]; then
  echo "$leftovers"
  echo "FATAL: __pycache__ remains after cleanup" >&2
  exit 1
fi

echo
echo "FULL-SELECTIVE PRE-SUBMIT OK"
