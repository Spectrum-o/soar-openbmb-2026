#!/usr/bin/env bash
# scripts/full_preflight.sh
#
# Single-command pre-submission validation. Runs:
#   1. hard_constraints_lint (strict mode — fails on any FAIL)
#   2. pack_submission --check-only (11-canary preflight)
#   3. md5sum (if --pack provided, packs the tarball and prints md5)
#
# This is the workflow that v3/v5/v5c violated — we packed and submitted
# without auto-validating against SUBMISSIONS.md Hard constraints. With
# this script, that class of mistake is caught BEFORE upload.
#
# Usage:
#   bash scripts/full_preflight.sh --variant <name>           # check only
#   bash scripts/full_preflight.sh --variant <name> --pack    # check + pack tarball
#   bash scripts/full_preflight.sh --all                       # check all variants

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT=""
PACK=0
ALL=0
OUTPUT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --pack)    PACK=1; shift ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --all)     ALL=1; shift ;;
        -h|--help)
            sed -n 's/^# \?//p' "$0" | head -25
            exit 0
            ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

if [ "$ALL" = 0 ] && [ -z "$VARIANT" ]; then
    echo "specify --variant <name> or --all" >&2
    exit 2
fi

# ---- helper ----
preflight_one() {
    local v="$1"
    local v_full="${REPO_ROOT}/${v}"

    if [ ! -d "$v_full" ]; then
        echo "  ✗ variant dir not found: $v_full" >&2
        return 2
    fi

    echo "============================================================"
    echo " FULL PREFLIGHT: $v"
    echo "============================================================"

    # Step 1: Hard Constraints linter (SUBMISSIONS.md row mapping)
    echo
    echo "[1/3] Hard Constraints linter (strict)"
    if ! python3 "$REPO_ROOT/tools/hard_constraints_lint.py" --variant "$v_full" --strict; then
        echo
        echo "  ✗ Hard Constraints linter FAILED — do NOT pack until fixed" >&2
        return 1
    fi

    # Step 2: Latent-assertion lint (catches v23-style shards[0] blind-indexing
    # bugs in any quantize_*.py the variant ships)
    if [ -f "$REPO_ROOT/tools/lint_latent_assertions.py" ]; then
        echo
        echo "[2/3] Latent-assertion linter (v23 bug class)"
        # Scan any quantize_*.py inside the variant dir (the script handles
        # symlinks correctly by resolving them).
        local lint_paths=()
        while IFS= read -r p; do
            lint_paths+=("$p")
        done < <(find "$v_full" -maxdepth 2 -name 'quantize_*.py' -o -name 'apply_*overlay*.py' 2>/dev/null)
        if [ "${#lint_paths[@]}" -gt 0 ]; then
            if ! python3 "$REPO_ROOT/tools/lint_latent_assertions.py" "${lint_paths[@]}" 2>&1; then
                echo
                echo "  ✗ Latent-assertion linter FAILED — quant script has v23-style assertion bug" >&2
                return 1
            fi
        else
            echo "  (no quantize_*.py / overlay scripts to lint)"
        fi
    fi

    # Step 3: pack_submission --check-only
    echo
    echo "[3/3] pack_submission preflight"
    if ! python3 "$REPO_ROOT/tools/pack_submission.py" --variant "$v_full" --check-only; then
        echo
        echo "  ✗ pack_submission preflight FAILED" >&2
        return 1
    fi

    echo
    echo "  ✓ ALL PREFLIGHT CHECKS PASSED for $v"

    if [ "$PACK" = 1 ]; then
        local out="${OUTPUT}"
        if [ -z "$out" ]; then
            out="soar_${v#submission_}_$(date +%Y%m%d_%H%M).tar.gz"
        fi
        echo
        echo "[pack] writing to $out"
        if python3 "$REPO_ROOT/tools/pack_submission.py" --variant "$v_full" --output "$out" 2>&1 | tail -10; then
            echo
            md5sum "$out"
            ls -la "$out"
            echo
            echo "  ✓ TARBALL READY: $out"
        else
            echo "  ✗ pack failed" >&2
            return 1
        fi
    fi

    return 0
}

# ---- run ----
if [ "$ALL" = 1 ]; then
    OVERALL_FAIL=0
    for v in submission_*; do
        [ -d "$v" ] || continue
        if ! preflight_one "$v"; then
            OVERALL_FAIL=1
        fi
        echo
    done
    if [ "$OVERALL_FAIL" = 1 ]; then
        echo "===== SOME VARIANTS FAILED PREFLIGHT =====" >&2
        exit 1
    fi
    echo "===== ALL VARIANTS PASSED PREFLIGHT ====="
    exit 0
else
    preflight_one "$VARIANT"
    exit $?
fi
