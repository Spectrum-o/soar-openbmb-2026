#!/usr/bin/env bash
# Overwrite a quant artifact's tokenizer files with the base BF16 model's
# originals. Idempotent — safe to re-run.
#
# Why this exists: GPTQModel.save() re-serializes tokenizer.json from
# 3.6M -> 6.7M and synthesizes a separate chat_template.jinja sidecar
# (per the 2026-05-22 H4 finding in
# memory/feedback_gptqmodel_tokenizer_reserialize.md). transformers<4.45
# only reads inline chat_template from tokenizer_config.json, ignoring the
# .jinja sidecar; this yields an empty template and the model receives
# raw prompts with no <|im_start|> markers -> garbage output.
#
# RTN's pipeline copies tokenizer files unconditionally from input dir
# after the numpy quant step, which is why RTN platform-passed at 42 while
# every GPTQModel-based submission was platform=0. This script reproduces
# RTN's behavior on an already-quantized GPTQModel artifact, no re-quant
# needed.
#
# Source-level fix already landed in commit 57f9cef06:
# copy_runtime_assets() now overwrites unconditionally. This script is for
# applying the same fix to an existing quant artifact for fast local
# verification, without re-running the 60+ min quant pipeline.
#
# Usage:
#     bash scripts/overwrite_tokenizer_with_base.sh \
#         --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized \
#         --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA
#
# Flags:
#     --artifact <dir>   The quantized model directory to fix (required)
#     --base <dir>       The base BF16 model directory to copy from (required)
#     --dry-run          Print actions without modifying anything
#     --restore          Restore from the previous backup (revert this script)
#     -h, --help         Show this help

set -euo pipefail

ARTIFACT=""
BASE=""
DRY_RUN=0
RESTORE=0

usage() {
    grep -E '^# (Usage|Flags|    |bash )' "$0" | sed 's/^# //; s/^#//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --artifact) ARTIFACT="$2"; shift 2 ;;
        --base)     BASE="$2";     shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift ;;
        --restore)  RESTORE=1;     shift ;;
        -h|--help)  usage 0 ;;
        *)          echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [ -z "$ARTIFACT" ]; then
    echo "error: --artifact <dir> is required" >&2
    usage 1
fi

if [ ! -d "$ARTIFACT" ]; then
    echo "error: artifact dir not found: $ARTIFACT" >&2
    exit 2
fi

BACKUP_DIR="${ARTIFACT}/_tokenizer_gptqmodel_backup"

# --- Restore mode ---
if [ "$RESTORE" -eq 1 ]; then
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "error: no backup at $BACKUP_DIR — nothing to restore" >&2
        exit 2
    fi
    echo "[restore] reverting tokenizer overwrites in $ARTIFACT"
    for f in "$BACKUP_DIR"/*; do
        [ -f "$f" ] || continue
        name="$(basename "$f")"
        target="${ARTIFACT}/${name}"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[dry-run]   would mv $f -> $target"
        else
            mv "$f" "$target"
            echo "[restore]   restored $name"
        fi
    done
    if [ "$DRY_RUN" -eq 0 ]; then
        rmdir "$BACKUP_DIR" 2>/dev/null || true
        echo "[restore] done"
    fi
    exit 0
fi

# --- Normal mode requires --base ---
if [ -z "$BASE" ]; then
    echo "error: --base <dir> is required (unless --restore)" >&2
    usage 1
fi
if [ ! -d "$BASE" ]; then
    echo "error: base dir not found: $BASE" >&2
    exit 2
fi

# Files we manage. tokenizer.model (sentencepiece) is included for
# completeness even though it's typically already identical between
# base and quant.
MANAGED_FILES=(
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    tokenizer.model
    added_tokens.json
    chat_template.jinja
)

echo "===================================================================="
echo " overwrite_tokenizer_with_base.sh"
echo "===================================================================="
echo "  artifact: $ARTIFACT"
echo "  base:     $BASE"
echo "  dry-run:  $DRY_RUN"
echo "===================================================================="

# --- Step 1: snapshot current state for diagnosis ---
echo "[step 1/4] artifact tokenizer files BEFORE overwrite:"
for f in "${MANAGED_FILES[@]}"; do
    if [ -f "$ARTIFACT/$f" ]; then
        size=$(stat -c%s "$ARTIFACT/$f" 2>/dev/null || stat -f%z "$ARTIFACT/$f")
        printf "  %-30s %12s bytes\n" "$f" "$size"
    fi
done

echo "[step 1/4] base tokenizer files for reference:"
for f in "${MANAGED_FILES[@]}"; do
    if [ -f "$BASE/$f" ]; then
        size=$(stat -c%s "$BASE/$f" 2>/dev/null || stat -f%z "$BASE/$f")
        printf "  %-30s %12s bytes\n" "$f" "$size"
    else
        printf "  %-30s          (absent in base)\n" "$f"
    fi
done
echo

# --- Step 2: backup current artifact tokenizer files ---
if [ -d "$BACKUP_DIR" ]; then
    echo "[step 2/4] backup already exists at $BACKUP_DIR — keeping it"
    echo "           (run with --restore to revert to backup contents)"
else
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would mkdir $BACKUP_DIR and move current files there"
    else
        mkdir -p "$BACKUP_DIR"
        moved=0
        for f in "${MANAGED_FILES[@]}"; do
            if [ -f "$ARTIFACT/$f" ]; then
                mv "$ARTIFACT/$f" "$BACKUP_DIR/$f"
                moved=$((moved + 1))
            fi
        done
        echo "[step 2/4] backed up $moved tokenizer file(s) to $BACKUP_DIR"
    fi
fi

# --- Step 3: copy from base ---
copied=0
for f in "${MANAGED_FILES[@]}"; do
    if [ -f "$BASE/$f" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[dry-run]   would cp $BASE/$f -> $ARTIFACT/$f"
        else
            cp -p "$BASE/$f" "$ARTIFACT/$f"
            copied=$((copied + 1))
        fi
    fi
done
if [ "$DRY_RUN" -eq 0 ]; then
    echo "[step 3/4] copied $copied tokenizer file(s) from base"
fi

# --- Step 4: verify byte-identical with base ---
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (skipping verify step)"
    exit 0
fi

echo "[step 4/4] post-overwrite byte-diff vs base:"
all_ok=1
for f in "${MANAGED_FILES[@]}"; do
    if [ -f "$BASE/$f" ] && [ -f "$ARTIFACT/$f" ]; then
        if cmp -s "$BASE/$f" "$ARTIFACT/$f"; then
            printf "  %-30s IDENTICAL\n" "$f"
        else
            printf "  %-30s DIFFERS (unexpected after overwrite)\n" "$f"
            all_ok=0
        fi
    elif [ -f "$BASE/$f" ]; then
        printf "  %-30s MISSING in artifact\n" "$f"
        all_ok=0
    elif [ -f "$ARTIFACT/$f" ]; then
        printf "  %-30s ORPHAN in artifact (not in base; might be GPTQModel-synthesized)\n" "$f"
    fi
done

if [ "$all_ok" -eq 1 ]; then
    echo
    echo "[done] artifact tokenizer now matches base. To revert: --restore"
else
    echo
    echo "[done] artifact tokenizer overwritten but verification flagged issues above." >&2
    exit 1
fi
