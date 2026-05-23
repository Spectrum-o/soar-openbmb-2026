#!/usr/bin/env bash
# v5d (bf16_config_fix) prepare_model.sh
#
# CRITICAL FIX vs v5c: write a modified config.json that removes
# auto_map.AutoConfig. SUBMISSIONS.md hard constraint row 44:
#
#   "Removing auto_map.AutoConfig is required for SGLang's
#    MiniCPMHybridConfig to win the isinstance check"
#   (model_runner.py:1494 minicpm_hybrid_config property +
#    hybrid_linear_attn_backend.py:1456 assertion)
#
# v3 / v5 / v5c all symlinked the input dir's config.json unchanged →
# SGLang fell into trust_remote_code AutoConfig path → MiniCPMHybridConfig
# isinstance() returned False → loading failed with the transformers
# model_type list dump.
#
# 1849 worked because quantize_gptqmodel_w4a16.py:946 explicitly removes
# auto_map.AutoConfig (and quantize_config.json overrides config.json with
# the corrected version).
#
# This script mimics that single transformation, otherwise leaves the
# model byte-equivalent to baseline BF16.
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

echo "[prepare_model] BF16 config-fix: symlinking ${INPUT_DIR} -> ${OUTPUT_DIR} with config.json modified"

# OUTPUT_DIR should be a real directory (NOT a symlink), so we can write
# a fresh config.json into it while symlinking weights from INPUT_DIR.
if [ -L "${OUTPUT_DIR}" ]; then
    rm -f "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

# Symlink every file from INPUT except config.json (we'll write a fresh one)
shopt -s nullglob dotglob
for src in "${INPUT_DIR}"/*; do
    name="$(basename "${src}")"
    if [ "${name}" = "config.json" ]; then
        continue
    fi
    target="${OUTPUT_DIR}/${name}"
    if [ -e "${target}" ] || [ -L "${target}" ]; then
        rm -rf "${target}"
    fi
    if ! ln -sf "${src}" "${target}" 2>/dev/null; then
        cp -r "${src}" "${target}"
    fi
done
shopt -u nullglob dotglob

# Write a fresh config.json with auto_map.AutoConfig stripped + defensive
# read-only-property cleanup (SUBMISSIONS.md row 45 / v18 root cause).
python3 - "${INPUT_DIR}/config.json" "${OUTPUT_DIR}/config.json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

auto_map = cfg.get("auto_map", {})
removed = None
if isinstance(auto_map, dict) and "AutoConfig" in auto_map:
    removed = auto_map.pop("AutoConfig")
    if not auto_map:
        cfg.pop("auto_map", None)
    else:
        cfg["auto_map"] = auto_map

# Defensively strip derived read-only properties that crash AutoConfig
# under MiniCPMHybridConfig (v18 root cause, SUBMISSIONS.md row 45)
for k in (
    "has_sparse_attention",
    "has_lightning_layers",
    "full_attention_layer_ids",
    "sparse_layer_ids",
    "lightning_layer_ids",
    "mamba2_cache_params",
):
    cfg.pop(k, None)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(f"[prepare_model] config.json written; removed auto_map.AutoConfig={removed!r}", flush=True)
PY

echo "[prepare_model] done"
