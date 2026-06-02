#!/usr/bin/env bash
# scripts/postq_reload_smoke.sh
#
# Isolated GPU smoke for the POSTQ "second reload" (prepare_model.sh step-3),
# the step that has crashed on every POSTQ submission so far. It runs ONLY the
# `--post-quant-kv-only` reload on an ALREADY-PREPARED artifact, so you catch
# bugs #6 / #7 / #8 in ~minutes instead of a ~54-min full prepare or a platform
# slot.
#
# Background (see experiments/POSTQ_FAILURE_INVENTORY_20260602.md):
#   The POSTQ reload runs as a 2nd process that re-loads the qzeros-fixed +
#   overlay-mutated W4A16 artifact via gptqmodel's GPTQModel.load, then forwards
#   to measure fp8 KV scales. Failure gauntlet observed/predicted:
#     #6 config-class (FIXED 2026-06-02, audit-confirmed)  -> ValueError config_class
#     #7 split/fused skip-name mismatch (FIX in RELOAD7FIX) -> q/k/v missing .qweight
#     #8 HF-modeling forward needs InfLLM-v2 sparse kernels -> forward fails / ran_forwards==0
#   This script tells you EXACTLY which stage you reach.
#
# REQUIREMENTS: a GPU box with the SOAR env (transformers 4.57.1, gptqmodel
# 7.0.0) activated, and an artifact that already went through prepare step-1
# (GPTQ quantize + qzeros fix) AND step-2 (selective-BF16 overlay). Any leftover
# OUTPUT dir from a prior POSTQ run that crashed in step-3 works — step-1/2 had
# already finished. If you have none, run prepare_model.sh once to the point it
# enters step-3 (it will produce the artifact), then re-run this repeatedly.
#
# USAGE:
#   bash scripts/postq_reload_smoke.sh \
#       --artifact /root/autodl-tmp/.../prepared_w4a16_overlay_artifact \
#       --input    /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \
#       --quantizer submission_gptqmodel_full_w4a16_selective_bf16_last7attn_rmsopfusion_calib300_damp01_fp8kv_hp224_postq_multicalib8k_diag_offload/quantize_gptqmodel_w4a16.py
#
set -uo pipefail

ARTIFACT="" ; INPUT="" ; QUANTIZER=""
NUM_SAMPLES="${KV_CALIB_NUM_SAMPLES:-4}"   # tiny by default — we want fast pass/fail, not good scales
MAX_LEN="${KV_CALIB_MAX_LEN:-2048}"
SAFE_MAX="${KV_CALIB_SAFE_MAX:-224}"
CALIB_JSONL="${CALIB_JSONL:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --quantizer) QUANTIZER="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --calib-jsonl) CALIB_JSONL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -z "$ARTIFACT" ] || [ -z "$INPUT" ] || [ -z "$QUANTIZER" ] && {
  echo "usage: $0 --artifact <prepared_dir> --input <bf16_dir> --quantizer <path/to/quantize_gptqmodel_w4a16.py>" >&2
  exit 2
}
[ -d "$ARTIFACT" ] || { echo "FATAL: artifact dir not found: $ARTIFACT" >&2; exit 2; }
[ -f "$QUANTIZER" ] || { echo "FATAL: quantizer not found: $QUANTIZER" >&2; exit 2; }
[ -z "$CALIB_JSONL" ] && CALIB_JSONL="$(dirname "$QUANTIZER")/perf_public_set.jsonl"

echo "=============================================================="
echo "POSTQ reload smoke"
echo "  artifact : $ARTIFACT"
echo "  input    : $INPUT"
echo "  quantizer: $QUANTIZER"
echo "  calib    : $CALIB_JSONL  (num_samples=$NUM_SAMPLES max_len=$MAX_LEN safe_max=$SAFE_MAX)"
echo "=============================================================="
echo "[smoke] preflight greps on the artifact (no GPU needed):"
python3 - "$ARTIFACT" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
qc = d / "quantize_config.json"
if qc.is_file():
    dyn = (json.loads(qc.read_text()).get("dynamic") or {})
    hf = [k for k in dyn if k.endswith((".q_proj$", ".k_proj$", ".v_proj$"))]
    fused = [k for k in dyn if k.endswith(".qkv_proj$")]
    print(f"  quantize_config.dynamic: fused qkv_proj rules={len(fused)}  HF q/k/v alias rules={len(hf)}")
    print(f"  -> bug #7 guard {'PRESENT (RELOAD7FIX)' if hf else 'ABSENT (will likely hit #7 on q/k/v)'}")
else:
    print("  WARN: no quantize_config.json in artifact")
PY
echo "--------------------------------------------------------------"
echo "[smoke] running isolated POSTQ reload (this is prepare step-3 only)…"
echo "        watch for: [postq-kv] loading… -> GPTQModel.load -> [N/N] done"

set -x
python3 "$QUANTIZER" \
    --post-quant-kv-only \
    --input "$INPUT" \
    --output "$ARTIFACT" \
    --calib-jsonl "$CALIB_JSONL" \
    --post-quant-kv-num-samples "$NUM_SAMPLES" \
    --post-quant-kv-max-len "$MAX_LEN" \
    --post-quant-kv-truncation-side left \
    --post-quant-kv-safe-max "$SAFE_MAX" \
    --post-quant-kv-output "fp8kv_scales_SMOKE.safetensors" \
    --post-quant-kv-report "fp8kv_scales_SMOKE_report.json"
rc=$?
set +x

echo "=============================================================="
if [ $rc -eq 0 ]; then
  echo "[smoke] RESULT: reload + forward + scale-write SUCCEEDED (rc=0)."
  echo "        -> #6/#7/#8 all cleared. POSTQ prepare step-3 is viable."
  echo "        Next risk is SERVE-TIME quality only (POSTQ measure-source HF-marlin"
  echo "        != serve-source sglang-marlin; treat as an A/B vs DENSEQKV 78.67)."
else
  echo "[smoke] RESULT: FAILED (rc=$rc). Match the traceback to the gauntlet:"
  echo "        - 'config_class ... not consistent'            -> #6 (config-class) — unexpected; RELOAD_CONFIG_FIX should cover it"
  echo "        - KeyError / missing .qweight on q_proj/k_proj/v_proj of a lightning layer -> #7 (split/fused skip) — needs RELOAD7FIX"
  echo "        - 'ran zero successful forwards' / sparse-kernel / flash_attn import     -> #8 (HF-modeling forward env)"
  echo "        See experiments/POSTQ_FAILURE_INVENTORY_20260602.md for each."
fi
echo "=============================================================="
exit $rc
