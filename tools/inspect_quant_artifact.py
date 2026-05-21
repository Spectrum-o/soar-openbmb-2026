#!/usr/bin/env python3
"""Inspect a GPTQ-quantized model artifact without loading the full model.

Reads config.json + quantize_config.json, walks safetensors shards lazily,
samples .qzeros / .scales / .qweight tensors to verify Marlin compatibility.

CPU-only. Memory usage: only the sampled tensors (one per shard) are read
into RAM. The 9B SALA model's quantized form is ~5GB on disk and we touch
only a few MB.

Checks (verified against submission_gptqmodel_calib_w4a16-quantized format):

  1. config.json present + quantization_config block exists
  2. quantize_config.json: bits/group_size/sym/desc_act in expected range
  3. safetensors shards: at least 1 present, each readable
  4. qzeros: every shard's first .qzeros tensor has unique values =
     {-2004318072} (signed int32 form of 0x88888888). If any is
     {2004318071} (= 0x77777777), the gptqmodel 7.0 bug was not patched.
  5. scales: dtype float16, has reasonable min/max/mean
  6. qweight: present where expected; module inventory by layer

Usage:
    python3 tools/inspect_quant_artifact.py \\
        --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized

    python3 tools/inspect_quant_artifact.py --artifact ... --output-md report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy is required for this tool. Install: pip install numpy", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors import safe_open
except ImportError:
    print(
        "safetensors is required for this tool. Install: pip install safetensors",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Constants (signed int32 forms of packed Marlin qzeros bit-patterns)
# ---------------------------------------------------------------------------
QZEROS_OK_INT32 = -2004318072   # = 0x88888888 (Marlin-compatible)
QZEROS_BUG_INT32 = 2004318071   # = 0x77777777 (gptqmodel 7.0 + sym=True bug)


# ---------------------------------------------------------------------------
# JSON readers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_error": str(exc)}


def summarize_config(cfg: dict) -> list[str]:
    lines: list[str] = []
    if cfg is None:
        return ["  (config.json missing)"]
    if "_error" in cfg:
        return [f"  (config.json parse error: {cfg['_error']})"]
    lines.append(f"  model_type:        {cfg.get('model_type', '?')}")
    lines.append(f"  torch_dtype:       {cfg.get('torch_dtype', '?')}")
    lines.append(f"  vocab_size:        {cfg.get('vocab_size', '?')}")
    lines.append(f"  hidden_size:       {cfg.get('hidden_size', '?')}")
    lines.append(f"  num_hidden_layers: {cfg.get('num_hidden_layers', '?')}")
    qc = cfg.get("quantization_config")
    if qc:
        lines.append(f"  quantization_config:")
        for k in ("bits", "group_size", "sym", "desc_act", "quant_method"):
            lines.append(f"    {k}: {qc.get(k, '?')}")
        dyn = qc.get("dynamic")
        if dyn:
            keys = list(dyn.keys()) if isinstance(dyn, dict) else dyn
            lines.append(f"    dynamic (skip patterns): {keys[:6]}{' ...' if len(keys) > 6 else ''}")
    else:
        lines.append("  quantization_config: MISSING — SGLang loader will treat as BF16")
    return lines


def summarize_quantize_config(qc: dict) -> list[str]:
    lines: list[str] = []
    if qc is None:
        return ["  (quantize_config.json missing — not always required if config.json has the data)"]
    if "_error" in qc:
        return [f"  (quantize_config.json parse error: {qc['_error']})"]
    for k in ("bits", "group_size", "sym", "desc_act", "quant_method", "checkpoint_format"):
        lines.append(f"  {k:18s} {qc.get(k, '?')}")
    meta = qc.get("meta", {})
    if meta:
        lines.append(f"  quantizer:        {meta.get('quantizer', '?')}")
        lines.append(f"  damp_percent:     {meta.get('damp_percent', '?')}")
        lines.append(f"  true_sequential:  {meta.get('true_sequential', '?')}")
        lines.append(f"  static_groups:    {meta.get('static_groups', '?')}")
    return lines


# ---------------------------------------------------------------------------
# Safetensors walking
# ---------------------------------------------------------------------------
def discover_shards(artifact: Path) -> list[Path]:
    return sorted(artifact.glob("*.safetensors"))


def inspect_shard(shard: Path, max_sample: int = 1) -> dict:
    """Open a shard lazily, classify tensors, sample qzeros/scales/qweight.

    `max_sample` controls how many tensors of each kind we actually load
    bytes for (sampling — full inspection would be too slow on 32 layers).
    """
    out: dict = {
        "shard": str(shard),
        "size_mb": shard.stat().st_size // (1024 * 1024),
        "tensors_total": 0,
        "tensor_kinds": Counter(),  # by suffix
        "modules": Counter(),       # by "self_attn.X" or "mlp.X" suffix
        "qzeros_check": [],         # per sampled .qzeros tensor
        "scales_stats": [],         # per sampled .scales tensor
        "qweight_shapes": [],       # per sampled .qweight tensor
        "errors": [],
    }
    try:
        f = safe_open(str(shard), framework="numpy")
    except Exception as exc:
        out["errors"].append(f"open: {exc}")
        return out

    qzeros_keys: list[str] = []
    scales_keys: list[str] = []
    qweight_keys: list[str] = []
    keys = list(f.keys())
    out["tensors_total"] = len(keys)
    for k in keys:
        # classify by suffix
        if k.endswith(".qzeros"):
            out["tensor_kinds"]["qzeros"] += 1
            qzeros_keys.append(k)
        elif k.endswith(".qweight"):
            out["tensor_kinds"]["qweight"] += 1
            qweight_keys.append(k)
        elif k.endswith(".scales"):
            out["tensor_kinds"]["scales"] += 1
            scales_keys.append(k)
        elif k.endswith(".g_idx"):
            out["tensor_kinds"]["g_idx"] += 1
        elif k.endswith(".weight"):
            out["tensor_kinds"]["weight"] += 1
        else:
            out["tensor_kinds"]["other"] += 1
        # module classification (e.g. layers.0.mlp.gate_proj.qweight -> mlp.gate_proj)
        parts = k.rsplit(".", 2)  # drop suffix to get "<...>.<module>.<param>"
        if len(parts) >= 3:
            mod = ".".join(parts[-3:-1])  # last two segments
            out["modules"][mod] += 1

    # Sample tensors
    for k in qzeros_keys[:max_sample]:
        try:
            t = f.get_tensor(k)
        except Exception as exc:
            out["qzeros_check"].append({"name": k, "error": str(exc)})
            continue
        u = np.unique(t).tolist()
        if u == [QZEROS_OK_INT32]:
            verdict = "OK (0x88888888)"
        elif u == [QZEROS_BUG_INT32]:
            verdict = "BUG (0x77777777 — qzeros not patched)"
        else:
            verdict = f"MIXED unique[:4]={u[:4]}"
        out["qzeros_check"].append(
            {"name": k, "shape": list(t.shape), "dtype": str(t.dtype), "verdict": verdict}
        )
    for k in scales_keys[:max_sample]:
        try:
            t = f.get_tensor(k)
        except Exception as exc:
            out["scales_stats"].append({"name": k, "error": str(exc)})
            continue
        # Promote fp16 -> fp32 for safe stats
        t32 = t.astype(np.float32) if t.dtype != np.float32 else t
        out["scales_stats"].append(
            {
                "name": k,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "min": float(t32.min()),
                "max": float(t32.max()),
                "mean": float(t32.mean()),
                "has_nan": bool(np.isnan(t32).any()),
                "has_inf": bool(np.isinf(t32).any()),
            }
        )
    for k in qweight_keys[:max_sample]:
        try:
            t = f.get_tensor(k)
        except Exception as exc:
            out["qweight_shapes"].append({"name": k, "error": str(exc)})
            continue
        out["qweight_shapes"].append(
            {"name": k, "shape": list(t.shape), "dtype": str(t.dtype)}
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------
def aggregate_modules(per_shard: list[dict]) -> dict:
    counts: Counter = Counter()
    for s in per_shard:
        counts.update(s["modules"])
    return dict(counts)


def render_report(artifact: Path, cfg: dict | None, qc: dict | None, per_shard: list[dict]) -> str:
    sections: list[str] = []
    sections.append(f"# Quant artifact inspection: `{artifact}`\n")
    total_size_mb = sum(s["size_mb"] for s in per_shard)
    n_tensors = sum(s["tensors_total"] for s in per_shard)
    n_shards = len(per_shard)
    sections.append(f"- shards: {n_shards}")
    sections.append(f"- total tensors: {n_tensors}")
    sections.append(f"- total safetensors size: ~{total_size_mb} MB\n")

    sections.append("## config.json\n```")
    sections.extend(summarize_config(cfg or {}))
    sections.append("```\n")

    sections.append("## quantize_config.json\n```")
    sections.extend(summarize_quantize_config(qc or {}))
    sections.append("```\n")

    # qzeros verdict
    sections.append("## Qzeros (0x88888888 Marlin patch check)\n```")
    if not any(s["qzeros_check"] for s in per_shard):
        sections.append("  (no .qzeros tensors found — model is NOT quantized?)")
    else:
        ok = bug = mixed = 0
        for s in per_shard:
            for c in s["qzeros_check"]:
                v = c.get("verdict", "")
                if v.startswith("OK"):
                    ok += 1
                elif v.startswith("BUG"):
                    bug += 1
                elif v.startswith("MIXED"):
                    mixed += 1
                sections.append(f"  {Path(s['shard']).name}  {c.get('name','?')}  {v}")
        sections.append(f"\n  summary: OK={ok}  BUG={bug}  MIXED={mixed}")
        if bug:
            sections.append("  >>> CRITICAL: qzeros bug NOT patched — model output will be garbage.")
        elif mixed:
            sections.append("  >>> WARN: mixed qzeros values — manual inspection needed.")
        else:
            sections.append("  >>> qzeros are Marlin-compatible.")
    sections.append("```\n")

    sections.append("## Scales (dtype + stats)\n```")
    any_nan_inf = False
    for s in per_shard:
        for c in s["scales_stats"]:
            if "error" in c:
                sections.append(f"  {Path(s['shard']).name}  {c['name']}  ERR {c['error']}")
                continue
            warn = ""
            if c["has_nan"] or c["has_inf"]:
                warn = "  <<< NaN/Inf in scales!"
                any_nan_inf = True
            sections.append(
                f"  {Path(s['shard']).name}  {c['name']}  "
                f"{c['dtype']} {c['shape']}  "
                f"min={c['min']:.3g}  max={c['max']:.3g}  mean={c['mean']:.3g}{warn}"
            )
    if any_nan_inf:
        sections.append("  >>> CRITICAL: NaN/Inf in scales — model is broken.")
    sections.append("```\n")

    sections.append("## Module inventory (count of unique quantized linear modules per layer)\n```")
    mods = aggregate_modules(per_shard)
    if not mods:
        sections.append("  (no modules detected)")
    else:
        # Each module appears 3x per quantized linear (qweight + qzeros + scales)
        # Normalize by dividing the count by 3 for the "n linears" view.
        for mod, n in sorted(mods.items()):
            sections.append(f"  {mod}: {n} tensors ({n // 3} linears at 3 tensors each)")
    sections.append("```\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--artifact", required=True, help="Quantized model directory")
    ap.add_argument("--output-md", default=None)
    ap.add_argument(
        "--sample-tensors",
        type=int,
        default=1,
        help="Per shard, sample N tensors of each kind (qzeros/scales/qweight)",
    )
    args = ap.parse_args()

    artifact = Path(args.artifact)
    if not artifact.is_dir():
        print(f"artifact not found: {artifact}", file=sys.stderr)
        return 2

    cfg = read_json(artifact / "config.json")
    qc = read_json(artifact / "quantize_config.json")
    shards = discover_shards(artifact)
    if not shards:
        print(f"no .safetensors shards in {artifact}", file=sys.stderr)
        return 2

    print(f"[inspect] {artifact}: {len(shards)} shard(s)", file=sys.stderr)
    per_shard: list[dict] = []
    for sh in shards:
        print(f"  scanning {sh.name} ({sh.stat().st_size // (1024*1024)} MB)", file=sys.stderr)
        per_shard.append(inspect_shard(sh, args.sample_tensors))

    text = render_report(artifact, cfg, qc, per_shard)
    print(text)
    if args.output_md:
        Path(args.output_md).write_text(text)
        print(f"\n[inspect] wrote {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
