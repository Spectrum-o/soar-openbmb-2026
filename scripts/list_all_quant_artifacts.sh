#!/usr/bin/env bash
# scripts/list_all_quant_artifacts.sh
#
# Inventory all quantization artifacts under /root/autodl-fs/zyn/models/
# (and a few common alternates). For each artifact:
#   - size + mtime
#   - key config.json fields (model_type, dtype)
#   - quantize_config.json highlights (bits, group_size, sym, desc_act)
#   - presence/count of .safetensors shards
#   - qzeros 0x88888888 sanity check (samples one .qzeros tensor, requires
#     `safetensors` Python pkg; skipped silently if unavailable)
#
# Usage:
#   bash scripts/list_all_quant_artifacts.sh
#   bash scripts/list_all_quant_artifacts.sh /custom/dir   # override root
#
# Read-only; safe to run any time. Designed for the AutoDL server.

set -u

ROOT="${1:-/root/autodl-fs/zyn/models}"

if [ ! -d "$ROOT" ]; then
    echo "[inv] root does not exist: $ROOT" >&2
    exit 1
fi

echo "[inv] scanning $ROOT (depth 2)"
echo

# Discover candidate dirs: anything containing config.json
mapfile -t DIRS < <(find "$ROOT" -mindepth 1 -maxdepth 2 -type d \
    -exec test -f '{}/config.json' \; -print 2>/dev/null | sort)

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "[inv] no model dirs (with config.json) found under $ROOT"
    exit 0
fi

print_field() {
    # $1: file, $2: jq path
    local file="$1"
    local path="$2"
    if [ -f "$file" ]; then
        if command -v jq >/dev/null 2>&1; then
            jq -r "$path // \"-\"" "$file" 2>/dev/null
        else
            python3 -c "import json,sys; d=json.load(open('$file')); print(d.get($path, '-'))" 2>/dev/null \
                || echo "-"
        fi
    else
        echo "-"
    fi
}

for d in "${DIRS[@]}"; do
    size="$(du -sh "$d" 2>/dev/null | cut -f1)"
    mtime="$(stat -c '%y' "$d" 2>/dev/null | cut -d. -f1)"
    cfg="$d/config.json"
    qcfg="$d/quantize_config.json"
    n_shards="$(find "$d" -maxdepth 1 -name '*.safetensors' | wc -l)"
    bits="$(print_field "$qcfg" .bits)"
    gs="$(print_field "$qcfg" .group_size)"
    sym="$(print_field "$qcfg" .sym)"
    desc="$(print_field "$qcfg" .desc_act)"
    qm="$(print_field "$qcfg" .quant_method)"
    mtype="$(print_field "$cfg" .model_type)"

    echo "--- $d"
    echo "    size: $size   mtime: $mtime"
    echo "    config.json: model_type=$mtype"
    echo "    quantize_config.json: quant_method=$qm bits=$bits group_size=$gs sym=$sym desc_act=$desc"
    echo "    safetensors shards: $n_shards"
done

# Optional: qzeros sanity check on the first shard of each artifact.
# Slow-ish (mmap one tensor per shard); skip with NO_QZEROS_CHECK=1.
if [ "${NO_QZEROS_CHECK:-0}" = "1" ]; then
    echo
    echo "[inv] qzeros check skipped (NO_QZEROS_CHECK=1)"
    exit 0
fi

if ! python3 -c "import safetensors" 2>/dev/null; then
    echo
    echo "[inv] qzeros check skipped (safetensors not installed)"
    exit 0
fi

echo
echo "[inv] qzeros 0x88888888 check (sample one .qzeros per artifact):"
for d in "${DIRS[@]}"; do
    python3 - "$d" <<'PY'
import sys, glob, os
try:
    from safetensors import safe_open
except ImportError:
    sys.exit(0)
d = sys.argv[1]
shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
if not shards:
    print(f"  {d}  no shards")
    sys.exit(0)
shard = shards[0]
try:
    with safe_open(shard, framework="pt") as f:
        for name in f.keys():
            if name.endswith(".qzeros"):
                t = f.get_tensor(name)
                u = t.unique()
                # 0x88888888 as signed int32 == -2004318072
                # 0x77777777 as signed int32 ==  2004318071
                vals = u.tolist()[:4]
                ok = all(v == -2004318072 for v in vals)
                bug = any(v == 2004318071 for v in vals)
                tag = "OK (0x88...)" if ok else ("BUG (0x77...)" if bug else f"mixed {vals}")
                print(f"  {d}  shard={os.path.basename(shard)} qzeros={tag}")
                break
        else:
            print(f"  {d}  no .qzeros tensor in {os.path.basename(shard)}")
except Exception as exc:
    print(f"  {d}  qzeros check error: {exc}")
PY
done
