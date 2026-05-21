#!/usr/bin/env python3
"""Compare tokenizer files between a base BF16 model dir and a quant artifact.

Stand-alone CPU-only diff designed to catch the H4 (2026-05-22) failure
mode: GPTQModel's `model.save()` re-serializes the tokenizer via newer
HF tokenizers library, producing:

  - tokenizer.json grows from ~3.6MB to ~6.7MB (more verbose schema)
  - tokenizer_config.json keeps an inline `chat_template` field
  - chat_template.jinja appears as a separate sidecar file (new format)
  - added_tokens.json appears containing special-token additions

transformers >= 4.45 reads chat_template from either the inline field OR
the .jinja sidecar. Older transformers reads ONLY the inline field, so
the same artifact loads with two different chat_template states across
versions. If the inline and sidecar diverge AT ALL, that drift surfaces
as garbage output on the older-transformers platform while the newer-
transformers AutoDL session looks fine.

This tool prints:
  - Per-file size and existence diff between base and artifact
  - chat_template duplication risk: inline-in-config vs sidecar-jinja
  - Whether the two chat_template sources actually agree byte-for-byte
  - Files present only in one side
  - Vocab size cross-check (added_tokens.json affects this)

Usage:
    python3 tools/check_tokenizer_compat.py \\
        --base /root/autodl-fs/models/OpenBMB/MiniCPM-SALA \\
        --artifact /root/autodl-fs/zyn/models/submission_gptqmodel_calib_w4a16-quantized

    # Optional: write a markdown report
    python3 tools/check_tokenizer_compat.py --base ... --artifact ... \\
        --output-md /tmp/tokenizer_diff.md

Exit codes:
    0   no drift detected (artifact tokenizer matches base byte-for-byte
        on every managed file)
    1   drift detected (mismatch in any file OR unexpected sidecar present)
    2   CLI / IO error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


MANAGED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
)


@dataclass
class FileStatus:
    name: str
    base_size: int | None
    artifact_size: int | None
    base_md5: str | None
    artifact_md5: str | None

    @property
    def identical(self) -> bool:
        return (
            self.base_md5 is not None
            and self.artifact_md5 is not None
            and self.base_md5 == self.artifact_md5
        )

    @property
    def base_only(self) -> bool:
        return self.base_md5 is not None and self.artifact_md5 is None

    @property
    def artifact_only(self) -> bool:
        return self.base_md5 is None and self.artifact_md5 is not None

    @property
    def differs(self) -> bool:
        return (
            self.base_md5 is not None
            and self.artifact_md5 is not None
            and self.base_md5 != self.artifact_md5
        )


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(base: Path, artifact: Path) -> list[FileStatus]:
    out = []
    for name in MANAGED_FILES:
        bp = base / name
        ap = artifact / name
        out.append(FileStatus(
            name=name,
            base_size=bp.stat().st_size if bp.is_file() else None,
            artifact_size=ap.stat().st_size if ap.is_file() else None,
            base_md5=md5_file(bp) if bp.is_file() else None,
            artifact_md5=md5_file(ap) if ap.is_file() else None,
        ))
    return out


def inspect_chat_template_inline(p: Path) -> str | None:
    """Return the chat_template string from tokenizer_config.json, or None.

    Some serializers store it as a string, some as a list of dicts; we
    canonicalize to a single string (joining list entries) for comparison.
    """
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    tpl = cfg.get("chat_template")
    if tpl is None:
        return None
    if isinstance(tpl, str):
        return tpl
    if isinstance(tpl, list):
        # HF historical form: list of {"name": ..., "template": ...}
        parts = []
        for entry in tpl:
            if isinstance(entry, dict) and "template" in entry:
                parts.append(str(entry["template"]))
        return "\n---\n".join(parts) if parts else None
    return None


def inspect_chat_template_sidecar(dir_path: Path) -> str | None:
    p = dir_path / "chat_template.jinja"
    if not p.is_file():
        return None
    try:
        return p.read_text()
    except OSError:
        return None


def vocab_size_from_tokenizer_json(p: Path) -> int | None:
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    model = data.get("model", {})
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)
    added = data.get("added_tokens")
    if isinstance(added, list) and isinstance(vocab, dict):
        return len(vocab) + len(added)
    return None


def render_report(base: Path, artifact: Path, files: list[FileStatus]) -> tuple[str, bool]:
    """Build a markdown report. Return (text, drift_detected)."""
    lines: list[str] = []
    lines.append(f"# Tokenizer compat: `{artifact}` vs `{base}`")
    lines.append("")
    lines.append("## File-level diff")
    lines.append("")
    lines.append("| File | Base size | Artifact size | Verdict |")
    lines.append("|---|---|---|---|")
    drift = False
    for f in files:
        bs = f"{f.base_size:,}" if f.base_size is not None else "(absent)"
        as_ = f"{f.artifact_size:,}" if f.artifact_size is not None else "(absent)"
        if f.identical:
            verdict = "IDENTICAL"
        elif f.differs:
            verdict = f"**DIFFERS** (md5 base={f.base_md5[:8]} artifact={f.artifact_md5[:8]})"
            drift = True
        elif f.base_only:
            verdict = "**only in base** (artifact is missing this; will fall back to defaults at load)"
            drift = True
        elif f.artifact_only:
            verdict = "**only in artifact** (synthesized by GPTQModel.save; suspect)"
            drift = True
        else:
            verdict = "neither side has it"
        lines.append(f"| `{f.name}` | {bs} | {as_} | {verdict} |")
    lines.append("")

    # Chat-template inline vs sidecar
    lines.append("## Chat-template duplication check")
    lines.append("")
    base_inline = inspect_chat_template_inline(base / "tokenizer_config.json")
    base_sidecar = inspect_chat_template_sidecar(base)
    art_inline = inspect_chat_template_inline(artifact / "tokenizer_config.json")
    art_sidecar = inspect_chat_template_sidecar(artifact)

    def _presence(s: str | None) -> str:
        if s is None:
            return "(absent)"
        return f"{len(s)} chars"

    lines.append("| Location | Base | Artifact |")
    lines.append("|---|---|---|")
    lines.append(f"| `tokenizer_config.json` inline `chat_template` | {_presence(base_inline)} | {_presence(art_inline)} |")
    lines.append(f"| `chat_template.jinja` sidecar | {_presence(base_sidecar)} | {_presence(art_sidecar)} |")
    lines.append("")

    if art_inline is not None and art_sidecar is not None:
        if art_inline == art_sidecar:
            lines.append("- Artifact has BOTH inline and sidecar with IDENTICAL content. transformers>=4.45 reads either consistently. transformers<4.45 reads only inline (which is present). **Likely safe across versions.**")
        else:
            lines.append("- **WARNING: Artifact has BOTH inline and sidecar but they DISAGREE.** transformers>=4.45 vs <4.45 will load different chat_templates. This is the H4 failure mechanism. Overwrite with base tokenizer to fix.")
            drift = True
    elif art_inline is None and art_sidecar is not None:
        lines.append("- **WARNING: Artifact has chat_template.jinja sidecar but NO inline in tokenizer_config.json.** transformers<4.45 will see an empty template -> prompts lack <|im_start|> structure -> model output is garbage. **H4 failure mechanism confirmed.**")
        drift = True
    elif art_inline is not None and art_sidecar is None:
        lines.append("- Artifact has inline chat_template only (no sidecar). Both old and new transformers read the same value. **Safe.**")
    else:
        lines.append("- Artifact has NEITHER inline nor sidecar chat_template. apply_chat_template will fail or fall back to default. If the eval harness calls apply_chat_template, output will degrade.")
        drift = True
    lines.append("")

    # Vocab size cross-check
    lines.append("## Vocab size cross-check")
    lines.append("")
    base_vocab = vocab_size_from_tokenizer_json(base / "tokenizer.json")
    art_vocab = vocab_size_from_tokenizer_json(artifact / "tokenizer.json")
    lines.append(f"- base `tokenizer.json` vocab entries: {base_vocab}")
    lines.append(f"- artifact `tokenizer.json` vocab entries: {art_vocab}")
    if base_vocab is not None and art_vocab is not None and base_vocab != art_vocab:
        lines.append(f"- **WARNING: vocab size diverged by {art_vocab - base_vocab} entries.** Token IDs may not map back to model embedding rows -> certain prompts will encode to out-of-range IDs.")
        drift = True
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if drift:
        lines.append("**DRIFT DETECTED.** Recommend running:")
        lines.append("")
        lines.append(f"    bash scripts/overwrite_tokenizer_with_base.sh \\")
        lines.append(f"        --artifact {artifact} \\")
        lines.append(f"        --base {base}")
        lines.append("")
        lines.append("Then re-eval to confirm the platform-=0 mechanism is resolved.")
    else:
        lines.append("No drift detected. Artifact tokenizer matches base byte-for-byte.")

    return "\n".join(lines), drift


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base", required=True,
                    help="Base BF16 model dir (the ground-truth tokenizer source)")
    ap.add_argument("--artifact", required=True,
                    help="Quantized model dir to check")
    ap.add_argument("--output-md", default=None,
                    help="Write markdown report here")
    args = ap.parse_args()

    base = Path(args.base)
    artifact = Path(args.artifact)
    if not base.is_dir():
        print(f"--base dir not found: {base}", file=sys.stderr)
        return 2
    if not artifact.is_dir():
        print(f"--artifact dir not found: {artifact}", file=sys.stderr)
        return 2

    files = scan(base, artifact)
    report, drift = render_report(base, artifact, files)
    print(report)
    if args.output_md:
        Path(args.output_md).write_text(report)
        print(f"\n[check_tokenizer_compat] wrote {args.output_md}", file=sys.stderr)
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
