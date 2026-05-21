#!/usr/bin/env python3
"""Pack a SOAR submission tarball from a variant directory.

Wraps the staging→cp -L→drop pycache→tar→verify→md5 pattern we used
manually for v18-v22. Symlinks in the variant dir (typically `sglang/`
and the bundled `flash_attn` wheel in v17_minconfig) are dereferenced
so the tarball is self-contained on the platform.

Verifications performed:
    1. tar -t pass (tarball reads back)
    2. top-level entries listed
    3. quantize_gptqmodel_w4a16.py inside tarball has both critical fixes:
       fix_qzeros_for_marlin() and truncation_side="left"
    4. SGLANG_SERVER_ARGS is exported in prepare_env.sh
    5. final md5 + size printed

Usage:
    # pack from current repo root (default)
    python3 tools/pack_submission.py \\
        --variant submission_gptqmodel_calib_w4a16 \\
        --output soar_v23_mlp_only_multi_window.tar.gz

    # check-only: don't write tarball, just validate the variant dir
    python3 tools/pack_submission.py \\
        --variant submission_gptq_v17_minconfig \\
        --check-only

The tool refuses to pack if the variant doesn't contain
`quantize_gptqmodel_w4a16.py` with both fixes, unless --force is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def grep_file(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(errors="replace")
    except Exception:
        return False


def validate_variant(variant_dir: Path) -> list[str]:
    """Return a list of validation problems. Empty list means OK.

    Mode is auto-detected:
        - If quantize_gptqmodel_w4a16.py exists → GPTQ mode (must have
          qzeros fix + truncation_side="left" + hardened-fix markers +
          H4 tokenizer-overwrite fix)
        - Else → BF16 / identity mode (only prepare_env/prepare_model required)

    Per-check rationale + commit references:
        - fix_qzeros_for_marlin presence: gptqmodel 7.0 qzeros=7 bug
          (see SUBMISSIONS.md "Hard constraints / GPTQModel-specific")
        - POST-CHECK marker: hardened fix (commit d8aaaf1e4) catches
          silent-layout-drift on the platform; the old fix could no-op
        - copy_runtime_assets guard removal: H4 (commit 57f9cef06)
          ensures base BF16 tokenizer wins over GPTQModel re-serialized
          version; required for transformers<4.45 platforms
        - gptqmodel pin: H2 (commit d8aaaf1e4) prevents the loose
          `>=7.0,<8.0` install gate from accepting a different .x
        - DIAGNOSTIC blocks: commit d8aaaf1e4 surface what the platform
          quantization actually does so failures are debuggable remotely
        - chunked-prefill 8K isolation: 2026-05-22 v23 isolates the
          quant pipeline fix from the +83% perf tweak so v23 platform=0
          → fix attribution is unambiguous
    """
    problems: list[str] = []

    quant_script = variant_dir / "quantize_gptqmodel_w4a16.py"
    if quant_script.is_file():
        # ---- GPTQ path: validate all known critical fixes ----
        if not grep_file(quant_script, "fix_qzeros_for_marlin"):
            problems.append("quantize script: no fix_qzeros_for_marlin found (qzeros bug not patched)")
        if not grep_file(quant_script, "POST-CHECK"):
            problems.append("quantize script: fix_qzeros_for_marlin not hardened (no POST-CHECK marker); vulnerable to silent layout drift on platform — see commit d8aaaf1e4")
        if not grep_file(quant_script, 'truncation_side = "left"'):
            problems.append('quantize script: truncation_side="left" not set')
        # H4 (commit 57f9cef06): the copy_runtime_assets fix removed the
        # `if not target.exists()` guard and added a docstring canary
        # `CRITICAL 2026-05-22: this function now OVERWRITES`. We check
        # for the canary because the buggy literal string still appears
        # in the docstring of the FIXED version (it's quoted as part of
        # the explanation), so a naive grep for the guard string gives
        # a false positive.
        if not grep_file(quant_script, "CRITICAL 2026-05-22: this function now OVERWRITES"):
            problems.append(
                "quantize script: copy_runtime_assets fix canary missing — "
                "tokenizer files in the artifact may be GPTQModel's re-serialized "
                "(6.7MB tokenizer.json + sidecar chat_template.jinja). On platforms with "
                "transformers<4.45 the chat_template will load empty -> garbage output. "
                "Fix: remove the `if not target.exists()` guard so copy unconditionally "
                "overwrites GPTQModel's writes with base BF16 originals, and add the "
                "documented `CRITICAL 2026-05-22: this function now OVERWRITES` marker "
                "in the docstring. See commit 57f9cef06 / H4 in "
                "experiments/PLATFORM_DEBUG_HANDOFF.md."
            )
    # else: BF16 path; no quantize script expected.

    prepare_env = variant_dir / "prepare_env.sh"
    if not prepare_env.is_file():
        problems.append("missing prepare_env.sh")
    else:
        if not grep_file(prepare_env, "SGLANG_SERVER_ARGS"):
            problems.append("prepare_env.sh: SGLANG_SERVER_ARGS not exported")
        # H2: gptqmodel pin (only when quantize script also present)
        if quant_script.is_file():
            if not grep_file(prepare_env, "GPTQMODEL_PIN") and \
               not grep_file(prepare_env, "gptqmodel==7.0.0"):
                problems.append(
                    "prepare_env.sh: gptqmodel not pinned to ==7.0.0 — platform may use a different 7.x .x release "
                    "than AutoDL local. See commit d8aaaf1e4 / H2."
                )
        # Variable isolation: chunked-prefill 65536 confounds v23 acc
        # attribution. Should stay at 8192 in submission tarballs until
        # v23 confirms the quant pipeline fixes work.
        if grep_file(prepare_env, "chunked-prefill-size 65536"):
            problems.append(
                "prepare_env.sh: chunked-prefill-size 65536 detected in SGLANG_SERVER_ARGS. "
                "This was a 2026-05-21 cherry-pick and was reverted on 2026-05-22 because it "
                "confounds v23 platform-acc attribution (v23 already differs from v22 in the "
                "hardened qzeros + tokenizer fixes; adding chunked-prefill 65536 makes a "
                "platform=0 result ambiguous between fix-failure and chunked-prefill poisoning). "
                "Use 8192 here for v23/v24; bring 65536 back in v25+ once the quant fixes are "
                "proven on platform. The 65536 config still lives in run_sala.sh for local bench."
            )

    prepare_model = variant_dir / "prepare_model.sh"
    if not prepare_model.is_file():
        problems.append("missing prepare_model.sh")
    elif quant_script.is_file():
        # Diagnostic blocks (commit d8aaaf1e4): platform-side observability.
        if not grep_file(prepare_model, "DIAGNOSTIC:"):
            problems.append(
                "prepare_model.sh: no DIAGNOSTIC: prints — platform log will not show "
                "calib jsonl size / output dir layout / safetensors file count. Without "
                "these, a platform=0 result is hard to attribute remotely. See commit d8aaaf1e4."
            )

    # Optional but standard for GPTQ
    if quant_script.is_file() and not (variant_dir / "perf_public_set.jsonl").is_file():
        problems.append("note: perf_public_set.jsonl not bundled (calibration will fall back to AutoDL or synthetic)")

    return problems


def stage_variant(variant_dir: Path, stage: Path) -> tuple[int, list[str]]:
    """Copy variant_dir/. into stage/ with symlinks dereferenced.

    Returns (file_count, cp_warnings).
    `cp -rL` warnings on broken symlinks are non-fatal (cp continues).
    """
    result = subprocess.run(
        ["cp", "-rL", f"{variant_dir}/.", str(stage)],
        capture_output=True, text=True,
    )
    warnings = [line for line in (result.stderr or "").splitlines() if line.strip()]
    # cp exits non-zero if ANY symlink was broken, but the rest of the tree
    # is still copied. We accept this so v17_minconfig (with a known broken
    # sgl-kernel .clang-format inside the bundled sglang) packs cleanly.
    for pyc in list(stage.rglob("__pycache__")):
        shutil.rmtree(pyc, ignore_errors=True)
    n_files = sum(1 for _ in stage.rglob("*") if _.is_file())
    return n_files, warnings


def make_tarball(stage: Path, out: Path) -> None:
    subprocess.check_call(
        [
            "tar", "--owner=0", "--group=0",
            "-czf", str(out),
            "-C", str(stage), ".",
        ]
    )


def verify_tarball(out: Path) -> dict:
    """Read the tarball back, return summary stats."""
    info: dict = {}
    info["size_mb"] = out.stat().st_size // (1024 * 1024)
    info["md5"] = md5(out)
    listing = subprocess.run(
        ["tar", "-tzf", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    info["n_entries"] = len(listing)
    top = sorted({l for l in listing if l.count("/") <= 1 and l.strip("./") and not l.endswith("/")})
    info["top_level"] = top
    return info


def pack(variant_dir: Path, out: Path, force: bool) -> int:
    problems = validate_variant(variant_dir)
    fatal = [p for p in problems if not p.startswith("note:")]
    for p in problems:
        marker = "WARN" if p.startswith("note:") else "FAIL"
        print(f"  [{marker}] {p}")
    if fatal and not force:
        print(f"\n[pack] {len(fatal)} validation failure(s). Pass --force to pack anyway.")
        return 2

    with tempfile.TemporaryDirectory(prefix="pack_submission_") as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()
        print(f"[pack] staging {variant_dir} -> {stage}")
        n_files, warnings = stage_variant(variant_dir, stage)
        if warnings:
            print(f"[pack] cp warnings ({len(warnings)} line(s)):")
            for w in warnings[:5]:
                print(f"    {w}")
            if len(warnings) > 5:
                print(f"    ... ({len(warnings) - 5} more)")
        print(f"[pack] staged {n_files} files")

        print(f"[pack] tar -> {out}")
        make_tarball(stage, out)

    info = verify_tarball(out)
    print(f"[pack] OK")
    print(f"    path:    {out}")
    print(f"    size:    {info['size_mb']} MB")
    print(f"    md5:     {info['md5']}")
    print(f"    entries: {info['n_entries']}")
    print(f"    top-level entries:")
    for entry in info["top_level"]:
        print(f"      {entry}")
    return 0


def check_only(variant_dir: Path) -> int:
    problems = validate_variant(variant_dir)
    if not problems:
        print(f"[check] {variant_dir}: OK (no problems)")
        return 0
    fatal = [p for p in problems if not p.startswith("note:")]
    for p in problems:
        marker = "WARN" if p.startswith("note:") else "FAIL"
        print(f"  [{marker}] {p}")
    return 0 if not fatal else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--variant",
        required=True,
        help="Variant subdir name (e.g. submission_gptqmodel_calib_w4a16)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output tarball path (defaults to ./<variant>.tar.gz at repo root)",
    )
    ap.add_argument(
        "--repo-root",
        default=".",
        help="Repo root (default: cwd)",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Don't pack — only validate the variant dir",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Pack even if validation reports fatal issues",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    variant_dir = (repo / args.variant).resolve()
    if not variant_dir.is_dir():
        print(f"variant not found: {variant_dir}", file=sys.stderr)
        return 2

    if args.check_only:
        return check_only(variant_dir)

    out_path = (
        Path(args.output).resolve()
        if args.output
        else (repo / f"{args.variant}.tar.gz")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return pack(variant_dir, out_path, args.force)


if __name__ == "__main__":
    sys.exit(main())
