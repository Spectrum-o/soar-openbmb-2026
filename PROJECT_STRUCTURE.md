# Project Structure

This repository is a SOAR competition working tree on top of a customized SGLang fork.
The tree is split between upstream runtime code and competition workflow code.

## Top-level map

| Path | Role |
|---|---|
| `python/`, `3rdparty/`, `sgl-kernel/`, `sgl-model-gateway/` | Runtime and kernel code for the MiniCPM-SALA fork. |
| `docs/`, `benchmark/`, `examples/`, `test/` | Upstream docs, benchmarks, examples, and large integration tests. |
| `tools/` | Python utilities for packing, linting, diagnostics, and eval analysis. |
| `scripts/` | Operational entrypoints: preflight, local eval, smoke tests, and GPU session automation. |
| `experiments/` | Runbooks, investigations, and strategy docs. |
| `handoff/` | Compact session handoff context. |
| `outputs/` | Local eval outputs and prediction dumps. |
| `submission_*` | SOAR submission variants. These stay at repo root because scripts reference them directly. |
| `SUBMISSIONS.md` | Canonical submission log and hard constraints. |
| `W4A16_README.md` | High-level W4A16 recipe and env notes. |
| `CLAUDE.md` | High-signal operator notes for future sessions. |

## Recommended reading order

1. `PROJECT_STRUCTURE.md`
2. `CLAUDE.md`
3. `experiments/README.md`
4. `SUBMISSIONS.md`

## Why `submission_*` remains at repo root

This is the main source of visual clutter, but moving it right now is high-risk:

- `scripts/full_preflight.sh`, `scripts/local_eval.sh`, `scripts/sanity_check.sh`,
  `scripts/v24_auto.sh`, and several experiment scripts reference these root-level names.
- Many variants are symlink-based forks of a base variant.
- The pack workflow assumes `submission_<name>` paths from repo root.

So the current cleanup strategy is conservative:

- keep variant paths stable,
- improve repo navigation,
- keep notes and operational artifacts grouped under `experiments/`, `handoff/`, `scripts/`, and `tools/`.

## Practical navigation

- To run or debug a submission: start at `SUBMISSIONS.md`, then `scripts/full_preflight.sh`, then the target `submission_*`.
- To understand current strategy: start at `experiments/STATE_OF_PLAY.md`.
- To inspect automation: check `scripts/` and `tools/`.
- To inspect model/runtime changes: check `python/sglang/srt/`.
