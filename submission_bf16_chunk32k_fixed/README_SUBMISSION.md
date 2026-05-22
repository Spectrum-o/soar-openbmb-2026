# SOAR BF16 + Chunk32K (v5e) — chunked-prefill 32K isolated

> Built 2026-05-23 while v5d (BF16 base config-fix) is running on the platform.
> If v5d succeeds at ~19-22 final_score, v5e is the next step: same env +
> config-fix, but with chunked-prefill bumped from 8192 → 32768 (and
> --max-prefill-tokens bumped to match so the default 16384 cap doesn't
> defeat the bump).

## Diff from v5d (the diagnostic base)

**One file change, two flags added** to `prepare_env.sh:347`:

```diff
- export SGLANG_SERVER_ARGS="... --chunked-prefill-size 8192 --skip-server-warmup ..."
+ export SGLANG_SERVER_ARGS="... --chunked-prefill-size 32768 --max-prefill-tokens 32768 --skip-server-warmup ..."
```

`prepare_model.sh` identical to v5d (config.json with auto_map.AutoConfig stripped).

## What's still NOT included (deliberately)

- `--enable-mixed-chunk` — v3 and v5 both had it; we don't know if it
  was the actual cause of their crashes. Defer to v5f.
- chunked-prefill 65K — even more aggressive. Defer to v5g if v5e works
  and we want more throughput.

## Expected outcome

| Result | final_score | Action |
|---|---|---|
| ✅ Works ~25-30 min | **28-32** | **rank 20 secured**, this is THE target |
| ⛔ Crashes early | 0 | chunked-prefill 32K is incompatible somewhere; revert to v5d's 8192 |
| ⛔ Loads but acc drops | low | Unlikely (chunked-prefill is a scheduling knob, not a correctness knob) |

## Local evidence for the throughput gain

`memory/project_chunked_prefill_win.md`: 8K → 32K chunked-prefill on SALA local benchmark = +59% throughput, -60% TTFT. Scaling to platform: BF16 baseline 19.13 → ~28-30 final_score plausible.

## Submission gate

Only submit v5e AFTER v5d returns:
- v5d succeeds → submit v5e
- v5d crashes → DON'T submit v5e (auto_map.AutoConfig fix didn't help, chunked-prefill change won't either)
