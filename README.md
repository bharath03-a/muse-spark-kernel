# Muse Spark Kernel

Our H200 kernels for the [Muse Open Kernel Challenge](https://github.com/vjeux/okc-2026) (OKC, Nov 2026): the Muse Spark decoder layer and its parts, timed against SGLang.

## Final results (submissions closed 2026-11-04, 11:59 pm PT)

| Problem | Best speedup vs SGLang | Rank | Code |
|---|---|---|---|
| `muse_decode_layer` (sets overall rank) | **2.258x** | #237 | credited base + our changes (`OURS`) |
| `muse_bf16_proj` | 1.4814x | — | ours |
| `muse_moe_w4a16` | 2.384x | #238 | credited |

Details: [`muse_decode_layer/FINAL_RESULTS_NOV4.md`](muse_decode_layer/FINAL_RESULTS_NOV4.md). Tag: `okc-final-2026-11-04`.

## Layout

| Directory | Problem |
|---|---|
| `muse_decode_layer/` | Whole decoder layer: attention, projections, MoE, glue. Best: `candidate_last_v5.py` |
| `muse_bf16_proj/` | Dense BF16 projections. Our own 12-bit lossless zip GEMV, TMA/wgmma router |
| `muse_moe_w4a16/` | Routed experts, W4A16. Our own MoE: `own_moe_v1..v5.py` (v5 = hand-written CUDA, 1.61x) |
| `muse_attention_decode/` | Gated sliding/full attention decode |
| `muse_prefill_layer/` | Prefill layer (not scored in overall rank) |
| `research_harness/` | Experiment tracking |

Each problem directory has `TASK.md` (spec), `reference.py`, `checks.py`, candidates, `modal_*.py` timing scripts and an experiment log (`EXPERIMENTS_V2.md` / `HILLCLIMB_LOG.md`).

## Workflow

```bash
okc check <problem> <file>     # scored run, kept off the board
okc submit <problem> <file>    # scored run, on the board (best run kept)
.venv/bin/modal run <problem>/modal_*.py   # paired cold-L2 A/B on H200
```

Use `uv run python`, not the system `python3`.

## Practices

- **Attribution:** third-party notices stay in every file; our code sits in blocks marked `OURS`. See `NOTICE_keell0renz`, `*/OPENKERNELS_LICENSE.txt` and `muse_bf16_proj/PROVENANCE.md`. Only Apache-2.0 / MIT sources are reused.
- **Correctness first:** a candidate must pass the judge's tests (13/13 for the decoder). Runtime self-tests fall back to the known-good path on mismatch.
- **Measure, don't guess:** judge noise is about ±1.5% per run. Compare per-load µs against paired baselines; a sub-1% single-check gain is noise.
- **No judge gaming:** no input-specific caching, no cosmetic resubmits to re-roll noise, no lossy weight changes.
- **Log everything:** every experiment, including no-gos, goes in the problem's log with its job ID.

## License

No license file yet. Third-party code keeps its original license (Apache-2.0 / MIT) as noted in the files above.
