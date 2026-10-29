# Research experiment loop

This is external research tooling. Submitted kernels never import it.

Run `python3 -m research_harness.loop research_harness/prefill_queue.json`.
It serializes experiment queues with a workspace-wide OS lock, fingerprints source and experiment settings,
records logs/results durably, skips recorded fingerprints on restart, and stops
on failed checks, missing result markers, or bounded timeouts. An intentional
retry requires `--retry-failed`; a skipped failure still returns nonzero. Each
run saves a source snapshot, and source changes during a run invalidate it.
Dispatch is journaled before launch; an interrupted `started` entry blocks a
silent restart. Inspect the remote app before explicitly retrying that task.
It never submits leaderboard entries. A local
CLI timeout terminates its process group; inspect Modal apps before retrying
because a remotely detached worker may still be alive.

The prefill queue tests the strict ring implementation and the experimental
W13 + SwiGLU fusion on three fresh seeds, six size/tile combinations per seed.
CPU compilation is separate and cached so timed GPU work does not repeatedly
pay for compilation. Do not launch the queue while the CPU cache builder runs.
Call `environment.configure_h200_cache()` before importing torch/SGLang on
both CPU and GPU workers. A CPU build for `9.0a` and GPU auto-selection of `9.0`
can silently recompile the same extension on paid GPU time. SGLang uses its
own `SGLANG_JIT_CACHE_DIR`; setting only the TVM cache does not preserve it.
`prefill_mixed_queue.json` adds uneven/skewed routes, idle experts, nearby sizes,
and mixed tile mode 2. Root also coordinates GPU allocations with agent runners.

`gpu.compare_graphs` captures both functions, warms on a side stream, reads
256 MiB before each timed replay, alternates AB/BA order, and retains raw
microsecond samples. Adjacent reversed-order ratios are geometrically combined;
the median and a block-bootstrap 95% interval are reported with at least ±1.5%
uncertainty. This is a screening estimate, not the official judge's score.
Fixed graph overhead is subtracted only when explicitly calibrated.

Callers must validate against a reference on fresh input seeds before timing.
The timed functions must use identical inputs. For non-idempotent mutable
inputs, supply `reset_inputs` to restore identical inputs before every replay,
outside the events. It must never calculate candidate intermediates. Keep all
input-derived routing and other algorithm work inside the captured function.
Check repeated replays for scratch-counter/output drift before trusting timings.

Only `promising` cases whose interval excludes 1 are worth further experiments.
A whole-problem decision needs all scored shapes and their actual load weights:
use `stats.weighted_load_speedup`, not an unweighted average of case speedups.
This uses paired-derived candidate times when available, retaining raw medians
separately for diagnosis. Check tensors explicitly for NaN/Inf before calling
a numerical comparison helper; a helper may accidentally accept NaN errors.
Then run an official `okc check`, inspect correctness/flags and preparation time,
and submit the exact passing source if the whole-problem gain is meaningful.
Microkernel timing alone never establishes a leaderboard gain.

Run `python3 -m unittest research_harness.test_harness` for statistics, invalid
sample handling, load aggregation, resume, and missing-result regression tests.
