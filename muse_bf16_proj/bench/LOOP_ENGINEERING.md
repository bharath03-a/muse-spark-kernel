# Projection experiment loop

This external tool compares complete submission interfaces on identical weights
and fresh activations. It never starts Modal, queues judge jobs, changes the GPU
budget, edits submission.py, or promotes candidates.

CPU validation (no GPU allocation):

```sh
.venv/bin/python -m unittest bench.test_loop_stats bench.test_experiment_loop bench.test_loop_report bench.test_loop_correctness
.venv/bin/python bench/experiment_loop.py --manifest bench/loop_manifest.json --dry-run
```

The supplied manifest pins cp15 and cp16 source hashes for reproducing the
unpooler128 physical-layout comparison. It is a reproducibility example, not a
new unmeasured optimization. Plans pin each source, select token counts/layers,
and specify even paired repetitions, at least two timing cohorts, absolute and
relative practical-gain thresholds, and a cooperative wall budget.

On an authorized GPU environment (NOT authorized as a new Modal job under the
current exhausted campaign cap):

```sh
python bench/experiment_loop.py --manifest bench/loop_manifest.json --output bench/loop_results.jsonl
python bench/loop_report.py bench/loop_results.jsonl --judge-profile bench/profiles/top50_candidate.03960c2e352c40e3.json
```

Each side captures kernel_fn; CUDA events measure raw graph replay after a
256MiB read flush. AB/BA order alternates, adjacent balanced pairs are bootstrap
units, and both independent cohorts must clear the confidence thresholds.
Normal/outlier/onehot/zero tests and every fresh timing input are compared with
FP64 outside event timing. Raw samples, correctness and environment/source hashes
are retained. The report rebuilds decisions from samples and rejects partial,
failed or malformed runs. Score transfers include regressions and are explicitly
hypothetical, never judge evidence. No assumed 4us is subtracted by default.

CPU tests validate statistics, plans, reporting, input generation and numerical
correctness gates (including sparse row/probe corruption) using torch on CPU. GPU compilation,
capture, numerical checks and timing of the new runner are not yet validated.
The wall budget is cooperative: it is checked between operations and cannot
interrupt a hung GPU kernel or long prepare/compile. Both complete modules may
consume substantial preparation time and memory. Judge validation remains the
release gate for a candidate, and no result alone guarantees leaderboard rank.
