# Independent n64 tail resource experiment

This is a disconnected research kernel. No default, submission, or scored candidate is changed. The source is derived from the grid-constant prefill baseline, preserving the upstream Apache-2.0 attribution in `OPENKERNELS_LICENSE.txt` (copyright 2026 Lee Penkman).

## Mechanism and prior evidence

The current 128-token kernel computes a padded 128-token tile for expert remainders of 33–64 tokens. A separate 64-token kernel can halve the tensor work for those tails. This differs from the rejected upstream mixed n8/n16/n32 remainder kernel: that kernel compiled at 96 registers with spills, while n32 alone used 67 registers without spills. The latest pinned upstream e6b48c8 has no prefill source changes versus 47103fd.

Observed routing profiles give a mathematical ceiling on removed padded token work: about 7–10.3% at T=753, 3.2–4.2% at T=3012, and 0.6–1.2% at 16K. These are not latency forecasts. The new launch overhead, doubled weight-row tile count, activation scheduling changes, and queue imbalance can erase the gain. Expected whole-layer gain is at most a few percent.

## Configuration

Only `w4g_k<64,2>` changes its launch bound and register redistribution. It uses 384 threads (128 producer, 256 consumer), BM128, 5 stages, and lookahead 3. The queue selects final `count % 128` tails in [33,64]. Full K is accumulated before BF16 output; fused W13 preserves gate/up BF16 rounding before FP32 SiLU and BF16 intermediate output. Existing native banks and packing are reused.

The proposed producer/consumer register budgets are 32/104. Two blocks require 61,440 dynamically redistributed registers and 24 warps. Shared memory is 73,248 dynamic bytes per block (including 544-byte control structure and 1,024-byte scratch allowance); static shared allocation must be added from compiler output. If static shared remains 1,024 bytes, two blocks require 148,544 shared bytes. Compiler resources and SASS, rather than this budget alone, determine whether the hypothesis survives.

`prove_tail64.py` exhaustively checks counts 0..4096, proves the NCW2 packed row-block mapping is bijective for W13 and W2, and checks the control structure size. It passes. The prototype intentionally leaves all nonselected rows untouched. Future numerical checks must restrict comparisons to selected valid rows and sentinel-check other rows.

## Reproduction and state

Run `python3 muse_prefill_layer/research/build_tail64.py`, then the Modal CLI on `research/modal_tail64_cpu.py::compile_resources`. This function allocates CPU only. Resource experiment app: ap-M5HfO1ZhCJfm97lY6XAok8; local log `/tmp/prefill_tail64_cpu.log`.

Files: `build_tail64.py`, `kernel_sources/w4tail64.cu`, `modal_tail64_cpu.py`, and `prove_tail64.py`. Source SHA256: `633c58b2871a877acd61830022caf512674c2a8f0960897f4bb23c21b9e36347`.

MoE peer review confirms the packed offsets and register arithmetic. Halving NT while doubling row CTAs preserves total activation tile bytes versus the old128 tile. Runtime occupancy must still be measured with the final dynamic shared size. Integration must initially target mode0 only: mode1 wide256 owns a different tail partition, and adding this 128-relative queue could overlap writes. Each launch needs its own queue counters and complete reset.

Compilation is pending. No GPU execution is authorized yet. Promotion requires zero spill evidence first, then a bounded GPU test including boundaries 32/33/47/64/65, both W13 fusion and W2, and a full sequence comparison with dispatch overhead included.


## CPU result

The first resource build completed in 69.52 seconds. n64 NCW2 uses 80 launch registers, 0 stack bytes, 0 local bytes, 0 spill loads/stores, and 0 SASS LDL/STL instructions. Static shared is 1,024 bytes, so two blocks require 148,544 shared bytes and 61,440 launch registers. This satisfies the resource hypothesis; actual occupancy remains a GPU query.

A second CPU build adds an occupancy export and a mode0-only filtered full sequence. The original big queue skips remainders 33..64 only when `fullonly==2`; native call behavior remains unchanged. The candidate runs filtered big, original n32, and new n64 launches with independent counters. It is still disconnected from production.

The bounded GPU runner is `modal_tail64_cpu.py::probe`. It checks 3 fresh seeds, mixed tails at32/33/47/64/65 and128-offset variants, zero and negative activations, valid-row finite checks, untouched sentinels, canonical reference error, exact old/new equality, and graph replay. Both fused W13 and W2 are checked; a realistic profile also times complete W13→W2 sequences with launch overhead included. Root approved allocation after attention release; CPU v2 must finish first.


## H200 result: reject integration

Approved bounded GPU app `ap-5xBKkts2KStFrM5dgQ7pHL` completed. Runtime occupancy is exactly 2 blocks/SM. All 24 reference/finite/bitidentical/sentinel checks and all 27 paired replay guards passed across 3 seeds. The full realistic-profile W13→W2 sequence regressed consistently:

| Seed | Baseline us | Candidate us | Paired speedup |
| --- | ---: | ---: | ---: |
| 9391 | 487.49 | 593.33 | 0.8213 |
| 9392 | 488.42 | 595.94 | 0.8198 |
| 9393 | 488.37 | 594.96 | 0.8206 |

Profile W2 alone was ~0.787–0.788x and fused W13 ~0.845–0.847x. An eligible-only zero-activation/count control improved ~1.13–1.15x, showing the narrower kernel can help when nearly all work falls in its bucket. Mixed 32/33/47/64/65 tails regressed to ~0.70–0.71x. The useful width specialization is overwhelmed when a separate queue/launch must coexist with existing work. This is evidence against production integration, despite the successful two-block occupancy and zero spills. No official check or default change is warranted.

Caveat: this was a same-source queue ablation. The baseline native launch uses unchanged `fullonly=0` semantics, but the compiled big kernel contains the additional `fullonly=2` predicate required by filtering; it is not byte-identical to the scored grid-P binary. This prevents claiming a precise improvement against scored grid-P. The large and consistent negative result is sufficient to reject this prototype without spending a confirmation run.

Raw log: `/tmp/prefill_tail64_gpu.log`. Structured results: `tail64_paired_summary.json`. The second CPU build (`ap-jNwnRnTnfuih8dN6t5n1X2`) completed in64.44s, source SHA256 `7f1f4057c5497f6fbc98fc709fdc90cbee1053e7b9c9ffce488d291172c3ff42`, maintaining REG80 and zero stack/local/spills. GPU is released.
