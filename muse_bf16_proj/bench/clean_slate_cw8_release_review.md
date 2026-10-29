# CW8 pooler release peer review

Source-only audit, 2026-10-28. Read TASK.md and current CAMPAIGN_STATE.json first. No GPU, Modal or OKC jobs; canonical untouched. **No blocking source bug found.** This approves a gated experiment, not a speedup claim or promotion.

Candidate: `bench/top50_candidate_cw8_pooler.py`
SHA256: `dad17a60f386060cce3f66b7a34d36705aba1abc6269441d799ac521503ccf6d`

## Identity and precision

- Candidate starts with the exact bytes of both current submission.py and checkpoints/cp17_async_unpooler32_pass.py, SHA256 `d2c3272138c899a9643d42c58b1291680bccf99cc54a2dbf8e7e3110b091b85f`.
- Inlined issue/kernel/decoder ASTs equal bench/adaptive_record_async_warps.py after function-name normalization. Thus earlier pinned3.8 offline compile evidence applies to these exact bodies: CW8 WX, 196800B shared, 128 registers/thread, no stack/spills; scratch can reuse ring storage after final wait0 plus CTA fence.
- Uses unchanged cp17 `_ac_pack` and `_ac_order`. Compressed records encode the original BF16 low bytes/sign/exponent losslessly; any out-of-window value makes its whole record raw. Raw records are exactly rebased with a power of two. Nonzero subnormal/nonfinite/out-of-range rebases reject packing, so the wrapper falls back. Signed zeros retain bits. No approximate weight format introduced.
- CPU tests rerun: adaptive_record_warps_mapping_test.py and adaptive_record_async_mapping_test.py PASS. CW4/8 each cover all N64×K1024 weights exactly once; decoder mapping and ring depth2/3/4 proofs pass. AST parse passes. No external bench imports in the candidate.

## Wrapper and selection

- `_cw_base_prepare` and `_cw_base_kernel` capture cp17 function objects before redefining the wrappers. Base prepare runs first, retaining all cp17 state and its unpooler selection. cw8_enabled starts false; rejection leaves it false and removes only cw_packed.
- Compile/resource/numerical gates run separately for CW4 and CW8. Lambda default cw freezes the control; graph lambda default fn freezes its function. `_cw_run` allocates a fresh output on every ordinary call; captured graph allocation is retained by graph memory. Packed payload/metadata stay in state whenever enabled.
- `_r15_valid` uses FP64 references. `_ac_graph_check` captures once then overwrites input with fresh normal/outlier distributions including one-hot probes, replays each three times, compares global/row/probe errors, then checks exact-zero output. Numerical rejection prints a reason and excludes that control; exceptions print full tracebacks and retain cp17.
- CW4 timing is diagnostic. CW8 must exist in validated controls, then beat the current cp17 selected path in two separate31-repetition alternating-order cold-L2 rounds by more than max(0.5us,2.5%). Selection uses the second confirmation only if both pass. A time over260s disables it. The final dispatch accepts only pooler, m32, expected n, contiguous x and16B alignment. Every other call, including unpooler m32, invokes the cp17 kernel object.
- Outer started time precedes base prepare. Admission cutoffs210/245/250/255/260s share this same start, so no separate post-base260s budget is accidentally granted. As with existing cp17, these are admission/selection checks rather than cancellation of an in-progress synchronous compile; no new hard wall-clock guarantee is claimed.

## Scope

CW8 is a local eight-way K split with fixed32 output rows, distinct from ac2 [4,2,1] row-warps or cross-CTA split/reduction. Useful payload/X/math volume stays constant; total shared writes include masked tail zero-fill, and CW8 increases that overhead. Performance remains unmeasured for this candidate. Preserve cp17 unless the gated candidate passes official tests with a verified selected gain.
