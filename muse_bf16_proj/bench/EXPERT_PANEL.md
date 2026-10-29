# Expert panel: technical decision record

Scope: projection challenge, top50 objective, existing Modal cap remains binding.
This round is CPU/source research only; no new GPU jobs, no measured gains.

## Team and peer review

- expert_scheduling: decomposition, split reductions, cluster lifetime, shape coverage.
- expert_memory: physical traffic, lossless formats, exception overhead, numerical rules.
- expert_compiler: Triton3.8/Gluon API, PTX operand mapping, register/shared layout, async groups.
- Parent: objective, actual-profile score calculation, hypothesis gates, experiment budget.

Agents messaged each other directly. Compiler rejected a free v4-store assumption;
memory rejected multi-us zip-epilogue claims; scheduling incorporated both corrections.
Read expert_scheduling.md, expert_memory.md, expert_compiler.md for details and sources.

## Ranked work after cross-review

1. Direct register-source async cluster partial pushes. Distinct from the already-losing
   global cooperative reducer. Scalar stores can preserve the existing mailbox;
   v2 needs mailbox[token,row] -> mailbox[row,token] and a final store-layout plan.
   Preserve initialization and all remote-memory lifetimes. Validate Gluon physical
   layouts/lowering before claiming a runnable cluster implementation. 0.2–0.7us
   per affected shape is a HYPOTHESIS, not a benchmark result.
2. Q-group Gluon ring. Q2 must prefill S-2 and refill i+S-2, with wait2 and finalwait0.
   Changing wait1 alone is incorrect. Offline probe is separate from submission and
   must pass compile, outliers, onehot, changed-input and repeated graph checks later.
3. Physical weight-transpose vendor control. Existing test_mm_vs_linear.py authored
   this but has no saved result and uses inadequate host timing. cuBLASLt enumeration
   fixed one layout, so it does not reject this mechanism. This is a vendor-library
   alternative, explicitly NOT a new Triton-only kernel recommendation.

## Retired or bounded ideas

- BN32 dense mma_v2 and persistent continuation: measured losses in campaign.
- Global distributed router reduction: repeated ties/losses; don't retune indefinitely.
- x multicast on the ingress diagnosis: falsified by actual no-x ablation.
- Zip exceptions: ~0.013bits/weight; storage optimization is too small to matter.
- Perfect11bit rather than12bit full-matrix payload saves ~0.95us at4.4TB/s BEFORE
  added decoding. No promise of score gain or decoder implementation.
- Current zip already decodes into register fragments. That is not an untried idea.

## Score accounting

CPU calculator: top50_score_plan.py. Actual cp14 saved score1.426074, target snapshot
1.5271: requires6.62% lower geometric-mean time. Target changes with the board.
Hypothetical savings qkv16/32=0.5us each; router8/16/32=0.3us each and router128/256
=0.5us each yield1.437873, NOT top50. Tiny epilogue changes alone are insufficient.
At an equal relative improvement across loads, required savings are2.421,2.521,
3.532,3.956,4.579,5.805us for m1,8,16,32,128,256 respectively. These are allocation
targets, not forecasts or physical lower bounds.

## Release gates

No existing checkpoint or submission was changed by this panel. Future candidates need:
correct compiler lowering/resource counts; actual correctness including changed inputs
and nearby shapes; paired cold event-graph advantage over cp14; legitimate okc check.
Assembly and arithmetic establish mechanisms, not a proven performance improvement.
The top50 objective remains incomplete. Do not reset the GPU ledger or increase its cap.

## Isolated cluster-lowering probe

`expert_cluster_epilogue.py` now exercises the proposed register-source path
without any GEMM changes. Primary upstream evidence is Triton v3.8.0
`python/test/gluon/test_core.py`, async_shared_store_kernel/test_async_shared_store
(lines148–199). The probe changes CTA distribution from a split-plane axis to a
token axis, then sums split planes locally. It compares generic distributed stores
with async stores, checks generated PTX for st.async, tests changed inputs, and
reports paired raw graph timing. CPU syntax passed only; cluster redistribution,
GPU lowering, numerical correctness and performance remain UNVALIDATED. This does
not yet prove WGMMA accumulator redistribution avoids an expensive conversion.

## Canonical baseline update

Judge check03960c2e352c40e3 completed:9/9 tests, no flags,1.426996 speedup.
Its exact source (cp14 plus dormant TMA-output descriptor repair) now replaces
the old cp5 submission.py. The old source is preserved at
checkpoints/pre_top50_submission_backup.py. This is a baseline promotion,
not evidence of a new architectural speedup. Other checkpoints are unchanged.

## Current release state and router experiment
Canonical is cp18, retaining measured pooler m32 ~0.99us and cp17 unpooler m32 ~1.28us gains. Overall 1.45x/top50 are not established. No new Modal GPU jobs.

The cluster idea now has CPU compiler proof for 2/4/8 CTA epilogues and a complete Triton-only router BT64 prototype (not merely syntax). The mainloop uses hi/lo BF16 weights with separate FP32 accumulators, rank3 MMA2 and a 96KiB cp.async ring; the cluster reduction retains all peers through mailbox use. All four C4/C8 × STORE/ASYNC variants compile at 147–149 registers and zero spills. Source, layout, grid, tail masks and wrapper gates received independent expert review.

Pending judge check002617846e374c4c is the only live experiment. It targets the documented 64→128 CTA occupancy change and removes the global reducer used by earlier rejected routers. This is not evidence of a performance win. Generic and async controls must pass fresh-input FP32 correctness before paired timing; cp18 remains fallback. Raw CPU evidence: router_cluster_bt64_compile_fixed.out. Release source SHA e98d58899ed2f04cf0dce36aff7fff610e3c9b9a301477f68e3fcbb933192494.
