# Selective pf32 integration

Candidate: candidate_prefill_pf32_hybrid.py, reproducible research/build_pf32_hybrid.py. Baseline is the exact grid-P candidate checked in job4356078995734993 (268591.14us off-board), not the earlier unannotated backend.

Mechanism: preserve grid-P below65536allocated expert rows; use strict pf32 vector-store W4G backend at or above that capacity. PDL is explicitly disabled because the component experiment showed no additional benefit. Dispatch reads only shape/capacity, not input/routing values.

Original mixed experiment versus unannotated old backend passed18fresh numerical/replay cases, with smalluniform mode0 regression~2.5% and large~2.2–2.5% gains. That justifies a selective hypothesis, not promotion. New hybrid must be measured against current grid-P; its incremental gain may be smaller.

No duplicate packed weight bank is created. Both paths use the identical losslessly paired W13 and native W2 banks, BF16 gate/up/inter/down-output boundaries, BF16 effective norm weights and complete-KW2. Explicit diagnostic extensions remain on their original dispatch. Small-shape native fallback still prepares untimed, including FP32small-router input and slicedunpool.

The PF32CUDA source, export list and flags are unchanged from the already CPU-validated candidate and reuse extension pf2_w4g_pf32_strict_v1h. Final CUDA SHA256:4155fb532fd1817d3cdc0c5b302ab4df1643d61378f50f32c9ca5c6ead1f98e7. Only Python dispatch/loading is new; Python compilation passes. Cold judge preparation may add~73seconds for this second extension; full preparation wall time is printed. Compile failure or preparation watchdog falls back to the scored grid-P backend.

research/modal_pf32_hybrid_probe.py provides a bounded300sH200 comparison of CURRENTgrid-P versus hybrid, three fresh weight/input seeds,18cases total. Small controls include nearbycapacity6023; largecapacity131072 spans modes0/1/2. Actual local routes are distinct from allocated capacity, and all numerical/replay checks view only valid routed rows. The timed graph contains the complete W13fused→W2 sequence. No GPU run had launched when this note was written; root coordinates allocation after attention.

## Completed falsifying comparison

GPU app ap-oAhuOfMxSR2mizSfBb1TsP completed18/18finite, bit-identical old/new, sampled denseNVFP4reference and replay-guard checks. Maximum sampled reference relativeL2:0.000943. Against CURRENTgrid-P (three fresh seeds):

- Smallcontrols: median0.9998x,0.9992x,1.0005x.
- Largeuniform mode0:0.9890x (regression).
- Largeskew mode1:1.0079x;mode2:1.0071x. All large95%CIs include1 after the1.5%floor.

Decision:reject hybrid, no official check. Prior2.2–2.5%gain versus the unannotated backend was largely already captured by grid-P. This result prevents double counting component gains. Durable per-seed summaries: research/pf32_hybrid_paired_summary.json; full log /tmp/prefill_pf32_hybrid_gpu.log. GPU released.
