# W4G feed research, October27

Measured/scored starting point: promoted fused prefill274396.63us (job dcf056bc66d54bde). Earlier public profile e858d0f8482641e6 attributes W4 kernels~62% of753 sliding,~51%3012 sliding, and~40%16Kfull. Epilogue fusion is a useful~2–3% micro gain; larger gains need the tensor-core feed path.

## Register/shared occupancy

Current wide256/NCW2:384threads, launch168regs/thread then producer128x40 + consumers256x232 =64512registers. Current standard128/NCW4:640threads, launch96 then producer128x32 + consumers512x112 =61440registers. Big kernels have zero local stack/spills in actual SASS. Dynamic shared requirement approximately196128bytes wide,202384bytes standard; only one CTA fits perSM.

The producer128-thread warpgroup largely waits while thread0 submits packed-weight bulk TMA and activation-tile TMA. Consumers load packed weights from shared, unpack/scaled BF16 into register fragments, then issue RS WGMMA. Existing W4STEP already interleaves half-stage dequant with outstanding WGMMA; simply adding a second full register buffer previously made ptxas serialize at the112-register consumer budget (source comment).

## Decoupled SS prototype proposal

Keep activation TMA and exact dq_load/dq_half unchanged. Let producer threads decode packed weights into a SW128 BF16 shared A stage, then consumers use WGMMA_SS. For wide mode three activation+expanded-weight stages cost147456bytes, plus18432bytes packed scratch and~1.5KB metadata/alignment:~167KB versus current196KB. Standard needs~186KB with three stages. All fit within228KB opt-inSMEM.

Need a distinct decoded-ready barrier and proxy fence. Do not reuse the TMA-complete barrier as proof that software dequant stores are visible to WGMMA. Preserve per-slot consumption/release and all producer/consumer arrival counts. Three stages shorten current five/seven-stage TMA lookahead, so starvation can negate gains. Added BF16 shared stores/reads increase shared-bandwidth demand. Consumer register savings could transfer to producer dequant capacity; e.g. consumer208,producer64 gives61440registers in wide mode, but only compilation can establish feasibility.

The proposed helper research/ss_weight_stage.cuh maps existing RS register fragments to shared without changing bits. CPU research/prove_ss_weight_stage.py proves every BF16 element is addressed exactly once and in bounds for NCW1/2/4. It does NOT prove full kernel synchronization, arithmetic, or speed. Activation layout remains unchanged, avoiding previous mixed-scale/coalescing failures.

Falsifying sequence: existing W4PROF phase counters on uniform/skewed753/3012/16384, then standalone same full4096K WGMMA_SS versusRS wide256 pipeline with actualNVFP4weights and finite/correctness before timing. If dequant is already fully hidden, or shared-write synchronization dominates, stop before full-layer integration. Instruction-count reductions alone are not performance evidence.

## Parameter storage / instruction-layout ablation

MoE peer found136B parameter stack copy in ring and removed it with const __grid_constant__ P, later measured~3% ring gain. W4G already has stack0/local0/LDL0/STL0, so that spill mechanism does NOT transfer.

Nevertheless same-source CPU compilation shows W4G annotation changes generated code: static SASS instructions wide5112→3712, standard3560→2808, small2320→2136; big register allocations unchanged168/96, small64→62. Wide IMAD1056→545 but BRA181→243, suggesting changed compiler unrolling/control layout, with uncertain dynamic impact. This warrants an isolated paired timing only if explicitly allocated; no speedup inferred. Exact outputs: /tmp/prefill_base_w4g.sass and /tmp/prefill_grid_w4g.sass. Builder research/build_grid_constant.py; candidate_prefill_grid_constant.py. All originals preserved.

NVIDIA primary explanation of parameter-copy rule: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html . CUTLASS uses the same qualifier on its operator parameters: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/device_kernel.h .

## Diagnostic runner

research/modal_w4_phase_profile.py provides CPU precompilation (unique instrumented module prefill_w4g_phaseprof_h) and one240s-bounded H200 diagnostic. It checks finite, bit-identical full outputs versus the uninstrumented kernel before printing per-stage consumer wait/dequant/MMA/epilogue and producer wait/issue counters. These counters perturb scheduling and are diagnostic, not leaderboard timings. No GPU run had launched when this note was written.

## Completed GPU diagnostics

Phase app ap-OSp3tPrKiIoVId6CvXo6Fm completed18finite, bit-identical checks. Durable data: research/w4_phase_profile_summary.json; full log /tmp/prefill_w4_phase_gpu.log. At16384skew:

- Wide: consumer wait212cycles/stage, dequant412, MMA issue/retire480; respective warp-span fractions14.1%,27.3%,31.9%, epilogue6.0%. Producer wait-empty759cycles/stage, issue448.
- Standard: consumer wait264,dequant474,MMA623cycles/stage; fractions14.2%,25.5%,33.5%, epilogue4.3%. Producer wait-empty1128,issue431.
- Mixed mode's wide and standard classes show comparable fractions. Empty remainder classes have zero warp-stages and are excluded from bottleneck interpretation.

These instrumented warp-span fractions are not whole-kernel exclusive critical-path shares: warpgroups overlap, producer/consumer times cannot be added, and instrumentation changes scheduling. They support a nontrivial decode dependency chain, not a promised27% end-to-end improvement. With four producer warps versus eight wide consumer warps, producer decode can roughly double the instruction work assigned to each producer warp; current759cycle wait-empty slack versus~412cycle consumer dequant is encouraging but does not prove enough capacity after adding shared stores/barriers.

Grid-constant A/B app ap-8Y1gbn70Y5V4m8fUeLG2Lw passed18finite, bit-identical, watchdog and replay-guard cases, three fresh weight/input seeds.24pairedAB/BAreplays percase with read-only256MiBL2flush and95%CI floor1.5%. Medians across seeds:

| Rows/distribution/mode | Speedup | Evidence |
|---|---:|---|
|1001uniform/m0|1.0153x|borderline|
|1001skew/m2|1.0080x|inconclusive|
|3008uniform/m1|1.0099x|inconclusive|
|3008skew/m0|1.0015x|inconclusive|
|16385uniform/m2|1.0164x|all3CI lows>1|
|16385skew/m1|1.0165x|2of3CI lows>1|

Durable per-seed summaries: research/grid_constant_paired_summary.json; full log /tmp/prefill_grid_constant_gpu.log. Official whole-prefill check4356078995734993 was launched by root; no promotion claim until it completes.

New upstream HEAD47103fd9c1212868018eb615a4c6467e366e9ce0 is under root review before a larger SS rewrite. No additional GPU work launched.
