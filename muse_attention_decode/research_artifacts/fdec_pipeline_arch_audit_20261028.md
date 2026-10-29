# Fused decode pipeline audit, October 28

Source: `probe_fdec_sweep.py` fdec and run_fdec; no GPU experiment performed for this audit.

## Release hazards

The original synthetic probe allocated page-table capacity 80008, then sampled sequence positions in [74000,85999]. Its `torch.gather` therefore accesses out of bounds. The ensuing CUDA-context failure invalidates all reported failures in that run; it is not numerical evidence against any pipeline variant.

RBF6 additionally exposes a kernel prologue bug: the new-token repair predicate enumerates rpro[last], rpro[0], rpro[1], rpro[2]. For six buffers it omits rpro[3]. Replace this enumeration with the OR of all RBF-1 prologue entries before testing the variant on short or uneven splits. Otherwise a new token in the fourth prologue tile remains unwritten in shared memory.

Balanced dispatch is valid only for B<=32 because request lengths are mapped to one warp. Production eligibility must enforce this.

## Resource and queue derivation

Dynamic shared memory is max(32768*RBF+18688,67584). Thus RBF5=182528 B and RBF6=215296 B. Both allow only one CTA per H200 SM. RBF5 can use the 196 KiB shared carveout; RBF6 needs 228 KiB, reducing remaining unified L1 capacity. RBF7 needs248064 B and exceeds the block limit.

Each warp stages 16 keys and 16 values, each128 B, per ring tile:4096 B. RBF5 maintains four pending ordinary async-copy groups, RBF6 five; the waits are3 and4 respectively. RBF6 adds4 KiB of demand-copy lookahead per warp without increasing resident warps. The four-segment page-table ring holds512 entries; two initially loaded128-entry segments and the subsequent per-eight-tile refill cover RBF6's eight-tile row lookahead.

PFF4 issues one128 B L2 bulk-prefetch per lane: lower16 lanes fetch K rows, upper16 fetch V rows. This is exactly4096 B per future tile, not duplicated K/V rows. It extends the demand pipeline by four tiles (16 KiB/warp) without extra shared memory. All cache rows have128 B stride and base alignment; new-token and masked rows are excluded. Prefetch completion is not a correctness dependency. Benefit remains a timing hypothesis: request overhead and queue pressure may outweigh latency hiding.

Balanced fl8 determines a keys-per-CTA target c by binary search such that sum_b ceil(length_b/c)<=G, where G is the occupancy-derived grid size. Its lower bound ceil(max_length/NS) ensures each request's actual nsp<=allocated NS. Prefix sums map CTAs to request/split; excess CTAs return. With these large shared-memory footprints, occupancy gives132 resident CTAs, allowing all final-arrival reducers to remain co-resident. Every request retains split zero for the new cache write.

## If these variants lose

Do not assume lowering RBF yields two CTAs/SM. RBF3 still needs116992 B before allocation granularity/reservation, slightly more than half of228 KiB. Removing the2048 B gate staging region would fix shared capacity, but baseline157 registers/thread at256 threads requires40192 registers/CTA, exceeding the65536-register SM budget for two CTAs. An occupancy experiment needs both a smaller staging footprint and a compiler-proven <=128-register allocation; launch_bounds alone may spill. This is a materially different mechanism but is not currently a proven speedup.

Primary architectural references: [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.3.0/hopper-tuning-guide/index.html), [PTX async bulk prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch).
