# Queue allocation and heavy-first experiments

Baseline: `candidate_moe_full_strict_gk.py`, official check6afd2a4709f54525,
7,763.22us /1.7323x. Its M32 uses v17 and is the least improved shape
(176.19us /1.206x); M128/256 use persistent ring (261.49/334.64us).

`research_moe_queue_atomic.py` changes only three v17 and five ring work-ID
allocations from atomicAdd into explicit relaxed device atom.add. Queue IDs
select work; they do not publish completed activations. Completion/release
counters remain unchanged. Numerical source and BF16 boundaries are unchanged.
`research/build_queue_atomic.py` asserts the inverse source transform exactly
reconstructs the baseline CUDA strings. G1/GK are identical.

CPU source/assembly results: baseline and queue v17 each have134 SHFL,
26 LDL and6 STL. Ring versions each have235 SHFL,5 MATCH,106 LDL and17 STL.
Thus removal of compiler warp aggregation is NOT established by aggregate
SASS: the historical motivation may no longer apply to this implementation.
ATOM count differs46→43 but this counter omits ATOMG and is not a valid
measure of executed atomic operations. Any speed claim requires measured
same-input paired comparison. Complete SASS persists in the shared cache
`/cache/moe_queue_audit`.

`research_moe_heavy_first.py` independently enables the pre-existing HEAVY1
branch in v17. Active experts are ordered by descending token count, breaking
ties by expert ID; each expert's own list, arithmetic, work count and completion
counters remain unchanged. Root authorized this after the upstream diary's
positive v14 heavy-first evidence. It is separate from queue changes for clear
attribution. Only M17–64 dispatch changes; G1/GK and larger ring remain identical.

Runner: `research/modal_queue_research.py`. CPU compile functions are bounded
600seconds. Combined GPU function is420seconds, defaults to baseline/queue/heavy
with a single immutable weight preparation and independent mutable scratch.
It validates finite outputs and FP32-reference tolerance on17 fresh cases,
then uses the common paired harness with24 AB/BA repetitions per comparison,
readonly oversized-L2 flush, replay-stability checks, source hashes and raw
environment/samples. Heavy's unchanged ring shapes are a negative control.

The first GPU attempt was stopped after inspection showed nvcc rebuilding
with auto-selected sm90 instead of CPU's sm90a. No timing result was produced.
GPU runner now explicitly sets TORCH_CUDA_ARCH_LIST=9.0a; CPU restoration is
required because the interrupted build may have changed shared Ninja flags.
No benchmark outcome is available yet. GPU allocation remains coordinated by
root, with attention taking priority before the combined MoE retry.
