# MoE takeover — October 26, 2026

Live board authenticated as `bharath03-a`: rank 775/886, speedup 1.2555x,
score 10,712.04499 us, best scored job `cc47bb2374d74902`.
The quoted 6,615.94 us / rank 145 does not describe this account's best run.
Top 50 at inspection: 2.2531x / 5,968.80714 us. Board positions change.

Current implementation: lossless NVFP4 Marlin packing in prepare, preallocated
scratch/output, expert route alignment, two Marlin W4A16 GEMMs, standalone
SwiGLU, fused post-expert norm + route weighted sum. All weights remain NVFP4.

Profiles of the best checked/scored implementation attribute 83% of bs1 time
and about 92–95% of larger cases to the two Marlin GEMMs. SwiGLU itself is
1–3%, so merely eliminating its launch cannot plausibly deliver the quoted
1.6x overall gain. Marlin already overlaps dequantization with tensor-core
work and repacks weight layouts offline (https://github.com/IST-DASLab/marlin).

## New off-board experiments

- `candidate_compact_align.py`: use actual local expert count (32) for route
  alignment rather than 256; retains the mapping of nonlocal routes to -1.
  Check job `e37b8df3cb654c7a`.
- `candidate_direct_gemv.py`: same route optimization; single-token path
  replaces Marlin with packed FP4 GEMV, parallel full-K reduction and fused
  gate/up/SwiGLU. Retains BF16 dequantized weights and BF16 gate/up/down
  intermediate boundaries; uses Marlin for all larger token counts.
  Check job `7123ab6e42a945a7`.

Both candidates pass Python syntax compilation. GPU correctness/performance
remain pending; do not promote either based on source inspection alone.
`submission.py` has not been modified by this takeover.

## Next priorities

Read the two check results and profiles before spending more checks. If direct
GEMV wins, tune row tile and warp count under CUDA graph replay, then extend
to tiny batches with route-aware reuse. If it loses, focus on the GEMMs'
weight reuse, expert tile padding and unpacking throughput, rather than
standalone activation fusion. Larger batches are the dominant remaining gap.
The previous all-Triton m32 candidate scored 57,087 us and is not a good base.

Modal is installed in `../muse_bf16_proj/.venv/bin/modal`. Its existing app
uses the judge SGLang image and H200. Keep experiments bounded and preserve
cold-L2 CUDA graph timing to avoid selecting misleading eager wins.

## Direct GEMV mathematical audit

For each route the kernel checks `local_start <= expert < local_start +
local_count`; nonlocal routes write zeros without reading weights. Raw banks
retain all global experts, so raw indexing uses the global expert ID. This
is distinct from Marlin's local packed-bank indexing and is intentional.

Packed element k selects nibble `(byte >> (4 * (k % 2))) & 15`, with
magnitudes `(0, .5, 1, 1.5, 2, 3, 4, 6)` and sign bit 8. Row-major scale
index is `row * (K/16) + k/16`. Gate rows are 0:4096; up rows are
4096:8192. Each dequantized value uses per-expert FP32 global scale and FP8
block scale, converts transiently to BF16, then multiplies BF16 activations
with FP32 accumulation. Gate/up accumulations round to BF16 before SwiGLU;
SwiGLU output rounds to BF16 before down projection; down output rounds to
BF16 before existing fused post norm. This matches the required activation
boundaries, while still needing reference tolerance validation because the
reference itself accumulates and retains all intermediates in FP32.

The kernel handles exactly the existing width-4096 architecture. Four rows
per program and grid 1024 cover every output row without masked tails.
`num_tokens == 1` is a public shape specialization. No input-derived state
is saved across calls; no output cache, timing changes or module patching.
Prepare stores both original and losslessly repacked NVFP4 weights, never
persistent dequantized weights. Existing Marlin banks support local_start 0
and counts 32/256 only, inherited from submission; broader arbitrary shard
starts would need a separate prepare change.

## Modal outcome

H200 direct-kernel probe completed successfully at
https://modal.com/apps/bharathvelamala1929/main/ap-hlY7d6GsoQPWCstpRI9cJA .
Three input seeds 7/17/27 passed FP32 reference checking with relative L2
0.5462%, 0.5613%, 0.5393%. Two GEMVs alone took 439.62 / 309.66 / 570.34 us
in cold-L2 CUDA graph replay (includes graph replay overhead). This is far
above historical whole-kernel bs1 Marlin time (~55 us), so the candidate
is rejected for promotion. Same weights/routes Marlin comparison is running
as a separate bounded 300s Modal function; log `/tmp/moe_modal_compare.log`.
The direct candidate is retained as a correctness-tested experimental source,
not an improvement. No scored submissions were made.

CPU Modal inspection of installed Marlin C++ confirms thread_k and thread_n
are hardcoded -1 in the public wrapper (`moe_wna16_marlin.cuh` lines940–944).
The underlying launcher accepts explicit thread_k/thread_n/sms and chooses
from small-batch (128,128,256)/(64,128,128) and large-batch
(64,256,256)/(64,128,128) configurations. A custom independent wrapper could
expose controlled tiling without modifying installed modules; this is a more
promising follow-up than the scalar Triton GEMV. Filtered source was saved
in `/tmp/moe_marlin_cpp.log`. That approach still needs compilation, checks,
and measurements and is not implemented here.

## Independent tile wrapper experiment

`candidate_marlin_tiles.py` now embeds the full independently compiled CUDA
header source, preserving Apache attribution. The C++ copy exports a new
function that adds runtime tile_k/tile_n parameters; all math, quantized
storage, activation precision and kernel launches remain the original
implementation. Installed modules/files are not edited. `marlin_tuned.cuh`
is retained for readable review but is not needed for eventual single-file
submission. The candidate supports EP8 local bank only; this covers the
challenge tests, though full-expert configuration is not supported.

`modal_tile_sweep.py` tests auto,128x128,64x128,64x256 on identical weights
and routes at bs1,32,256, checking every result before graph timing. First
build failed because SGLang DEFAULT_INCLUDE omits its csrc directory; an
explicit include path derived from the installed jit_utils module fixes
header resolution. A second bounded 300s run is pending at
https://modal.com/apps/bharathvelamala1929/main/ap-HaecHcm64GEbRGUVmSMifv
(log `/tmp/moe_modal_tiles_v2.log`). No performance claim is made yet.


## WX52 strict research result

See `research/WX52_EXPERIMENT.md` and `profiles/research_wx52_gpu.json`.20paired cases/40reference comparisons pass; mixed performance means broad dispatch rejected. Normal M256 ~2–3%win; normal M32/M128 regression5–7%. Skewed large gains require actual whole-router validation. Original scored default unchanged by this experiment.

## Actual whole frontend route probe prepared

External research-only AST extractor and paired harness: `research/WHOLE_ROUTE_PROBE.md`. CPU cache warm completed104.2seconds, seven extension builders persisted, no GPU consumed. Verified frontend source hash recorded in docs/log. Actual router seed is not exposed in public job records; default5353 router explicitly tagged surrogate. Peer reviewed stream joins, local-ID rotation and scratch clones. GPU validation pending coordinator allocation; do not submit extractor/harness.

## Strict v35 whole audit/repair (2026-10-28)

Read `research/V35_STRICT_BOUNDARY_AUDIT.md`. Current raw v35 whole source had missing BF16 gate/up and completed-down boundaries plus FP32 effective normweights in live GK/WX paths. Root reports scored job5ab92e7e91d64475 withdrawn and verified strict a698 default restored. This agent did not mutate or submit scored sources.

Separate repair `muse_decode_layer/candidate_v35_wxpin32_strict.py`, reproducible `research/build_v35_whole_strict.py`:34verified precision/grid-P hunks into five embedded CUDA strings, BF16 effective nw, no reachable raw FS tuning/force/fallback. Projection/attention/glue top-level AST and WX32 shape pin unchanged. Attention peer review requested; root owns one official check. No Modal GPU allocation remains; do not launch GPU or duplicate checks.

Prior GK round-store CPU audit finished successfully, both configurations0spills, barrier reduced by1. No GPU measurement occurred before the hard budget closed. Details `research/GK_ROUND_STORE.md`; retain as unmeasured research, no performance claim.

## Strict repair promoted; source-only overlap model

Root reports repaired v35 whole job5648767e139d441b passed13/13 at12501.65us and promoted; stored profile confirms this. Raw5ab92 remains withdrawn. No performance check should reuse the earlier uncorrected raw source.

`research/WHOLE_ROUTING_OVERLAP_MODEL.md` traces the actual router-side/pooler-main fork/join and per-load hidden-window estimates. A last-valid-token-warp metadata epilogue can produce deterministic route tables without an extra launch and without a CTA-barrier deadlock on partial token groups. Attention peer reviewed the publication protocol. Potential producer hiding is supported by source scheduling;4–5us net consumer-setup savings needed for2–3% whole improvement remains unproven. No prototype/check/GPU launched. GPU budget remains held at1782.6/1800seconds; preserve all scored defaults.

## External routing diagnostic prepared (not launched)

`research/build_route_diagnostic.py` + `research/route_diagnostic_runtime.py` emits `muse_decode_layer/probe_strict_v35_route_phases.py`. Original strict-v35 AST remains an identical prefix; all ordinary kernel_fn/source definitions unchanged. Exactly3 incremental diagnostic JIT builds, original normal preparation/tuning unchanged. End-only compact JSON records source-region routing/setup clocks, fixed-seed output identity gates, own-kernel resources and direct main/side ready CUDAgraph event timings. Task's300s prepare budget is retained via255/270/285s gates; no450s prepare workaround. See `research/ROUTE_DIAGNOSTIC_REVIEW.md` for limitations and labeled M256nonN8control. Attention independent source review requested. No Modal GPU/OKC launch; root must allocate any check.

2026-10-28 source-only route diagnostic timer revision: separate compiler-fenced `rd_stamp` for all new GK/WX region/ownership markers; original gtime and barriers untouched. GK ring-end reads new slot16; slots20/21 report per-case median stamp-floor ns. Exact checked strict-v35 text/AST remains prefix; Python/static checks pass; JSON stress1918B. SHA2198731d46b02a9e67e7609962932606e16af4b4a0e20de88c547617794478ba. No launch; focused peer review pending.

Focused attention peer review PASSED timer revision SHA2198731d46b02a9e67e7609962932606e16af4b4a0e20de88c547617794478ba: asm syntax, compiler ordering, GK16 parsing and buffers verified. `stampFloorNs` is an observed control interval, not a GPU retirement fence or subtractable constant. Root notified; no launch.
