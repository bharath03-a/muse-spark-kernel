# Whole frontend routing probe (external research only)

`whole_frontend_extract.py` AST-copies the verified `candidate_cp14_attention_strict_moe.py` into an independent research namespace, retains the original17 statements through the route-side-stream join/record_stream, and replaces the expert call and remaining decoder with a return of actual normalized expert input, local IDs, and top-k weights. It omits the embedded MoE body/prepare wrapper. No installed package or official candidate is modified. No harness code should be submitted to OKC.

`modal_whole_routes.py` warms exact CUDA sources/build flags on CPU (sm90a and the shared cache helper), then optionally prepares whole weights once, uses judge `reference.case_fields` busiest-bank selection for each fresh seed, extracts real frontend outputs, clones those outputs before scratch can be overwritten, and times only the two MoE backends with independent mutable scratch. Each pair gets FP32-reference/finite validation and the common balanced AB/BA replay-stability guard. Reports include the32-entry local route histogram and source identity.

Peer review by prefill specialist confirmed stream joins, local-ID rotation, prepare-wrapper retention and cloned-input/state semantics. This measurement removes preceding whole-kernel L2/interference context, so it is evidence for routing-dependent behavior; an integrated official check remains necessary before promotion.

## Routing identity limitation

`okc status 7e7f05d4f8ed46a2 --json` records and current public `okc show muse_decode_layer --json` contain no router_seed, router_load or router_distinct fields. The local newer reference supports them but its `weights_fields` returns{} without targets. The probe therefore defaults to the published weights seed5353's original router and explicitly labels its reports `default5353_surrogate`. An explicit known router seed may be passed to the runner. Do not describe this as an exact reconstruction of a hidden calibrated router seed.

`whole_route_cases.json` preserves exposed original scored benchmark seeds and current public configs. Fresh inputs add an independent seed offset. Cases cover M32 decode32, M128 verify16, M256 verify32, each sliding/full and original soft/agentic settings.

## Commands (coordinate GPU allocation first)

CPU:
```
muse_bf16_proj/.venv/bin/modal run muse_moe_w4a16/research/modal_whole_routes.py::compile_frontend
```
GPU, only after coordinator allocation:
```
muse_bf16_proj/.venv/bin/modal run muse_moe_w4a16/research/modal_whole_routes.py::compare_actual_routes --seed-offset 917
```
Evaluate shape-only M256 first: standalone normal M256 had a2–3% gain and heavy routing large gains. Require fresh normal seeds and actual-router cases. M32/M128 shape-only WX remains rejected because standalone normal loses5–7%. A GPU-side histogram-based dispatcher is a separate implementation with timed histogram/branch cost; do not branch on a CPU-read input statistic, known seed, or distribution label.

CPU cache warm completed successfully in104.2seconds, all seven extension builders including slice, shared volume committed. Source SHA256: `6e9384e363de3a56e9944ab101f82e2f0cba86b3396d488f205557ccc9cb3dae`. Log: `profiles/research_whole_frontend_cpu.log`; Modal app `ap-nVzf4MtLmajJlyGpp0lwQR`. Driver-linked CPU imports cannot execute, but completed binaries were checked at the actual `_jit_compile` artifact path and persisted. CUDA builder imports that do not require libcuda loaded successfully. GPU validation remains pending coordinator allocation.
