# pf32 upstream adaptation audit

Pinned upstream: https://github.com/lee101/openkernels/tree/47103fd9c1212868018eb615a4c6467e366e9ce0 . Apache-2.0, copyright2026LeePenkman; full license retained in ../OPENKERNELS_LICENSE.txt. Exact source input /tmp/muse-openkernels-47103fd/muse_prefill_layer/submission.py.

Candidate candidate_prefill_pf32_strict.py is built by research/build_pf32_strict.py. It is isolated from promoted submission.py and pending candidate checks.

## Portable changes versus our earlier strict fused mixed backend

- Paired BF16 vector epilogue stores instead of per-lane scalar stores; XOR4 lane exchange packs adjacent features. CPU research/prove_pf32_epilogue.py proves complete nonoverlapping feature/token coverage for13partialtoken boundaries.
- Programmatic dependent launch enabled only for allocated expert activation rows>=65536 (first-turn shapes), with waits/triggers and launch attributes. This changes scheduling, not work quantity. CUDA graph correctness and fresh-input replay remain necessary.
- W13SwiGLU fusion is already in our backend, so it is not a new independent gain.
- Optional small-tail n8/n16 WGMMA (W4REMN) defaults0; regular big W4 kernels still use the previous widths.
- Persistent W13/W2 overlap across experts is implemented as w4p/mode5, but W4MODES defaults0/1/2; it is not enabled by this adaptation. It needs an isolated ablation.
- W2SS sums-of-squares epilogue defaultsFalse; upstream reports +110usW2 versus -80uscombine, a net loss. We keep it disabled and reject nonempty SS in the strict host interface.
- Cluster multicast variants defaultdisabled (W4MC0) and upstream comments report no win. No reason to spend our checks on them yet.

## Precision and fallback fixes

Raw upstream ringQ and fusedW13 apply SwiGLU to FP32scaledgate/up; we round both inputsBF16 before FP32SiLU/product and finalBF16store. RingQdown outputs are roundedBF16 before effective norm-weight multiplication; the norm then remainsFP32. Effective post-expert weights are roundedBF16 in all prepared states. W4 and big paths retainBF16I andY.

Raw upstream permits BF16half-KW2partials and FP32combine. We remove half-K autotuner choices and enforce complete K reduction in both Python production dispatch and C++W4entry. W2SS is disabled because its sums use unrounded down accumulators.

Raw upstream T<=256 fallback lazily prepares an embedded decode backend inside kernel_fn. We remove that branch and append our statically namespaced local SGLang baseline, preparing its nativeMarlin state untimed. This retains the correct FP32small-router input and <=8token slicedunpool, without runtime sibling imports or input-derived caches.

Production scored prefill uses the FA3PF path, so new standalone attention algorithms embedded in the upstream file are not selected for the fixed scored head configuration. No unrelated attention or denseBF16 optimization is claimed.

## Larger-mechanism interpretation

The upstream gap document is historical: section5 describes an earlier producerSS implementation, while current source is RS. Its updated section7 prioritizes heavy-expert weight reuse, big/remainder load balance and persistent overlap, because small-load tail kernels re-stream weights and big tiles have too few waves. This is stronger motivation than blindly repeating SS.

Their routing analysis suggests about1.8localroutes/token, not a uniform one-route assumption; use actual generated counts for roofs. Proposed SS shared staging still has a passing addressing proof in our research folder, but it is disconnected and unmeasured. Existing W4PROF shows decode dependency time is nontrivial, with producer slack that may or may not cover software decode plus added shared traffic/barriers.

## Verification state

Python compile and pairedstore mapping proofs pass. CPU-only W4backend compilation launched via research/modal_pf32_cpu.py; log /tmp/prefill_pf32_cpu.log. No new GPU/check run was launched for pf32. Compilation alone does not establish graph correctness, preparation<300s, or leaderboard improvement.


CPU compile completed successfully, app ap-yIPfvtpi1mraMLwQTObdVG. All three W4G kernels and persistent w4p have STACK0/LOCAL0. Launch register counts: small67, wide168, standard96, persistent168. This proves the strict C++ adaptation compiles on sm90a, not correctness/performance. No GPU or official check launched by this worker.

## Primary negative evidence for blind flags

Upstream git commit7ea5374 explicitly restores W4REMN0: n32-only uses67registers without spill; mixed n8/n16/n32 uses96registers plus spill. Thus turning W4REMN1 back on is already falsified.

Upstream docs/ideas.md line757 reports persistent mode5/w4p losing despite correctness: T753673versus461us,16K8065versus5436us. Dependency waits116us and idle10% did not overcome chunked n64 MMA throughput/register pressure. PDL chaining helped16K3% but hurtT7536%/T30121%. Do not repeat those unchanged variants.

A DISTINCT proposed n64-only33..64tail class avoids mixed-width compilation: NT64/NCW2/BM128,384threads,5stages~73KBdynamicshared, potential producer32+consumer104 register redistribution permits2CTA/SM. SourceMOE_PROF gives2–3such tails at753 (7.0–10.3%of all padded MMA work removable),3–4at3012(3.2–4.2%), only0.6–1.2%at16K. New kernel could reduce padding and improve waves but doubles some activation-tile reloading and adds a launch. Root allocation requested for one disconnected CPU-only resource proof before any GPU experiment. No new flag sweep launched.
