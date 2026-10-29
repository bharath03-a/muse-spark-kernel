# CW8 shared-metadata experiment handoff

2026-10-28. No GPU/Modal/OKC jobs by this agent. Canonical cp17 untouched.

## Artifacts

- `adaptive_record_async_warps_meta.py`: generic CW4/CW8 kernel, shared metadata enabled by constexpr; global-metadata ablation retained. SHA256 `a6b09c96916b549dadb9dfc99aea4857b5d35ce7d8a5411ae95adbc7b19cadb2`.
- `adaptive_warps_meta_compile_only.py`: offline compiler/PTXAS/SASS driver; no tensor allocations or GPU launch.
- `adaptive_record_warps_meta_mapping_test.py`: executed with system Python; PASS.
- `top50_candidate_cw8_meta_pooler.py`: self-contained exact cp17 prefix; SHA256 `8c943911bf27e71f7713466e18541fece2064aa2c7fc5f5cab96388e378f1b1c`.

## Isolated mechanism

Measured predecessor CW8 improved its family from24.096 to22.304us but did not beat canonical22.208us. This experiment preserves its codec, payload, R32 mapping, eight K partitions, shared X and depth3. It preloads exactly256 uint32 metadata records (1024B) per pooler CTA once, rather than putting a dependent global metadata load on each async issue.

Preload logical coordinates [step,record] use original `_ac_pack` row16/K64 order:
`rid=(n0//16+record//CW)*(K//64)+(record%CW)*KI+step`.
An unswizzled [KI,2*CW] allocation is reinterpreted as [KI,2*CW,1], following the previously compiled ac2 descriptor pattern. Each issue reads `.index(min(step,KI-1)).load(cl)`; pred=false then zeroes metadata for final masked copy groups. One prologue CTA barrier publishes metadata. Existing per-iteration barriers and final async drain remain unchanged. No decoded weight shared stores or new numerical format.

## Verified source/CPU evidence

- Candidate begins with exact bytes of canonical and cp17 checkpoint, both SHA256 `d2c3272138c899a9643d42c58b1291680bccf99cc54a2dbf8e7e3110b091b85f`.
- Normalized AST identity between inlined candidate and generic issue/kernel/decoder proved.
- Existing compressed/raw decoder, CW4/CW8 weight coverage and ring proofs rerun PASS.
- New CPU proof covers CW4/8, K1024/4096/8192, two CTA origins, depths2–4: every original metadata record loaded once, shared physical view equivalent, flags/offsets exact, all masked-tail descriptor indices in bounds. Probe, driver, candidate parse successfully.
- Independent peer approved source/indexing/lifetime and wrapper. No code changes requested.

## Parent-provided pinned3.8 offline compilation

Read `adaptive_warps_meta_compile_only.out`:

| Variant | Registers/thread | Shared bytes | Spills/stack | Static global loads | CTA barriers |
|---|---:|---:|---|---:|---:|
| CW8_GLOBAL |128|196800|0|32|6|
| CW4_META |156|99424|0|2|7|
| CW8_META |158|197824|0|1|7|

CW8 metadata lives at shared0x30000 and flags at0x30400, without overlap. Static mma.sync32 and cp.async70 remain unchanged. The metadata variant adds only the prologue barrier; no new barrier inside the loop. Higher register use is a performance risk, though the SMEM constraint already limits CW8 to one CTA/SM.

## Wrapper gates

Outer timer starts before cp17 prepare. Both CW8 GLOBAL and META compile/resource checks, FP64 numerical tests and fresh captured-graph normal/outlier/one-hot/zero tests must pass. GLOBAL is diagnostic. META alone can enable after beating cp17 in two independent31-repetition alternating-order cold-L2 rounds by more than max(0.5us,2.5%). Selection is disabled past260s, with earlier admission reserves. Compile exceptions/numerical rejections print diagnostics and retain cp17. Dispatch restricts pooler/m32, expected N/K, contiguous input and16B alignment; unpooler stays cp17. Fresh output per ordinary call; frozen closures and retained payload state.

This handoff establishes source/API/resource feasibility, not GPU correctness or a speedup. Parent owns any judge decision.
