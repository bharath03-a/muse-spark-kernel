# Current whole v35 precision-boundary audit

Audited source: `muse_decode_layer/candidate_v35_wxpin32.py` (scored job5ab92e7e91d64475). This file is unchanged by this audit.

TASK.md line103 requires BF16 wherever SGLang keeps BF16. Baseline Marlin materializes BF16 gate/up before fused SiLU/product, materializes completed BF16 down output before post-expert norm multiplication, and prepares the effective norm weight as BF16. These are the explicit boundaries retained in the independently checked strict-v56 backend.

The current v35 embeds raw higher-precision/factored MoE math across these boundaries:

* CUDA_K string begins physical line9128. CUDA lines434/479 evaluate SiLU directly on scaled FP32 gt/up. CUDA403 factors W2GS into route cf; CUDA580 factorsGS² into norm. Completed down output is not rounded to BF16 before norm. These are live small-M GK tuning candidates.
* CUDA_X string begins physical line10025. CUDA260 evaluates SiLU directly on scaled FP32 gt/up. CUDA264 combinesGS and normweight, CUDA270–271 round the already weighted value to BF16 instead of the completed down output before normweight. DL_PIN at physical14111 explicitly activates WX for M32, so this is a live path.
* Effective nw preparation at physical10854 is FP32(1+w), without the baseline BF16 effective-weight materialization.
* CUDA/base,ring,G also omit established strict boundaries. They remain available tuned/fallback paths.
* CUDA_F contains the same direct FP32 SiLU and factored down math. Although upstream retires FS initially, physical14107 re-enables FS(1,3) for tuning. A strict repair must patch it completely or remove it from reachable dispatch. Preserving unmodified FS as a tuning candidate would leave the repair incomplete.

Passing numerical checks does not demonstrate compliance with the explicit activation contract. No judgment about intentionality is made. No submission, withdrawal or scored-source mutation was performed by this audit.

Repair feasibility: all old-v56→strict-v56 context hunks match embedded base,ring,G,GK,WX exactly once, including v35's independent PDL wrappers. Replaying those precise hunks preserves v35 glue/PDL and the shape-only WX32 pin. Effective nw requires one prepare expression edit. FS should be disabled from tuning/force/fallback unless separately repaired and checked; the verified strict-v56 implementation does not use FS. Root owns the withdrawal/promotion decision and official checks.

Audited source SHA256: `197896a0b019aad5c5cb3429114c06e4eb3a983127b62bbcd76568aca08cbfdf`.

## Separate repair prepared

Candidate `muse_decode_layer/candidate_v35_wxpin32_strict.py`, builder `research/build_v35_whole_strict.py`. The original audited source remains unchanged.34exact context hunks replay previously verified boundaries into base/ring/G/GK/WX while retaining their PDL additions. MoE extension names are unique. Effective normweight now rounds BF16 in prepare. Retired raw FS is removed from tune candidates, and any forced/stale FS configuration routes through strict shape-only default. The default is always v17/ring, so no recursion to FS. WX32 pin is unchanged.

Canonical expression for every repaired live path is g=BF16(GS1*sumKgate),u=BF16(GS1*sumKup),inter=BF16(silu(FP32(g))*FP32(u)); q=BF16(GS2*sumKdown), weighted=FP32(q)*FP32(BF16(1+w)), followed by the existing FP32 norm and route reduction. Split-K partials stay FP32 until their complete reduction; GS applies before each stated BF16 materialization. Gate/up BF16 boundaries match fused SGLang SiLU inputs; no separate BF16 SiLU tensor is introduced.

Builder proves all non-MoE top-level AST unchanged. This includes all frontend/projection/attention/glue functions, prepare wrappers and shape-only WX32 pin. Independent attention specialist peer review requested. No Modal GPU run or official check launched by this agent.

Repair candidate SHA256: `8dc4c65ec5facdd12dfe345fab90a85ccd2324370c0a79a8f2a6d84cb0815140`.
