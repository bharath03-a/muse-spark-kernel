# Openkernels reference experiments

Source: https://github.com/lee101/openkernels
Pinned revision: `d2057db2618075e960b7b8465180729bd87ccc38` (Apache-2.0, copyright 2026 Lee Penkman).
Full license preserved in each affected challenge directory as OPENKERNELS_LICENSE.txt. Current submission.py files remain unchanged.

## Why this source changes the approach
The repository implements exactly these Muse challenges. Its custom CUDA attention reads scattered FP8 pages directly and fuses preparation/output normalization; its persistent MoE kernels fuse route partitioning, SwiGLU, expert normalization and summation with NVFP4 repacking and streaming. These are materially different from the earlier scheduling-only experiments.

## Checks
| Candidate | Purpose | OKC check |
|---|---|---|
| muse_attention_decode/candidate_openkernels_attention.py | Pinned attention backend, standalone comparison | 59cbe07c5b5941a3 |
| muse_moe_w4a16/candidate_openkernels_moe.py | Pinned persistent MoE backend, standalone comparison | e591e5372bbb4a1a |
| muse_decode_layer/candidate_openkernels_attention.py | Integrate attention with our original projections/MoE/norms | f1e076dca5db4640 |
| muse_decode_layer/candidate_openkernels_fused.py | Integrate attention and persistent MoE with our projections/norms | fba499920caa4ef9 |

All four compile as Python; judge numerical correctness, compilation/preparation bounds and speed are pending. Do not treat upstream README results as our scores.

## Combined integration details
- Static identifier prefixing embeds both components into one submission; no runtime sibling imports.
- Whole-layer QKV projection output is reshaped to the standalone attention contract, then flattened for our O projection.
- Persistent MoE receives already-rotated local IDs with local_start=0. Nonlocal IDs are negative and rejected by the unsigned local-expert range check.
- Weights retain NVFP4 and BF16 representations. The source repacks weights during preparation.
- Removed original Marlin bank/module preparation and old route alignment in the combined candidate because persistent MoE performs partitioning itself.
- Existing residual/norm/routing and dense-projection paths retained for an isolated backend comparison.

## Other confirmed results
- Compact MoE local alignment e37b8df3cb654c7a: correctness pass,10821.57us vs best10712.04us; no promotion.
- Whole layer packing-only3e54097dbc934eba: correctness pass,18141.96us vs best18147.76us; negligible change, no promotion.
- Modal cp14 v1 probe exceeded420s during preparation; no numerical result. Skipping redundant old dense work alone does not yet prove300s preparation bound.
- Tuned Marlin wrapper GPU probe failed on include path; source path subsequently corrected by agent but no successful sweep yet.

## Coordination state
Three subagents reported account usage limits and stopped. Do not assume their final proposed CPU precompile or XQA experiments were launched; inspect handles before resuming. Root continued source audit/integration and launched the four checks above.

## Persistent CPU compilation
Launched attention-only compile on Modal CPU (8 cores,16GiB,1200s bound), app `ap-ErEa4NcDodYZ51PB1ZsxOG`, local CLI session35743, log `/tmp/openkernels_cpu_compile.log`. Runner `modal_research_compile.py` persists PyTorch, Triton and TVM-FFI caches. Harness-only architecture helpers select H200 kernels without querying a GPU; candidate source remains unchanged. Completion not yet verified. MoE CPU compilation function exists but has NOT been launched.

## New upstream snapshot for the next research iteration

Read-only remote verification found HEAD at `47103fd9c1212868018eb615a4c6467e366e9ce0`, substantially newer than our existing candidates' `d2057db2618075e960b7b8465180729bd87ccc38`. Fetched public origin/master and created a separate detached research checkout at `/tmp/muse-openkernels-47103fd`. Original research checkout and all scored sources remain intact.

New source contains attention v75 (batch-one sliding single-kernel cluster path), MoE v49–55 research, whole layer v33, and prefill pf24–32 (paired SwiGLU, chunked remainders, persistent/PDL paths). Public scores/comments are unverified claims for our account; they guide experiments, not promotions. Apache-2.0 attribution must identify the actual revision used by each new candidate.

Experts are auditing numerical boundaries and portability before any candidate is checked. Preserve our strict BF16 gate/up/down/effective norm handling, complete-K W2, eager small-shape fallback and source-to-job hashes. Keep BF16 projection ownership with the other chat; whole-layer integration should use vetted component snapshots without editing that folder. New upstream docs also contain failed experiments and issue/roofline analysis; consult them before repeating mixed-scale or producer-dequant work. Unrelated production stack notes are outside this task.
