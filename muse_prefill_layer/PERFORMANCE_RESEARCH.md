# Whole-layer performance model and peer research

Source basis: pinned openkernels revision d2057db2618075e960b7b8465180729bd87ccc38; local TASK.md and saved judge profiles. Findings are models and source audits, not new measured scores.

## Score mechanics and target

Whole-decode takeover best 18,147.76 us versus then top50 11,059.14 us requires a 39.1% latency reduction, or 1.641x speedup. A2x component improvement can achieve that only if that component occupies at least78.1% of the weighted critical path. No single norm, SwiGLU or schedule metadata change has that profile share.

Every load counts equally in the log of the score. Within a load, multiply each case latency by its task weight before estimating a gain. Whole decode has six loads; prefill has three. Absolute total GPU time across all cases is not the score.

From whole-decode profile24236da7d52a4240 (18,286us, close to takeover best):

| Load | Weighted latency us | Full-context share | Current / combined ideal floor |
|---|---:|---:|---:|
| decode1 | 8300.9 | 27.2% | 3.26x |
| decode16 | 15515.2 | 39.2% | 2.71x |
| decode32 | 19230.3 | 48.6% | 2.66x |
| verify1 | 11617.6 | 27.5% | 2.96x |
| verify16 | 29347.9 | 32.7% | 3.32x |
| verify32 | 42059.9 | 36.1% | 3.86x |

Halving the ENTIRE full-context decode32 case reduces its load by24.3%, but whole score only about4.8%. Halving only attention saves less. Multiple complementary gains across loads are required.

Prefill profile610d33044d004ff7: full-context cases account for32.0%,33.4%,29.8% of turn1,turn4,first-turn load latency. The138x sliding16K shape dominates the first-turn load. Improving only long full attention does not solve prefill.

## H200 roofline

TASK hardware constants:989TFLOP/s BF16 and4.8TB/s; ridge point about206FLOP/byte. Dense layer weights total94,371,840 BF16 values above8tokens, about188.7MB cold traffic, or39.3us ideal. At<=8tokens sliced unpool reduces this to130.0MB, about27.1us. Dense weight-only arithmetic intensity is approximatelyT FLOP/byte, making small decode BW-bound,128tokens mixed, and256+ increasingly compute-bound. These ideal floors exclude launches, input/output traffic, router, experts and attention.

One4096wide expert has50,331,648 FP4 weights across both GEMMs, about28.3MB including one FP8 scale per16weights. Ignoring activations, intensity is about3.56r FLOP/byte for r routes to that expert; crossover is roughly58routes. At753tokens with roughly one local route pertoken, average~23.5routes/expert remains BW-sensitive, with popular experts crossing into compute-bound. At16K average~512routes/expert, grouped BF16 MMA throughput and dequantization scheduling dominate.

Materializing all32 experts as BF16 writes~3.22GB then reads~3.22GB. That traffic alone costs>=1.34ms at peak BW, plus compressed reads and scale/decode work. This is why on-the-fly W4 grouped WGMMA is a serious16K target, while dequant+groupedBF16 can still outperform a weak unpack pipeline. Upstream retains both and tunes by token count/expert skew.

## Fusion and streams

Baseline prefill expert GEMMs consume~65–70% on sliding shapes; separate activation is only~1–1.5%. In-register SwiGLU can additionally avoid intermediate traffic, but its visible kernel share does not justify a promised2x gain.

RMSnorm after projection needs a row-wide sum over all output columns. A conventional tiled GEMM CTA owns only part of a row, so a fused epilogue requires cross-CTA reduction or a second pass; simply moving the formula into each tile gives wrong normalization. Fuse producer partial statistics plus a final norm/residual kernel only when avoiding a full intermediate offsets communication and occupancy costs.

Upstream prefill res_a writes normalized h plus one scalar perrow, while res_b reconstructs the earlier residual update from a and the original stream. This trades a few FP32 operations for avoiding a large residual intermediate; validate floating-point order against the residual-update metric. Source alone does not prove a win.

Router and pooler streams share dense compute/HBM. Under coldL2, simultaneous weight reads can serialize at bandwidth; large GEMMs may occupy all SMs. Time integrated graphs, not sum of perstream kernel durations. Source upstream explicitly notes16K W4 wins4–5% in full layer even when its mini-context tuner misranks it by~0.6ms. Tuning must eventually use the full critical path.

## Preparation and portability

cp14 integration candidate_v1 timed out420s on Modal without numerical output. Full weights fit H200; generation0.540s. Cold SGLang Marlin compilation consumed several minutes before cp14 ZIP51.8s and r9 49.3s builds. This is compile failure to complete within probe budget, not numerical failure. v2 skips unused old projection compilation/warmup; still no300s guarantee.

Persist TVM_FFI_CACHE_DIR=/cache/tvm-ffi before SGLang imports; default ~/.cache/tvm-ffi is ephemeral. TORCH_EXTENSIONS_DIR and TRITON_CACHE_DIR do not cover it. CPU precompile should use identical image, architecture, flags and hashes. Judge may have its baseline JIT already cached, so Modal cold preparation is a pessimistic environment, not definitive judge rejection.

Upstream MoE v48 starts tuning only if elapsed<200s and stops tuning at270s, but this happens AFTER repacking and many ThreadPoolExecutor8 extension builds. That does not bound cold compilation. A reduced proven default config family may be required if cold judge compilation exceeds budget. Repacking must remain lossless and shape-general.

cp14 remains H200-specific sm90a with TMA/WGMMA/cluster occupancy assumptions; extension build failure falls back to Triton in several paths. CustomSlice and compressedBF16 self-tests protect mathematical correctness; they do not demonstrate end-to-end speed. Compressed paths should keep lossless escape verification and fresh-input execution.

## Peer review

Attention expert isolated exact ALU E4M3-to-half conversion (ACV1024), exhaustively proven for254finite encodings. It may remove repeated F2FP conversion without approximating exponentials or adding PCV producer synchronization. Keep it separate from scheduling changes and validate whole-layer impact.

MoE expert identifies already optimized packed weights/scales, rotation-free decode and4608B TMA rings. Older fake-dequant experiment only~12% faster; generic unpack rewrites have limited upside without a stronger measured bottleneck.

Activation-contract issue to resolve before promotion: upstream v48 directly applies SwiGLU to scaled FP32 GEMM accumulators then rounds inter toBF16, while baseline stores gate/up inBF16 first. Higher precision is not a precision reduction, but TASK literally says BF16 wherever baseline keepsBF16. An explicitBF16 gate/up rounding variant would mirror baseline contract; correctness alone does not resolve rules interpretation.

No new GPU/check runs were launched for this research phase. Queued upstream whole-layer and prefill checks remain the next source of performance evidence.

## Follow-up boundary and compilation audit

Independent AST audit of new MoE BF16 variant:9SwiGLU sites replaced across6CUDA source strings (CUDA1,L2,F2,G1,K2,X1). Each has exactly one helper. Gate/up are rounded after global scales, then converted to FP32 for fused SiLU/product, with original BF16 product storage. This matches SGLang default activation kernel kRoundActivation=false: no extra BF16 SiLU intermediate should be introduced.

Prefill queued upstream candidate has the same FP32 gate/up boundary in Q_CUDA ring path near line11081. Its W4 and big paths explicitly materialize BF16 Hh and then act2, so those paths already match gate/up storage boundary. Do not blindly alter every expf occurrence: attention sigmoid, routing sigmoid and alreadyBF16swg inputs have different contracts.

Prefill leanHopper preparation still builds PF2/LT/ringQV/W4, repacks, and prepares Marlin bank; parallel compilation starts before the tuner budgets. Cold startup needs CPU-precompiled exact-source cache or a smaller proven default family. No safe300s guarantee follows from tuner guards.

Quantified residual materialization bound: omitting one FP32 stream write+read at16Kx8192 saves~1.074GB, or>=224us at peakBW; at a~17ms first-turn sliding shape this is~1.3%, and confined to first-turn it gives only~0.4% geometric problem gain. Worth combining with larger expert gains, insufficient standalone top50 strategy.

Attention peer subsequently measured exact ACV and producer PCV: both pass12checks but lose. B16verify current135.25us versus PCV157.46/ACV143.55; B32 current243.04 versus PCV285.79/ACV261.86. Reject both. This supports a latency/occupancy dependency model rather than optimizing conversion instruction count alone.
