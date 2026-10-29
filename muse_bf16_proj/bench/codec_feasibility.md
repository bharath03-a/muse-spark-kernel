# Bounded codec feasibility audit — 2026-10-27

Scope: exact BF16 weights, current cp14 register decoder, original normal weight distribution, no GPU jobs or canonical edits. Conclusion: **no proven new codec performance candidate**. The narrower E8/11 representation below passes storage but is unlikely to pass compute. The final addendum proposes a distinct record-fallback scheme motivated by measured large-m correction cost; it remains unvalidated. Do not spend scarce checks without an isolated local experiment.

## Actual baseline

`_zip_pack` stores12bits/weight in MMA-fragment order:8low bits (7mantissa+exponentLSB) and4bits (sign+three-bit exponent-pair window). Decoder uses two LOP3, one SHF, four PRMT per eight weights, then MMA. Exact power-of-two rebasing is undone after accumulation. Out-of-window values use independent sparse correction warp, per-row CSR with64-bit entries (Kindex32+FP32 original32). At reported p=0.00014, payload+entries+row pointers is12+64p+32/8192=12.01287bits/weight. Register decode, larger bulk copies, no-x, wider row blocks and compressed large-m paths already tested; this audit does not revive them.

## One specific alternative evaluated: E8/11 plus compact exact corrections

Prepare chooses eight most frequent **full exponent** values; stores8bits sign+mantissa plus3bits palette index. All8codes are used. Out-of-palette entries receive a placeholder decoded BF16 value and are corrected by original-minus-placeholder products, exactly as baseline. Sparse entries pack a13-bit Kindex and original BF16 bits16 into one uint32; row offsets remain32bit. This supports K<=8192; zero/NaN/Inf handling must fall back or preserve bits, never silently clamp. Weight values remain exact; correction rounding needs the existing one-hot/FP64 tests.

Physical main stream: each warp-owned group of8weights occupies11bytes rather than12; combine groups into128/256B-aligned records and explicitly count padding. Vectorized8weight decode now needs extraction of8three-bit codes from24bits, palette lookup, and moving sign from the low-byte stream into BF16 high bits. The existing nibble-compatible LOP3/PRMT construction no longer applies. There is no claim of equal instruction count or Gluon register mapping.

### CPU synthetic feasibility measurement

Executed /usr/bin/python3, standard library only: random.Random(5151),300000 Gaussian samples scaled by1/sqrt(8192), each rounded first to FP32 via struct, then BF16 round-to-nearest-even using `(bits+0x7fff+((bits>>16)&1))>>16`. This is a distribution study, not the exact torch generator sequence and not H200 performance evidence.

| Palette size | Missing fraction | Kept exponent values |
|---|---:|---|
|7|0.0224067|115..121|
|8|0.0135900|114..121|
|9|0.0086967|114..122|
|16|0.00007333|107..122|

E8/11 storage is11+32*0.01359+32/8192=11.43879bits/weight before alignment. Compared with12.01287 baseline this saves0.57408bits/weight, **4.78%**, not25%. On33554432weights the saving is2.408MB, at4.4TB/s only**0.547us**. Qkv ceiling is0.308us. These are byte-only ceilings before decoder and exception work.

Full pooler exceptions increase from approximately4698 at reported baseline p to456005, about97x. Each requires Kindex/original decode, activation gathers for each token, and FP32 correction accumulation. m16 means7.30million correction products instead of approximately75000. FP32 compute peaks do not model this sparse dependency/gather loop. Faster entry storage does not remove the dominant work. Consequently reject E8/11 as an immediate kernel candidate.

## Why microblock exact fallback does not rescue this cheaply

Using64element tile rows and verbatim exponent bytes avoids sparse FMA corrections, but at p0.01359 the probability of any escape is1-(1-p)^64≈58.3%. Three-bit palette plus raw8bit exponents in those rows consumes approximately8+3+5*0.583+1/64≈13.93bits/weight, already worse than12before offsets/alignment. A16element granule lowers fallback frequency but adds more metadata, offsets, and irregular fetch/decode work, breaking the current uniform aligned fragment stream. Never reserve worst-case16bit slots and call logical payload compressed.

Changing sparse correction to full decoded-fragment reconstruction could avoid sparse FMA, but requires per-fragment exception masks/indices and insertions. At least one additional bit/weight for a dense mask immediately consumes the11vs12payload saving; sparse maps need per-record metadata and prefix/popcount addressing. No measured instruction path establishes this win.

## References and transfer limits

- ZipServ https://arxiv.org/html/2603.17435v1 : fixed-layout fused register decode. Baseline already implements the relevant principle; paper headline throughput is not shape-matched proof.
- Unweight https://research.cloudflare.com/papers/unweight-2026.pdf §§3.3,3.6,8:64element tile-row verbatim escape routing, palette reconstructive WGMMA, and producer/consumer placement. The64element granule matters; interpreting rows as fullK is wrong. Its multikernel Huffman preprocess/cross-layer overlap is not free inside an isolated projection.
- NVIDIA PTX https://docs.nvidia.com/cuda/parallel-thread-execution/ : PRMT, SHF and MMA contracts. No hardware arbitrary bit-deposit shortcut should be assumed for24-to32bit exponent expansion.

## Engineering decision

Preserve current codec. Focus the next authorized GPU work on uncompressed register lifetime/spills and remote partial staging, or distinct W-multicast token-axis reuse. E8/11 cannot plausibly close6.4% aggregate from byte saving alone. Compiler peer review:3bit extraction, sign reinsertion and palette lookup may exceed its5% traffic budget; no fundamental impossibility claimed without codegen.

## Addendum: actual old RS implementation and a materially different exception scheme

`bench/cand_zip128.py:z8_step` already decodes directly into register A for four RS WGMMA k16 instructions. The premise that current MMA2packing necessarily requires a shared-memory roundtrip for MMA3 is false here. Reusing this path without identifying a new cost is a repeated experiment.

Historical v3 paired raw measurements in `bench/profiles/cand_zip128_v3.3ed357b68216473f.log`:

- pooler m128 baseline25.68us, RS full38.85, nofix30.03;
- unpooler m128 baseline23.23, RS full31.39, nofix25.36;
- qkv m128 baseline20.98, RS full30.94, nofix25.02.

This identifies6–9us sparse fix cost in those large-m implementations, very different from small-m z9's0.6us total epilogue/compute ablation. Never transfer the small-m bound to these variants.

**Specific alternative: verbatim16x64 MMA record fallback, not E8/11.** Prepare marks every1024weight record containing any current12bit escape, storing the **entire record** as raw BF16 already permuted into identical A-fragment register order. Other records retain1536byte12bit representation. Compact concatenated records are1536or2048bytes, both512B multiples; uint32 record offset plus a format bit identify them. Shared ring slots reserve2048B, but HBM payload remains variable compact size. Raw fallback loads four16B vectors/lane instead of three and skips decode; compressed record preserves existing7op decode. A warp handles one record, so format branch is warpuniform; all warps reconverge before WGMMA fence/issue. No sparse correction warp, x gathers, correction sums or cancellation formula remain. Exact exceptions are supplied directly to tensor cores.

At reported p0.00014 and independent Gaussian values, q=1-(1-p)^1024≈13.35% records fall back. Expected physical cost=12+4q+32/1024≈12.565bits/weight before minimal final alignment. This is4.6% more traffic than current12.013bits, around0.53us extra per full67MB source matrix at4.4TB/s. In return it removes the measured6–9us correction path in the old large-m RS design. Those old nofix kernels STILL lose to baseline by2–4us, so eliminating fix alone does not yield a performance candidate. Removing fix warps may change occupancy, but that is an unmeasured hypothesis.

Correctness: independently unpack every record and assert BF16 bit equality; specifically one-hot columns covering every escaped record, output FP64 tolerance, outlier activations and changed inputs. Compute original BF16 products with FP32 accumulation then one BF16 cast; no approximate residuals. Handle NaN/Inf by disabling optimized path if numerical contract forbids output nonfinite.

Implementation hazard: producer needs runtime record-size metadata and dynamic mbarrier expect_tx accounting; fixed512byte slot offset arithmetic is insufficient. Increasing shared stage slots33% may reduce ring depth or occupancy, so compare equal shared capacity as well as equal stage count. Full record classification must use actual measured q, not independence approximation. If actual q*512byte added traffic plus metadata exceeds nofix gap, reject before GPU. Compiler peer review requested; no GPU experiment authorized or run.

### Implemented CPU prototype and observed record rate

`bench/adaptive_record_codec.py` is a standalone standard-library pack/unpack prototype. Running `/usr/bin/python3 bench/adaptive_record_codec.py` passes exhaustive65536BF16word coverage in unrebased mode; explicit compressed mixed-sign/palette coverage; mixed raw/compressed; exact normal rebasing roundtrip; nonfinite-rebase rejection; and1,048,576synthetic Gaussian BF16weights. Saved result `bench/adaptive_record_codec_study.json`:146escapes,137/1024rawrecords=13.37890625%,12.56640625bits/weight including metadata, exact roundtrip. This is original-distribution syntheticCPU evidence, not prepared judge weights or throughput evidence.

API `pack_records(words,base=None,rebase_raw=True)` returns payload(bytes),metadata(listuint32),base,raw_exponent_shift,records,raw_records,bits_per_weight. Meta bit31=raw; low31=offsetunits16 (up to34GB). Record starts remain512B-aligned. Raw layout:64B/lane; for each j0..3 one16B vector is exactly the four uint32 MMA A-register operands, so it requires no runtimefragment shuffle. Compressed layout is currentcp14exact3x512Bplanes. Optimized kernel raw operands MUST be rebased by2**(112-2base), matching compressedoperand exponent112..127; final scale2**(2base-112) remains. Prototype guard rejects subnormals/nonfinites/overflow/underflow in rawrecords; caller disables wholelayer if guard fails. General unrebased prototype can still roundtrip arbitrary16bit patterns.

New kernel should use predicated masked raw/compressed loads plus integer select, rather than a weight-dependent Python dispatch or runtime branch. The loop/shape path remains uniform. Compiler peer implements directLDG GluonMMA2probe. CPUstudy does not establish GPUperformance.

Small-m bounds: FINDINGS[B] full/no-fix poolerm1=20.2/20.15rawus, so simply deleting fixwarp offers no demonstrated win there; poolerm16 full21.3/loadsonly20.1, decode20.8 implies roughly0.5us for fix+epilogue together. An added~0.53us traffic can erase these benefits. This codec is most compelling if deleting the ninth warp changes measured occupancy/register allocation, or in large-mwhere historicalfixcostwas6–9us. Those mechanisms require measurement; none guarantees aggregate top50.
