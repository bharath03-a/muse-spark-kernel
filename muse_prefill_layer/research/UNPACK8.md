# Lossless eight-operation W4G unpack

The promoted grid-P prefill kernel still uses the old 11-integer-operation FP4 bit extraction. Upstream `docs/fp4_dequant_survey.md` at pinned commit e6b48c8 gives an eight-operation placement: direct mask, rotate-left 10 plus mask, rotate-left 13 plus mask, and masked multiply by `0x1084` plus mask. The four BF16 scale multiplies remain unchanged.

`build_unpack8.py` creates isolated `candidate_prefill_unpack8.py`. Only the W4G CUDA consumer, its extension name, and its private prepare-time code placement change. Existing Q-ring/native packing stays unchanged. The `_w4pack` function is structurally identical except for its placement helper, so scale folding, zero handling, per-16 block-scale ownership, fused W13 row pairing, and native fallback remain intact. All three modes share the modified W4G kernel templates; no dispatch threshold changes.

`prove_unpack8.py` passes:

- A bijective 32-bit code placement, with linear paths proved on every basis vector and the multiply path checked on all 256 possible masked inputs. This covers all 2^32 words.
- All 1,904 combinations of FP4 code and finite positive normal E4M3 scale yield exactly BF16-representable products. No new rounding is introduced.
- Unique `(row, k16 group)` scale tags across all 64 stages, 16 physical row warps, 32 lanes, and every FP4 pair and half. The A/B/A/B scale assignment remains tied to the row, independent of the K-half.
- Negative scales still fold into FP4 signs. Zero scales still canonicalize codes to zero. Subnormal scales still reject the fast bank, retaining the original fallback behavior.

This proof covers algebra and packing indices, not GPU correctness or performance. The existing paired W13 row bijection proof covers BM64/128/256. A later approved GPU test must include varying scales for every block, zero-route CSR offsets, long-tail and uneven expert counts, fused and unfused W13, W2, all three modes, and fresh graph replay checks.

Measured W4 consumer dequant spans are approximately 25–27% of consumer-stage cycles. Cutting three operations from a 15-operation extraction-plus-multiply sequence suggests roughly a 5% W4 upper opportunity if that fraction shrinks proportionally; overlap and issue mix can make the realized result smaller or larger. W4 accounts for about 43–51% of weighted prefill profile time. This is a plausible few-percent whole-layer change, not a demonstrated speedup.

CPU-only resource comparison: app `ap-UNm6Xq3OreTi3CE9y3G8sF`, session 56224, log `/tmp/prefill_unpack8_cpu.log`. No GPU is allocated. Do not promote before numerical and paired timing evidence.


## CPU compiler result

CPU compilation completed in 68.77 seconds. Registers remain n32=62, n128=96, n256=168. All kernels retain zero stack/local allocation, zero spill instructions, and static shared memory of1,024 bytes. Full-kernel static instruction counts decrease n32 2,136→2,112, n128 2,808→2,792, and n256 3,712→3,688. HMUL2 remains24 in every kernel, confirming unchanged BF16 scale multiplication. The resulting SASS includes rotate SHF operations and the `0x1084` multiply. Static instruction counts include queue/epilogue/compiler scheduling and are not dynamic performance estimates. Details are in `unpack8_resource_summary.json`.

No GPU test has run. The remaining GPU budget is insufficient; preserve this as a ready research candidate rather than spend an official check without component validation.
