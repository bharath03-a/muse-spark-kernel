# BF16 provenance and baseline

## Independent base
`submission_before_public.py` is the independent base. The user confirmed on 2026-11-02 that we wrote it.

- It is byte-identical to `bench/top50_candidate_cw8_meta_pooler.py`. That file is one of our `top50_candidate_*` variants, all of which are in snapshot `f4fed0a`.
- It contains no third-party attribution markers. About 1.5% of its distinct non-trivial lines also appear in the credited files.
- It has these components:
  - CUDA wgmma/TMA/DSMEM kernels: `_build_cuda`, `CUDA_SRC`
  - Triton `_skinny_gemm`, `_bf16_gemm`/`_gemv`, and `_big_gemm` split-K
  - Gluon TMA-ring
  - Stacked hi+lo router: `_router_big`
  - 12-bit lossless zip: `_zip_*`
  - r9 wgmma generator: `_r9_*`

## Credited (not ours)
| File | Contents |
|---|---|
| `candidate_keel.py`, `submission.py` (live) | keell0renz's code, with attribution |
| `candidate_A_alygg.py` | Public kernel, with attribution |
| `candidate_ours_v1/v2.py`, `candidate_v2_rs32.py` | Contain credited code, so they are **derived** |
| `candidate_own_v1..v3.py` | Our base plus our own additions; see OWN_REWRITE_NOTES.md |

## Per-shape baseline (µs, from existing profiles; no new GPU spend)
- Ours: `profiles/submission.1b72808844b144bc` (2026-11-01 16:52, before the public promotion at 22:47), 1.430x overall.
- Credited: `profiles/candidate_keel.e9283643e06e4677`, 1.572x overall.
- I attributed the "ours" profile by timing. I did not hash-verify it. I will re-run the base before any claim relies on it.

| layer | m | sglang | ours | credited | ours/credited |
|---|--:|--:|--:|--:|--:|
| o | 1 | 7.28 | 5.77 | 5.13 | 1.125 |
| pooler | 1 | 21.38 | 13.96 | 13.70 | 1.019 |
| qkv_gate | 1 | 14.40 | 9.47 | 8.88 | 1.066 |
| router | 1 | 10.34 | 3.82 | 3.93 | 0.972 |
| unpooler | 1 | 7.20 | 3.92 | 3.56 | 1.101 |
| o | 8 | 6.98 | 6.04 | 5.62 | 1.075 |
| pooler | 8 | 21.44 | 14.26 | 13.72 | 1.039 |
| qkv_gate | 8 | 14.62 | 9.62 | 9.30 | 1.034 |
| router | 8 | 26.11 | 4.78 | 4.69 | 1.019 |
| unpooler | 8 | 6.78 | 3.98 | 4.21 | 0.945 |
| o | 16 | 6.82 | 6.15 | 5.64 | 1.090 |
| pooler | 16 | 21.44 | 16.77 | 14.18 | 1.183 |
| qkv_gate | 16 | 14.11 | 11.85 | 9.51 | 1.246 |
| router | 16 | 14.82 | 4.97 | 4.68 | 1.062 |
| unpooler | 16 | 18.34 | 15.06 | 14.37 | 1.048 |
| o | 32 | 6.98 | 6.13 | 6.02 | 1.018 |
| pooler | 32 | 21.79 | 17.87 | 14.97 | 1.194 |
| qkv_gate | 32 | 14.42 | 11.94 | 10.32 | 1.157 |
| router | 32 | 15.20 | 4.99 | 5.04 | 0.990 |
| unpooler | 32 | 18.83 | 16.11 | 14.76 | 1.091 |
| o | 128 | 7.26 | 7.22 | 6.59 | 1.096 |
| pooler | 128 | 24.77 | 21.31 | 17.68 | 1.205 |
| qkv_gate | 128 | 16.93 | 14.50 | 12.03 | 1.205 |
| router | 128 | 16.30 | 6.64 | 5.74 | 1.157 |
| unpooler | 128 | 20.05 | 19.03 | 17.32 | 1.099 |
| o | 256 | 8.86 | 8.92 | 8.80 | 1.014 |
| pooler | 256 | 30.94 | 27.12 | 24.42 | 1.111 |
| qkv_gate | 256 | 19.47 | 18.15 | 17.89 | 1.015 |
| router | 256 | 17.79 | 9.73 | 7.07 | 1.376 |
| unpooler | 256 | 24.56 | 24.70 | 23.53 | 1.050 |

sum ours 344.8 us, credited 313.3 us

## Largest gaps (targets)
| Shape | Ours vs credited |
|---|---|
| router m256 | 1.38x |
| qkv_gate m16 | 1.25x |
| pooler m128 | 1.21x |
| qkv_gate m128 | 1.21x |
| pooler m32 | 1.19x |
| pooler m16 | 1.18x |
| router m128 | 1.16x |
| qkv_gate m32 | 1.16x |

Ours is already faster on router m1, unpooler m8 and router m32.
