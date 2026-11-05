# HILLCLIMB_LOG (decode_layer, base candidate_ownproj_v3.py) — 2026-11-03

Harness: scratchpad hc_sweep.py — one H200 container, v3 prepared once, knobs patched as module globals at graph-capture
time; paired interleaved cold-L2 graphs vs unpatched v3 graph, ctx 80000 agentic/soft page=1, seeds 731/1949.
Ratio = base/cand, load ratio = 46/62 sliding + 16/62 full. Every variant passed ref.check_implementation.

## Knob inventory (runtime, no rebuild)
- PCX_CFG[(role,T)] = (P pre-wait stages 0..15, d0, g pacing, e); default P=BFP_PDL_N=2, roles qkv/o/un/pool/router, T<=32
- GLE_RC {1:8} (route carry NCH), GLE_ILV 1 (FP32 router interleave 0/1/2), T=1
- SL_CLUSTER_CONFIG {1:(8,256),8:(16,1024)}, SL_TILE_CONFIG {1:(4,1,T),8:(4,4,T)}, SL_ROUTE_CONFIG (num_warps/parts)
- _GRAFT_ROUTE_EARLY False (T>=16), OKC_W0_L8 1135 (q8 flat attn), OKC_W0_L1/PR 1024, _TUNED7 attn plans
- PDL_MAX_T 128, bfp_SK2_PICK/OWN_*_T/RQ_USE/_GRAFT_LT_PICKS (backend picks: concluded, skipped)
Compile-time (skipped: MoE OKC_* builds, NS/WGSFCI/stream-K concluded in moe EXPERIMENTS_V2; attn SEGK at 95% ceiling).

## Pruning (profile e8c43c677ed04470)
- full attention T>=16: ~4 TB/s, at gather ceiling -> drop. MoE: route/stream-K/NS/FCI already exhausted -> drop.
- remaining: sk2 GEMVs + glue at T=1..32 (~20% over bytes/4.8TB/s floor, 30-50% of decode time) -> PCX/GLE/SL knobs;
  q8 sliding attn (W0_L8); route trigger (all T).

## Noise
Null (identical config, second capture) per load: -0.45%..+0.6% at reps 24; -0.4..+0.4% at reps 120-200.
So the 0.3% gate is at the noise floor; rows below are raw (not null-corrected).

## Pass 1 (reps 24, 111 variant-T): raw seed731/seed1949
    pcx_o_T1_P6                  T=1    1.0057 1.0050 WIN
    slcl_T1_nw4                  T=1    1.0049 1.0069 WIN
    pcx_router_T32_P6            T=32   1.0048 1.0036 WIN
    pcx_qkv_T32_P1               T=32   1.0072 1.0031 WIN
    w0l8_1170                    T=8    1.0058 1.0028 
    pcx_router_T32_P4            T=32   1.0023 1.0092 
    pcx_pool_T32_P3              T=32   1.0020 1.0030 
    pcx_router_T32_P3            T=32   1.0066 1.0020 
    pcx_un_T32_P1                T=32   1.0019 1.0090 
    pcx_o_T16_P3                 T=16   1.0026 1.0019 
    route_early                  T=32   1.0018 1.0028 
    gle_rc16                     T=1    1.0018 1.0057 
    pcx_qkv_T1_P1                T=1    1.0016 1.0056 
    route_early                  T=256  1.0026 1.0015 
    pcx_o_T16_P6                 T=16   1.0022 1.0009 
    null_T32                     T=32   1.0042 1.0009 
    pcx_qkv_T16_P1               T=16   1.0066 1.0007 
    w0l8_1100                    T=256  1.0009 1.0007 
    pcx_o_T32_P6                 T=32   1.0006 1.0011 
    pcx_un_T16_P3                T=16   1.0004 1.0015 
    w0l8_1100                    T=128  1.0003 1.0038 
    pcx_o_T1_P4                  T=1    1.0002 1.0020 
    sltile_T1_nw8                T=1    1.0000 1.0042 
    pcx_o_T32_P3                 T=32   0.9999 1.0055 
    pcx_router_T8_P4             T=8    1.0035 0.9997 
    pcx_qkv_T1_P6                T=1    1.0022 0.9996 
    pcx_router_T8_P1             T=8    0.9996 1.0014 
    pcx_router_T16_P1            T=16   1.0034 0.9993 
    pcx_router_T8_P3             T=8    1.0000 0.9990 
    pcx_router_T1_P1             T=1    0.9988 0.9995 
    pcx_un_T32_P4                T=32   0.9988 0.9987 
    pcx_pool_T8_P3               T=8    0.9986 0.9992 
    pcx_router_T1_P3             T=1    0.9985 1.0030 
    sltile_T8_nw8                T=8    0.9985 0.9993 
    pcx_o_T8_P6                  T=8    0.9985 0.9991 
    pcx_pool_T1_P4               T=1    1.0004 0.9983 
    pcx_router_T16_P6            T=16   0.9983 0.9998 
    w0l8_1170                    T=256  0.9983 0.9997 
    pcx_pool_T32_P4              T=32   1.0005 0.9982 
    pcx_o_T32_P1                 T=32   0.9982 1.0060 
    pcx_o_T8_P3                  T=8    1.0043 0.9982 
    pcx_o_T1_P1                  T=1    0.9981 0.9991 
    pcx_pool_T1_P6               T=1    0.9981 0.9982 
    pcx_qkv_T8_P3                T=8    0.9984 0.9980 
    null_T8                      T=8    0.9980 1.0004 
    gle_ilv2                     T=1    1.0049 0.9979 
    pcx_un_T1_P1                 T=1    0.9978 1.0003 
    pcx_pool_T1_P3               T=1    0.9978 1.0037 
    pcx_un_T16_P4                T=16   0.9978 0.9978 
    slcl_T8_nw8                  T=8    0.9976 1.0003 
    slcl_T1_nw16                 T=1    1.0048 0.9975 
    pcx_o_T1_P3                  T=1    0.9974 1.0000 
    pcx_o_T32_P4                 T=32   1.0003 0.9974 
    pcx_o_T8_P4                  T=8    1.0005 0.9973 
    pcx_router_T32_P1            T=32   0.9976 0.9973 
    pcx_un_T16_P1                T=16   0.9972 0.9980 
    gle_rc4                      T=1    0.9972 0.9973 
    w0l8_1170                    T=128  0.9971 0.9985 
    null_T16                     T=16   0.9996 0.9970 
    pcx_router_T16_P4            T=16   0.9978 0.9970 
    pcx_qkv_T32_P3               T=32   0.9969 0.9983 
    pcx_qkv_T8_P1                T=8    0.9990 0.9969 
    pcx_pool_T32_P1              T=32   0.9969 0.9989 
    pcx_router_T1_P4             T=1    1.0036 0.9968 
    pcx_o_T16_P4                 T=16   0.9982 0.9967 
    route_early                  T=128  0.9975 0.9966 
    pcx_qkv_T1_P3                T=1    0.9973 0.9966 
    w0l8_1100                    T=8    0.9994 0.9965 
    route_early                  T=16   1.0003 0.9965 
    null_T256                    T=256  1.0061 0.9964 
    pcx_un_T1_P3                 T=1    0.9964 0.9977 
    null_T1                      T=1    0.9989 0.9963 
    pcx_pool_T32_P6              T=32   0.9961 0.9963 
    pcx_router_T16_P3            T=16   0.9996 0.9960 
    pcx_pool_T16_P3              T=16   1.0029 0.9959 
    pcx_un_T32_P3                T=32   0.9979 0.9959 
    pcx_un_T32_P6                T=32   0.9957 0.9977 
    pcx_pool_T16_P1              T=16   1.0005 0.9955 
    pcx_o_T16_P1                 T=16   0.9993 0.9954 
    pcx_un_T8_P3                 T=8    0.9953 0.9983 
    pcx_o_T8_P1                  T=8    1.0006 0.9952 
    pcx_pool_T8_P1               T=8    0.9952 0.9994 
    pcx_qkv_T16_P4               T=16   0.9949 0.9952 
    pcx_router_T1_P6             T=1    0.9949 0.9975 
    pcx_qkv_T32_P6               T=32   0.9942 0.9960 
    pcx_qkv_T8_P4                T=8    0.9972 0.9941 
    slcl_T8_nw4                  T=8    0.9956 0.9941 
    pcx_pool_T16_P4              T=16   0.9999 0.9938 
    sltile_T8_nw2                T=8    0.9940 0.9933 
    pcx_qkv_T32_P4               T=32   0.9932 0.9966 
    pcx_qkv_T16_P3               T=16   0.9930 0.9963 
    gle_rc2                      T=1    1.0038 0.9929 
    pcx_pool_T8_P4               T=8    0.9968 0.9927 
    pcx_un_T8_P1                 T=8    1.0003 0.9921 
    pcx_router_T8_P6             T=8    0.9919 0.9972 
    pcx_qkv_T1_P4                T=1    0.9988 0.9917 
    pcx_qkv_T8_P6                T=8    0.9948 0.9913 
    null_T128                    T=128  1.0052 0.9912 
    pcx_pool_T1_P1               T=1    0.9986 0.9912 
    sltile_T8_nw1                T=8    0.9931 0.9903 
    pcx_un_T16_P6                T=16   0.9940 0.9902 
    pcx_qkv_T16_P6               T=16   0.9893 0.9902 
    pcx_un_T1_P4                 T=1    0.9922 0.9891 
    pcx_pool_T16_P6              T=16   0.9933 0.9887 
    pcx_un_T8_P6                 T=8    0.9959 0.9884 
    sltile_T1_nw2                T=1    0.9867 0.9873 
    pcx_un_T8_P4                 T=8    0.9986 0.9856 
    gle_ilv0                     T=1    0.9916 0.9843 
    pcx_un_T1_P6                 T=1    0.9811 0.9957 
    pcx_pool_T8_P6               T=8    0.9824 0.9806 
    sltile_T1_nw1                T=1    0.9684 0.9710 
    

## Pass 2 finalists (reps 120)
    gle_rc16                     T=1    1.0104 1.0038 WIN
    slcl_T1_nw4                  T=1    1.0034 1.0040 WIN
    w0l8_1170                    T=8    1.0034 1.0045 WIN
    pcx_qkv_T32_P1               T=32   1.0037 1.0033 WIN
    route_early                  T=32   1.0029 1.0026 
    pcx_o_T1_P6                  T=1    1.0017 1.0058 
    pcx_o_T16_P3                 T=16   1.0030 1.0016 
    pcx_qkv_T16_P1               T=16   1.0014 1.0026 
    null_T8                      T=8    1.0012 1.0033 
    null_T256                    T=256  1.0028 1.0011 
    pcx_router_T32_P6            T=32   1.0014 1.0008 
    route_early                  T=256  1.0031 1.0007 
    null_T1                      T=1    1.0004 1.0029 
    null_T32                     T=32   1.0004 1.0027 
    pcx_router_T32_P3            T=32   1.0004 1.0019 
    pcx_qkv_T1_P1                T=1    0.9998 1.0040 
    pcx_router_T32_P4            T=32   0.9996 1.0003 
    null_T16                     T=16   1.0009 0.9986 
    pcx_un_T32_P1                T=32   0.9975 0.9999 
    pcx_pool_T32_P3              T=32   0.9959 0.9963 
    

## Stack (reps 200, all 6 loads; GLE_RC16 + SLCL T1 nw4 + W0_L8 1170 + PCX qkv T32 P1 + PCX o T16 P3 + ROUTE_EARLY)
    stack                        T=16   1.0037 1.0031 WIN
    null                         T=8    1.0030 1.0042 WIN
    stack                        T=32   1.0029 1.0027 
    stack                        T=8    1.0020 1.0035 
    null                         T=16   1.0018 0.9999 
    stack                        T=256  0.9996 0.9992 
    null                         T=32   0.9989 0.9984 
    null                         T=256  1.0005 0.9984 
    null                         T=128  0.9992 0.9983 
    stack                        T=128  0.9986 0.9969 
    null                         T=1    0.9959 0.9955 
    stack                        T=1    0.9978 0.9947 
    
Null-corrected stack: T1 +0.05%, T8 -0.08%, T16 +0.25%, T32 +0.42%, T128 -0.15%, T256 0.0% -> geomean ~+0.08%.
Verdict: below the 0.5% gate. No candidate_hc_v1.py built, no okc check. Runtime knob space of v3 is flat.
GPU: 3 containers, ~4.5 min measured + ~3 min prepare/startup (~0.13 H200-h).

## candidate_hc_v1.py (coordinator-approved build)
Per-T code changes (999,936 B): PCX o P3 @T16, PCX qkv P1 @T32, ROUTE_EARLY @T32 only (EARLY=T==32), GLE_RC {1:16}, SL_CLUSTER T1 nw4. W0_L8 skipped.
Modal pair_run vs v3, reps 120, L-weighted (seed731/1949): T1 1.0044/0.9928, T8 0.9957/0.9958 (no T8 change: noise), T16 1.0067/1.0023,
T32 1.0030/1.0040, T128 1.0010/1.0022, T256 1.0014/1.0000; geomean +0.08%. 13 canonical + 48 graph checks passed.
okc check a6cd29524212426c: 2.259x (CI 2.225-2.293), 10.01 ms, 13/13 passed. v3 check e8c43c677ed04470 was 2.2506x; live 2.2556x -> within noise.

## Gap/overlap audit of hc_v1 (a6cd29524212426c profile) -- 2026-11-03, NO-GO, no candidate built
Wall (event median) vs kernel-sum/GPU-busy per load:
- decode T1/T16/T32 and verify T1: wall < busy (PDL overlap already absorbs launch gaps; nothing idle to reclaim).
- verify T16 sliding/full: wall-sum = +2.3/+3.7 us (~1%); verify T32 sliding/full: +7.6/+17.2 us (2.5%/3.7%).
  Gaps sit at nvjet (cuBLAS) GEMM edges, which cannot carry griddepcontrol; closing them needs the projection-GEMM swap (NO-GO).
- Concurrency: chain is strictly serial (attn -> o_proj -> route -> moe_fused w/ shared expert fused). Route needs post-attn hidden; no
  independent branch left for a second stream. _graft_route is 2.8-3.5 us max.
Ceiling: closing ALL verify-T16/T32 gaps = ~+0.64% weighted geomean; realistic <0.3% (< +-0.5% check noise).
Weakest loads: sliding decode T32 1.76x, sliding decode T16 2.05x; both ~52% moe_fused (MoE NO-GO area). No byte-headroom work needed.
Checks used: 0. Modal: 0.

## Workload sweep: moe_fused + sliding attn per-T knobs at decode T16/T32 (base hc_v1) -- 2026-11-03, NO-GO
Harness: scratchpad wl_sweep.py (hc_sweep + per-T moe ext swap via state["ma_ext"] + nsm patch), cold-L2 paired vs hc_v1, reps 30, seed 731.
Ratio = base/cand; null 0.994-1.002. All builds compiled (33, 197 s).
- T16 MID (moe_helper0df): WSBLK1/3/4, WSBL0/1, FINPF0, CPF1, NACC2/8, L2H0, TDYN1, NS2/4, WSNR2, UWARP1 -> 0.984-1.004 (best wsblk3 1.004/1.001, cpf1 1.003/1.000).
  NW16 FAILS ref check. T16->b4 build 0.949/0.967. Persistent grid nsm 128/120/112/96/66: -1%..-26%.
- T32 b4 (WGS+B4FIN): WGSGS1, WGSNX2/4, GATHB4/16, WGSPR0, WGSREORD0, NS2/4, L2H0, MFIN0, B4L1, WGSDFIN0/3, B4FIN off -> 0.979-1.004
  (best gathb4 1.004). T32->MID build 0.920.
- Sliding attn _TUNED7 B16 n_splits 8/12/24/32, nc2, coop; B32 n 4/6/12/16, nc2, coop -> 0.972-1.000 except B32 n12 1.0065 (single seed).
Pass 2 (reps 120, 2 seeds) did not run (Modal function stopped at start). Every pass-1 winner <= +0.65% on one layer, i.e. < 0.3% weighted geomean.
Verdict: knob space at T16/T32 is flat; no candidate_wl_v1.py built. Checks used: 0. Modal: ~0.13 H200-h.
candidate_wl_v1.py (hc_v1 + _TUNED7 B32 q1 sliding n_splits 12, MID WSBLK 3, b4 GATHB 4; 999,828 B, one history comment trimmed):
okc check 8a7749d9b3374694: 2.243x (CI 2.210-2.277), 10.08 ms, 13/13. vs hc_v1 a6cd29524212426c 2.259x / 10.01 ms.
Per load vs hc_v1: dec32 sliding 148.2 -> 143.5 us (-4.7), dec16 sliding 123.2 -> 122.1, dec32 full 298.5 -> 294.7; verify
loads +2..+7 us (untouched paths: noise). Re-checks wl/hc/wl/wl deduped (jobs 281f8586, c98fd22f, e3ac315b, 6617d7ec). NO-GO.

## dw round 2026-11-04 (Modal whole-pair, hc_v1 base, 60 reps x 2 seeds, canonical+graph checks pass)
- candidate_dw_v1 = hc_v1 + _TUNED7 (32,1,True,1) n_splits=12 (isolated from wl_v1; wl scored-submit f56a40da = 2.228x confirmed dead).
  T32 sliding ratio 1.0067/1.0069 (both seeds); untouched T32 full 1.0000/0.9995, T16 full 0.9997/1.0001. Real ~-0.9us on dec32 sliding, ~+0.07% geomean.
- candidate_dw_v2 (+ (16,1,True,1) n_splits=8): T16 sliding 0.9952/0.9922 -> REJECTED.
- candidate_dw_v3 (ns16 @32): T32 sliding 0.9710/0.9809 -> REJECTED. ns12 is the peak.
- dw_v1 trimmed to 999,983 B for the 1,000,000 B limit; okc check pending.
- dw_v1 okc check ec7c9d7faa0542c2: 2.241x (CI 2.208-2.275), 10.09 ms, 13/13. Below live 2.2556x; judge draw noise dominates the Modal win. NOT submitted.
- candidate_gle8_v1 (dw_v1 + GLE_T[8]=True, GLE_RC[8]=2): Modal pair run fails RuntimeError 'gle_pr: shapes' (same as 2026-11-02 ours_v1). GLE fused pooler/router hardwired to T=1. REJECTED, no fix attempted.

## Verify-load sweep (T8/T128/T256) on wl_v1 -- 2026-11-04, NO-GO
Harness wl_sweep.py (+ tiny2 base), variants scratchpad wl_vv.json (50: _TUNED7 verify q8 sliding/full n_splits/nc/coop, tiny2 T8
WSBLK/CPF/NACC/NS/L2H + mid build at T8, main-build NS/L2H/WGSNX/GATHB/MFIN/WGSGS/WGSPR/NACC at T128/256, nsm 128/120/112), reps 30, seed 731.
Function hit the 2400 s timeout after T8 + T128 sliding (T128 full, T256 never measured); B4FIN at M=128 errors (as designed).
Null T8 sliding 1.0071. Best: t8_mid (MID build at M=8) 1.0116 full / 1.0107 sliding (99.8 -> 98.7 us, ~+0.4% null-corrected);
t8_WSBLK3 1.010 sliding / 1.003 full. All T128 sliding knobs 0.995-1.003. Nothing >= 1% beyond the null: no wl_v2, no check.
Modal: ~0.67 H200-h (timeout overran the 0.2 cap).
candidate_wl_v2.py (wl_v1 + MOE_TINY2_MAX_M 8->4, so M=8 uses the MID build; 999,861 B): okc check 71e588151fe944ad 2.233x
(CI 2.200-2.267), 10.13 ms, 13/13. verify1 sliding 108.2 us, full 111.8 us. Below live 2.2556x: NO-GO.

## pdl256 2026-11-04 (Modal whole-pair vs dw_v1, 60 reps x 2 seeds)
- candidate_pdl256_v1 = dw_v1 + PDL_MAX_T 128->256. T128 control (no change): 0.9974-1.0017.
  T256 sliding: 1.0050/1.0034 (real, ~-1.3us verify32 sliding); T256 full: 0.9999/0.9998 flat. ~+0.05% geomean stacked.
- dw_v1 scored submit dee0c23944964b83: 2.238x, 10.11 ms; board keeps live 2.2556x.

## inn round (lower precision only) -- 2026-11-04, NO-GO, no candidate built, 0 checks
Scope narrowed by coordinator to FP8 activations / FP8 wgmma (glue and mask ideas belong to other agents).
- FP8 activations into projections/MoE: every T <= 256 load is weight-bandwidth bound. Activations are 16-512 KB vs ~140 MB of 12-bit
  projection weights plus 24 MB per active FP4 expert, so halving activation bytes saves <0.3%. No gain.
- FP8 wgmma for verify MoE (T=128/256): the bottleneck is FP4 dequant integer work (ncu: integer-bound, 151-188 regs), not MMA rate.
  NVFP4 has a per-16-K E4M3 scale and FP8 wgmma has K=32, so the scale cannot be applied on the accumulator. It has to be folded
  into the weight, and e2m1*e4m3 is not exact in e4m3, so the result is lossy. sm_90 also has no hardware e2m1->e4m3 cvt, so the
  LUT cost stays the same. Ceiling: T128/256 MoE is ~2 of 6 loads, maybe 10-15% of their time => <=2% geomean even with a
  perfect kernel. A new FP8 MoE GEMM cannot be built and validated in 2 h with no Modal (the only GPU is okc check).
- Router BF16 hi-only (drop the lo stream): saves 4 MB per step, ~1 us/layer, <=0.7% geomean, below the +-1.5% judge noise. It needs sk2f
  and wide-kernel CUDA edits, and near-tie top-8 flips risk failing the experts check.
- Lossy 12->8/10-bit projection weights: the only lower-precision idea with gain above noise (~4-8%). It quantizes the model's BF16
  weights, which is model modification (judgemaxx risk). It also has e4m3 error ~3.6% vs the ~1.5% rel-L2 limit and routing flips. Rejected.
Verdict: no lower-precision idea clears noise within the time box and rules. Checks used: 0.

## Attention structure audit (mask_v*) on wl_v1 -- 2026-11-04, NO-GO, no candidate built
Checked each idea against the credited attn_kernel (OKC attention, wl_v1):
- Sliding window block skip: already in. Splits cover only [max(0, seq+1-window), seq+L); tiles off the window/causal edge skip the mask (lo_all/hi_all).
- Verify multi-query KV reuse: already in. One CTA holds L*16 = 128 rows per KV pair; verify32 full 191.6 us vs decode32 full 169.3 us (+13% for 8x queries).
- Paged gather: page indices prefetched in 16-byte windows (pidx), with speculative page-table width for B=1; full attn is at ~95% of gather ceiling.
- Gate + rmsnorm fusion: already in every finalize path (rmsnorm * sigmoid(gate) via hardware tanh); no separate gate kernel in profiles.
- Lighter rescale: already lazy (offset moves only when a score exceeds it by 2^8).
Headroom: sliding attn is 5.4-15.3 us (5-7% of each sliding load), latency-bound; a 10% cut there is ~0.3% geomean, under check noise (+-1.5%).
Checks used: 0. Modal: 0.

## Glue algebra/op-merge audit on wl_v1 (f56a40daa3f84287 profile) -- 2026-11-04, NO-GO, no candidate built
Glue kernels per load: T1/T8 glue_cluster_boundary + _gle_route/_sl_tile_route + _sl_cluster_finish; T16 glue_cluster_boundary x2 (6.4+3.2 us,
PDL-overlapped: times include predecessor wait); T32 _graft_row_boundary_pre x2 (8.7-9.9 us); T128/T256 _graft_row_boundary x2 (8.8-10.4 / 13.6 us).
- Row boundary bytes/row (HIDDEN 8192): bf16 branch + f32 stream in, f32 next + bf16 out = 12 B/elem. T256: 25 MB per call in 6.8 us = 3.7 TB/s
  (L2-assisted); T128: 12.6 MB in 4.4-5.2 us = 2.4-2.9 TB/s. Already at the bandwidth floor; math is not on the critical path.
- Folding post/pre norm weights, alpha/beta into GEMM weights: these vectors are preloaded before griddepcontrol.wait (cluster + _pre kernels),
  so folding removes zero bytes from the post-wait path. Gain 0.
- s2/next stream is a layer output; s1 (f32 [T,8192]) could be recomputed in boundary 2 from stream+branch1 instead of written/read:
  net -4 MB of 50 MB at T256 (~1 us of 482 us) plus an extra reduction. < 0.1% geomean.
- Route (2.0-3.6 us) is one launch, latency-bound; top-k/softmax already fused with pre-expert norm. Combine is inside moe_fused (MoE scope).
- No redundant casts/copies found in _graft_after_attention (router fp32 copy only at T<=8, consumed by fp32 pooler/router).
Ceiling of the whole glue scope without removing launches (= boundary fusion, NO-GO) is < 0.3% geomean, below +-1.5% judge noise.
Checks used: 0. Modal: 0.
- candidate_re16_v1 (pdl256 stack + ROUTE_EARLY at T16): T16 sliding 1.0011/1.0011, untouched 0.9969-1.0023. Tiny real win, stacked.
- BF16 restore: candidate_restore_v1 (comment-only) scored e1d9a38d 1.569x; board #423 -> #217.

## last round (router hi-only + verify32 knobs) on wl_v1 -- 2026-11-04, 3 checks, 0 Modal
- candidate_last_v1 = wl_v1 + fused T<=8 router (gle_pr) streams BF16 hi only (FP32 x, FP32 acc; 4 MB less/step); 999,956 B.
  check 9dc299859e674558: 2.270x (CI 2.236-2.304), 9.965 ms, 13/13 (expert selection OK). decode1 sl 66.1 (wl_v1 67.0/69.3),
  decode1 full 73.6 (73.1/74.9), verify1 sl 112.7 (106.3/104.3, noisy), verify1 full 113.2 (115.1/117.6). Router gain within noise;
  decode32 full 275.5 vs 294.7/300.3 is unrelated noise (path untouched).
- candidate_last_v2 = last_v1 + PDL_MAX_T 128->256. check 0671ff624368403a: 2.260x (2.226-2.294), 10.01 ms, 13/13.
  verify32 sl 308.6 / full 474.8 (wl_v1 312.4/305.8, 477.4/481.8): flat within noise.
- candidate_last_v3 = last_v2 + _TUNED7 (32,8,False,1) nc=4. check 8e632ac563004b08: 2.240x, 13/13. verify32 sl 314.6 / full 479.9: no gain.
Verdict: last_v1 2.270x nominally > live 2.2556x but CI overlaps; single-check gain is noise-level. No submit (not asked).

## 2026-11-04 att_v1/v2 (_TUNED7 decode q1 sliding splits)
- att_v1: B32 sl n14, B16 sl n12. att_v2: B32 sl n16, B16 sl n10. vs last_v1 9dc29985 (2.270x; d16sl 122.0, d32sl 143.9).
- v1: d16sl 123.0 (+1.0), d32sl 143.9 (0). v2: d16sl 127.8 (+5.8), d32sl 148.4 (+4.5). No gain; keep n12 / default. Untouched d32 full drifted +13-23 us (noise).
- jobs: att_v1 83340f2348a4424d 2.273x (9953.5 us); att_v2 70db6aae83104419 (10.12 ms).

## proj round (PCX/GLE_RC/EARLY on last_v1) -- 2026-11-04, 2 checks, 0 Modal, NO-GO
- candidate_proj_v1 = last_v1 + PCX o@T128 P3, qkv@T128 P1, o@T32 P3. check 538f65602c444c5b: 2.252x (2.218-2.286), 13/13.
  verify16 (T128) sl/full 243.6/328.7 vs base 245.5/329.2; decode32 sl/full 146.9/292.1 vs 143.9/275.5 (noise). Flat.
- candidate_proj_v2 = last_v1 + PCX o@T128 P4, qkv@T16 P1, GLE_RC {1:32}, EARLY T in (16,32). check 99340bf808344241: 2.246x (2.213-2.280), 13/13.
  decode16 sl 126.7 vs 122.0 (worse), decode1 67.5/73.9 vs 66.1/73.6 (flat). Reject.
Verdict: both below base 2.270x (9dc299859e674558), within noise. Projection config space exhausted.

## T>8 router hi-only + T8 MID on last_v1 -- 2026-11-04, 2 checks, 0 Modal
- candidate_last_v4 = last_v1 + fused pooler+router (sk2f, T16/T32) streams only the BF16 hi tiles (WBYR 2048->1024/warp,
  lo MMA dropped; hi-only stream sliced from the hi|lo router stream). 999,747 B. check 8050c3db5d8f41b6: 2.260x (2.227-2.294), 13/13.
  Fused kernel 17.7/17.1 (T16), 20.5/21.0 (T32) vs last_v1 18.4/17.4, 21.8/20.4 and 18.3/17.7, 20.8/20.4: ~-0.4 us/load, within noise.
- No other T>8 lo/hi reads on scored loads: standalone sk2/wide/gemm router routes only in tune/fallback; pr hi|lo cat only at T128/256 (unscored).
- candidate_last_v5 = last_v4 + MOE_TINY2_MAX_M 4 (T8 MID build). 999,780 B. check 3ac50dd3403141ca: 2.268x (2.234-2.302), 9.973 ms, 13/13.
  verify1 sl 99.0 / full 101.6 (v4 105.8/114.0, last_v1 112.7/113.2): real T8 win; T16/32 loads flat/noisy.
Verdict: last_v5 is ~ last_v1 (2.270x) overall within CI; best verify1. No submit (not asked).

## moed round (TINY/TINY2 builds, decode T=1 / verify T=8) on last_v1 -- 2026-11-04, 3 checks, 0 Modal
- moed_v1 (TINY_MAX_M 0: M<=8 on TINY2). fe17f74ddd0d45d3: 2.235x. decode1 65.3/74.6 (base 66.1/73.6), verify1 109.4/117.6. Flat.
- moed_v2 (TINY WSBLK1+FINPF1; TINY2 NACC2). 5e98e53c0a054c10: 2.259x. decode1 67.0/73.5, verify1 101.0/107.6 (base 112.7/113.2, prior 104-117).
- moed_v3 (TINY NACC2; TINY2 NACC2). 6d202b5104494c4f: 2.249x. decode1 67.0/74.0, verify1 103.0/109.1.
Verdict: TINY2 NACC2 maybe -3..5% on verify1 (2 checks agree) but within run-to-run spread; overall flat vs 2.270x base. No nsm grid knob tried (grid=nsm hard-passed; >nsm risks coop hang).

## brain round (per-load roofline on last_v5 3ac50dd3) -- 2026-11-04, 2 checks, 0 Modal, NO-GO
Model (4.0 TB/s): qkv 37.7 MB=9.4us, o 16.8 MB=4.2us, pooler+router(hi) 71 MB=17.8us, KV full 20.5 MB/seq (B32: 164us), 1 local expert 28 MB=7us.
At roofline: oq qkv 9.6, sk2_gle 17.2, full attn B32 170 (decode) / 187 (verify), MoE T256 200 (~32 experts 906 MB).
Largest gaps: moe_fused decode1 29.7 sl / 22.0 fl vs ~7us ideal (~15-23us, x46 sliding); moe dec32 sl 74 + router/glue (sl dec32 146 vs sol 84.5, -62us).
T16/T32 MoE knobs already NO-GO; untested lever was the M=1 build choice.
- brain_v1 = last_v5 + MOE_TINY_MAX_M 4->0 (M=1 -> TINY2). check 8d6c54a3fe854630: 2.248x (2.214-2.281), 10.06 ms, 13/13. dec1 67.0/73.1 (v5 67.6/74.1), moe 29.7/29.1: flat.
- brain_v2 = + TINY2_MAX_M 0 (M<=8 -> MID). check 690f93c4450b4f58: 2.234x (2.200-2.267), 13/13. moe dec1 sl 22.9 (-6.8) but fl 28.9 (+6.9); verify1 104.3/116.6 (v5 99.0/101.6): worse.
Verdict: decode1 MoE time is routing-dependent (expert count), not build-dependent. Keep last_v5.

## disp round (moe_fused per-T build choice) on last_v5 -- 2026-11-04, 3 checks, 0 Modal, NO-GO
- Map: T1 TINY (already), T8 MID, T16 MID, T32 B4FIN, T128/T256 main. vs 3ac50dd3403141ca (2.268x).
- candidate_disp_v1 = MID_MAX_M 32 (T32 -> MID). aa2fce0d4dbd49e9: 2.237x. decode32 sl 160.2 (+13.8), full 312.3 (+17.4). Worse.
- candidate_disp_v2 = MID_MAX_M 8, B4FIN at 8<M<=32 (T16 -> B4). 9ebb124b1c834408: 2.253x. decode16 sl 122.6 (-2.4), full 205.2 (+0.7). Flat.
- candidate_disp_v3 = B4FIN widened to M<=128. 7da02b66fc1d4a87: setup fail, "B4FIN build is for M <= 32" (hard kernel limit).
Verdict: current per-T dispatch already optimal among existing builds; keep last_v5.

## v8 round (M=8 MID variant + B1 q8 attention) on last_v5 -- 2026-11-04, 2 checks, 0 Modal, NO-GO
- candidate_v8_v1 = last_v5 + own M=8 MID build (key "mid8", WSBLK 3->2) + _TUNED7 (1,8,False,1) n252 nc8 coop1. 999,598 B.
  check a84f87a95ea34dc3: 2.253x (2.220-2.287), 13/13. verify1 sl/full 106.9/113.2 vs base 3ac50dd3 99.0/101.6 (worse); decode1 66.8/73.3 (flat).
- candidate_v8_v2 = last_v5 + own M=8 MID build with -DOKC_NS=4 (WSBLK 3 kept), no attention change. 999,499 B.
  check f551d0ad12764461: 2.230x (2.196-2.263), 13/13. verify1 109.1/115.7 (worse); decode1 68.3/73.8 (flat).
Verdict: both below base; base's 99/102 verify1 may be partly lucky. Third check skipped (1 check/day left). Keep last_v5.

## mid_v1 (MID moe at T128/T256) -- 2026-11-04, 1 check, 0 Modal
- MID capped at M<=16 by DF sentinel rows (INTER_SENT_ROWS 128 = M*8). candidate_mid_v1 = last_v5 + non-DF MID ("@moe_helper0") for M in {128,256}, DF sentinel refilled after each launch. 999,957 B.
- check bbe0c99f236945c6: 1.943x, 13/13. verify16 sl/full 339.1/425.0 (base 3ac50dd3 243.7/329.4); verify32 547.5/712.3 (312.0/487.8). Large regression: SPLIT re-streams weights per <=SK_T job + refill. Dead; keep main at T128/T256.

## stack_v1 (2026-11-04)
- candidate_stack_v1 = last_v5 + disp_v2 (T16 -> B4FIN, MOE_MID_MAX_M 8) + MID NACC 2 (TINY2 NACC 2 dead: TINY catches M<=4 first). 999,832 B.
- check e922ebc01da84d38: 2.245x (2.212-2.279), 10.07 ms, 13/13. vs 3ac50dd3403141ca 2.268x -> no-go.
- Per-load us (v5 -> stack): verify1 sl 99.0->104.4, full 101.6->109.4 (MID NACC2 hurts T8); decode16 sl 125.0->129.3, full 204.5->208.0 (B4FIN at T16 hurts). Others differ only by noise (same code path).
- Modal paired A/B (stack_v.json in scratchpad, wl_sweep on last_v5, ts 8,16) was blocked by permission; not run.

## xf wave 2026-11-04 late (base candidate_xf_wsblk3, MOE_MID WSBLK=3)
- xf check a17949472d134dc0: 2.268x, 9974.9us, 13/13 -> submitted 4f006c820954441e: 2.258x, 10.02ms. Board entry now xf.
- Wave1 checks (13/13 all, none submitted): xa GLE32 2.244, xb GLE8 2.236, xc SLCL8 2.237, xd q8s12 2.238, xe d16s12 2.232, xg GATHB4 2.215, xh PDL_N3 2.255.
- Wave2 checks (13/13 all, none submitted): y1 WSBLK4 2.234, y2 WS+GLE32 2.249, y3 PCXpool32 2.253, y4 PCXun32 2.241, y5 q8s8 2.229, y6 FINPF0 2.246, y7 NACC8 2.257; xs WSBLK3+PDL3 2.236.
- Wave3 Modal pairs vs xf (no okc left): z1 PCXrouter32P6 mixed 0.9965-1.0041; z2 GLE_RC24 prepare-fail (arange pow2); z3 SLtile parts2 mostly <1; z4 ns264 mixed; z5 PCXo32P6 mixed; z6 SLroute8 mixed. No submit.
