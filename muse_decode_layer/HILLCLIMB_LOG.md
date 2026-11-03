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
