# Expert compiler / instruction audit — 2026-10-26

No new GPU/Modal/judge jobs. Read TASK, cp14, FINDINGS, TOP50_CAMPAIGN and all available latest experiment summaries. No local cached PTX/SASS discovered in workspace. Findings below are static source facts or hypotheses, never measured gains.

## Highest priority: register-source asynchronous DSMEM push

cp14 tma_wgmma_kernel lines380–423 drains MMA, stages accumulators into local shared `mine[token,row]`, synchronizes, reloads float4 in row-major output ownership, scatters into remote recv, synchronizes cluster, then sums. Register-source st.async can bypass local staging/reload and one rendezvous. This is distinct from prior Triton *global-memory cooperative reduction* that lost; it is not the same experiment.

Gluon v3.8 exports `hopper.async_store(dst,value,mbarrier)` and calls `create_async_shared_store` without an explicit source layout conversion. dst must match tensor shape and dtype and requires cluster>=2. This proves expressibility, not absence of lowering shuffles. Primary source: https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py

PTX st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 / .v2.f32 / .v4.f32 are register-source Hopper operations (PTX8.1). Do not copy .b128 from current docs into old assembler: .b128 added PTX9.2. Remote destination and barrier must refer to same peer CTA; self-owned stores use normal local stores. Initial barrier publication/rendezvous and CTA lifetime protection remain mandatory. Source: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async

### Layout is the actual obstacle

cp14 accumulator mapping:
- acc[j][0] -> (row r0, token 8*j+2*q)
- acc[j][1] -> (row r0, token 8*j+2*q+1)
- acc[j][2:4] -> same two tokens, row r0+8.

Current mailbox has contiguous **rows**, whereas per-lane pairs have contiguous **tokens**. Therefore replacing current float4 scatter with a v4 source operation is NOT legal without remapping values. Two distinct prototypes:
1. Scalar st.async.f32 into existing mailbox; no lane shuffle, more store instructions.
2. Receive layout [sender,row,token], v2.f32 direct pushes; owner final stores may need layout conversion to coalesce [token,row] output.

Do not claim both zero conversion and v4 stores. The end-to-end result decides whether mailbox rearrangement helps. Need ptx inspection for st.async, shfl, st.shared/ld.shared and spill ld.local/st.local; measure source-staging removal independently from maxnreg.

A cp.async.bulk shared->cluster push still needs local shared staging. It cannot claim the same eliminated work.

## Secondary: ring asynchronous-group depth

Current Gluon ring prefetches S-1 tiles; after issuing iteration i it waits MMA outstanding<=1 and reuses slot (i+S-1)%S, formerly i-1. That wait is necessary for this indexing. Changing wait1 to wait2 alone creates an input lifetime race.

Legal controlled alternative: outstanding<=Q, prefill S-Q; refill tile j=i+S-Q, formerly slot i-Q. After issuing i, all groups <=i-Q must have completed. Q=2 increases MMA outstanding but decreases TMA lookahead by one stage. Retain final wait0 and dependencies. This is a hypothesis useful only if MMA issue/retirement limits current code; bandwidth-saturated shapes are unlikely to benefit. Do not add stages beyond resource limits. No numeric gain justified.

Primary WGMMA wait/deps contract: pinned source above; PTX https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group

## Register controls

maxnreg is not an occupancy promise. At 160–192KiB shared/CTA, register caps cannot create two resident CTAs/SM. Use uncapped/control variants only where generated spills or measured scheduling change; register allocation granularity means small caps may do nothing. Uncapping and PUSH must be separate experiments (external examples combine changes).

## Do not repeat without new evidence

BN32 mma_v2, continuous cross-tile persistence, stack-router and distributed global partial reduction have actual negative paired measurements in TOP50_CAMPAIGN. Earlier proposals calling these untried are stale. x-ingress is experimentally falsified. No BF16 precision reduction, PDL, or timing tricks.

## Panel consensus

Shared with scheduling agent: direct register st.async is distinct and potentially useful; bulk shared copy does not remove staging. Shared with memory agent: heterogeneous formats must retain fragment layout and count metadata/offset traffic, predication, alignment and exception work. Neither mechanism establishes top50 or any speedup until tested. Keep budget cap and baseline.

## 2026-10-27: clean-slate W multicast static audit

Reviewed clean_slate_weight_multicast.py against pinned3.8 hopper/tma.py, ampere/mbarrier.py, hopper/cluster.py and _layouts.py. All invoked APIs exist. allocate_mbarrier(batch=S) allocates logical [S,num_ctas] with CTA-distributed barriers, one8B barrier/stage/CTA. wd.nbytes_per_cta counts whole replicated W tile; xd.nbytes_per_cta counts token-sharded x tile. MMA output CGA splits token dimension consistently with x and reuses full W in eachCTA. rank2 shared tile layouts with leading ring dimension follow existing cp14 allocation convention; actual backend lowering remains unvalidated.

S4/BK128/BN64 resources perCTA:
- m128/C2:128KiB+32B barriers,32 FP32 accumulators/thread.
- m256/C2:192KiB+32B barriers,64 accumulators/thread.
- m256/C4:128KiB+32B barriers,32 accumulators/thread.
These exclude compiler padding, descriptors, temporaries and spills. Confirm compiled resources before timing. Add preflight assertions stages>=2, K/BK>=stages-1, CTAS in2/4, tile and shared-memory validity; current defaults valid.

wait1+periteration cluster barrier makes prior i-1 operands retire in every peer before refill i+S-1; finalwait0+cluster barrier protects peer shared-memory lifetime. This conservative ring trades coordination overhead for safety; no expected speedup established.

Smallest full-candidate integration: copy cp16 into NEW candidate, append self-contained prefixed kernel/helpers, preserve original prepare and state, wrapper dispatch pooler m256 only first; all other shapes call original kernel_fn. No imports of external bench files in submitted candidate. Matched unicast variant required to isolate multicast from output tiling and cluster scheduling. TMA request reduction is not automatically HBM-byte reduction because same-replay repeats can hit L2. No submission.py edits or GPU work performed.

Pinned sources:
https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/tma.py
https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/ampere/mbarrier.py
https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/cluster.py

### Single-CTA two-WG alternative (peer review)

Pinned test_core.py test_tma_mma_shared_inputs parametrizes Hopper warps=[4,2] (line1010), creates MMAv3 layout (1052), launches num_warps=product(warps)=8 (1086). Thus W reuse across two independent WGs can be expressed with one NVMMADistributedLayout([3,0],[4,2],[16,128,16]) accumulator[64,256], sharedW[64,64], sharedX[256,64] without warp_specialize partitions. The pinned test's N instruction is32; exactN128 configuration still requires compilation, although128 is native Hopper instruction width.

S4BK64 costs160KiB+32B, 64 FP32 accumulators/thread across256threads. Each WG reads sameW and distinct128tokens. TMA can requestW once/CTA; this does not remove two WG shared-memory operand reads. WGMMA waits operate perWG, so a CTA gl.barrier after wait1 and before refill is essential: bothWGs must retire oldslot. Finalwait0 precedes outputstore. No extra producer partition required in first prototype. This avoids cluster synchronization but may spend more CTA registers/instructions than baseline; no gain inferred.

https://github.com/triton-lang/triton/blob/v3.8.0/python/test/gluon/test_core.py

## Adaptive raw records: decode/fragment implementability review

cand_zip128 z8_step has a true16x64 warp-owned1536B compressed record, using3xuint4 perlane from offsets lane*16 + {0,512,1024}. Unlike zip_gluon_dev's jointlyencodedK128 halves, this C++record maps exactly to16x64 and is correct reference for newCPUcodec. Reuse its32lane physicalfragment order when designingGluon load layout; no decodedWsharedroundtrip is needed.

Proposed rawrecords2048B can store4xuint4 perlane in same perlane finalfragment order. Branchless GPU sketch (not runnable standalone):

    meta = load(record_metadata)
    raw = meta & flag_bit
    base = payload + (meta & offset_mask)
    c0,c1,c2 = load(base+lane_vector+{0,512,1024}, mask=not_raw, other=0)
    r0,r1,r2,r3 = load(base+lane_vector+{0,512,1024,1536}, mask=raw, other=0)
    decoded = seven_integer_ops(c0,c1,c2)
    afragment = select(raw, raw_fragment, decoded)
    warpgroup_mma(afragment, shared_x, acc, is_async=True)

Masks must suppress invalidaddresses on bothpaths, including inactive fourthrawload from1536B compressedrecords. Do not unconditionally issue bothpaths andhope select suppressestraffic. No datadependent Python/device controlflow required. LLVM/PTXpredication must be inspected; metadatauniformperwarp helpscoalescing butwarpgroup has4warpflags.

Rawrecords MUST store exact REBASED BF16, since existingfinalscale applies allaccumulations. OldCSR holds original FP32 and addsafterrescale; blindly copyingoriginalrawBF16 would be wrong. CPUcodec must verify invertibility for rawrebasing including zeros/subnormals/extremeexponents; fallback entirelayer if representation inadmissible. Reconstructactualexponentdecode could eliminate rebasing but is distinct and adds ALU.

Prefer16x64 before8x64. Eightrows crosses the perlane row+8 register component, requiring extraoffset/flag streams and reducing loadvector efficiency. Its lowerfallbackprobability is not free. CPUcodec should reportactualfallbackprobability, physicalalignment, metadata andpayloadbytes; do not useindependentGaussianestimate as benchmarkproof.

## Actual offline compiler evidence: adaptive direct and split

Parent-provided adaptive_compile_only.out uses exact Triton3.8/torch2.14 environment, CPU-only sm90a compilation. Direct76registers/no shared/no spill; split136registers/2048Bshared/no spill. These are compiler resources, not performance.

Split PTX audit: the sole loop backedge ends at extractedPTXline660. Hotloop has0sharedstores,0barriers,32mma.sync,14staticpredicatedvectorloadinstructions(two records, each3compressedOR4raw runtime),36scalar global32bit loads:4metadata loads and32Xoperand loads. All16staticsharedstoreinstructions occur in finalepilogue. Label `decoded_weight_shared_stores` is misleading: these are accumulator reduction/layout movement, not decodedweights.

Important: PTXAS `used1barrier` counts the barrier identifier/resource, not invocation count. Actual splitPTX emits31bar.sync instructions after the loop. The generic batch reduction and output-layout transformation repeatedly stagev4chunks through2048Bscratch, synchronizing each chunk.

Concrete nextpatch for source/CPUcompile validation: explicit fullscratch[4,32,32] FP32 (16KiB), storeacc once, one CTAbarrier, load with BlockedLayout([1,1,1],[1,32,1],[1,1,4],[1,2,0]) so batch lives only inregisters; gl.sum(axis0) becomes localperthread addition, output layout SliceLayout(0,redlayout). This replaces repeated interwarp reduction/layout synchronization. More shared capacity is affordable relative to136register footprint; actualcompile stats must confirm no newspill/layoutloops. No speedup claim before measurement.

Second identifiable bottleneck: Xpanel is32scalar packedglobal loads and32liveoperandregisters before first MMA. Subtiling B toK16 could shorten register/address lifetime but complicates Afraction slicing. It is lowerpriority than31epiloguebarriers. No R16 variant authored because its resourceadvantage has not been independently compiled.

### Explicit reduction scratch bank mapping

New separate adaptive_record_split_swizzle.py changes only scratch SwizzledSharedLayout from(1,1,1,[1,2,0]) to(8,2,4,[1,2,0]); adaptive_swizzle_compile_only.py compares both. For scalarstore of a fixed MMA accumulator register, lane g=lane//4 selectsrowg+constant and t=lane%4 selectstoken2t+parity. Plainrowfast layout gives only8distinctbanks across32lanes. Swizzle physicalrow=row XOR (((token//2)%4)*8) gives32distinctbanks for bothparities and rowoffset0,8,16,24. Fixedtoken outputreader row0..31 remains bankbijection. CPUenumeration passes.

vec8means32Bswizzlechunk forFP32, not a requirement to issue a single256bitstore. Actual generatedinstruction addressing/vectorwidth must be checked; this is sourcebankconflict hypothesis, not measured gain. No GPU jobs authored or launched by reviewer.

## Direct adaptive losses and async-byte-ring mechanism

Actual judge timing supplied by parent: direct pooler/unpooler318.72/162.176us; intra-CTA split109.344/58.24us, while canonical~22/21us. Both numerical checks passed and selection fell back. These losses rule out deploying direct synchronous LDG variants; a nicer epilogue cannot close this gap.

PTX shows loop-serial metadata load -> payload address -> packed load/decode ->32scalarXloads ->32MMA sequence, with no software prefetch ring. This identifies a concrete unoverlapped dependency chain; it does not prove cycle-level stall attribution without SASS/profiling. Current saved artifact has PTX but no SASS; new CPUcompile driver emits get_sass output if available.

New source adaptive_record_async_kernel.py uses pinned ampere.async_copy.async_load(smem,pointer,mask),commit_group,wait_group. Fixed2048Bslots permit variable1536/2048B record traffic by predicating fourthvector. W4R32 has8records/stage=16KiB; stageX[4,M,64] adds16KiB atM32. Depth3 Wonly48KiB or WX96KiB. Sharedflags add96B, reduction16KiB can reuse dead buffers after finalwait0 subject allocator. Both controlledSTAGE_X branches retain2CTAbarriers/iteration for sharedflag publication and complete slot consumption before reuse.

Allweights remain packed inshared; decode goes directly into MMAregisters via pureinlineASM7integerops/8weights. No sharedbasepointer API assumed. Physicalpackedstage[8,512]u32 withBlocked([1,4],[1,32],[4,1]) gives16u32 perlane perrecord. Puredecoder pack16 produces16packedBF16u32, splitlohi thenjoin/reshape/permutation forms[4,32,64] withtrivialDotAconversion assertion. GPUcompilation may reject the view; CPUmappingproof alone does not establish backendexpressibility.

adaptive_record_async_mapping_test.py passes8192stageviewelements, puredecoderoperandABI and depth2/3/4ring scheduling; earlier codec and partitionproofs also pass. Everyiteration commits a group including fullymaskedtail so constantwait(depth-1) tracks correct slot; finalwait0 drains before output/reuse. NoGPUjobs launched by this reviewer. CPUcompiler driver adaptive_async_compile_only.py comparesWonly/WX and emitsresource/PTX/SASS evidence without GPUtiming.

## Async CPU compilation follow-up

Actual CPUcompile artifacts: Wring104registers/49248shared/0spill; WXring128registers/98400shared/0spill. Puredecoder reshape and asserttrivial conversion compile successfully. No performance inference.

WX completePTX confirms wait_group2 thenbar before sharedoperandconsumption, secondbar before same-slot futurecopy, loopback then wait_group0+bar before firstscratchstore, scratchpublicationbar before reductionreads. Finalbarrier inserted bycompiler protects allthread oldstages→scratch aliasreuse; source lacked explicitbar afterwait0 but generatedcode is safe.

Static40ordinarysharedstores are8flagstoreinstructions(2 perissue ×3prefills+1loopbody) plus32finalFP32accumulatorstores. Hotloop has ONLY2ordinarysharedflagstores, no decodedweightsharedwrite. SASS showsLDGSTS async copies andLDS.128 encodedloads; stagedX uses sharedoperand loads. Wring printedPTX truncates before finaldrain, so completefinalreuseproof above applies specificallytoWX; sourceprotocol andsamecompilerresource/count provide supporting, not independentlyfull WringfinalPTX evidence.
