# Top-50 campaign, 2026-10-26

Live leaderboard snapshot: `top50_leaderboard.json`. Board best 1.4332,
rank 198; rank 50 is 1.5344 (snapshot 2026-10-27). Required geometric-mean
time reduction from board best: 6.60%. Top 50 has NOT been achieved.

Three subagents split dense MMA, cross-tile pipelining, and source research.
New performance experiments remain Python/Triton/Gluon; cp14 CUDA kernels
are retained only as measured baselines. No slower probe was integrated.

## Measurements: raw event times of cold CUDA graph replays

`top50_combined_final.out`:
- qkv m16: BN32 mma_v2 BK128/BK256 28.112/24.352 us vs torch cuBLAS21.328.
- qkv m32: 29.040/26.048 us vs21.248. Normal/outlier/onehot checks pass.
- o m256: persistent restart29.072, continuous26.528, nonpersistent19.856,
  cuBLAS14.000 us. Continuous prefetch helps its own control but loses
  absolutely. All three kernels passed numerical and changed-input checks.
- Compile-only failures in first two sessions are not performance evidence.

`top50_router_stack.out`:
- m128: stacked512 GEMM+sum16.096 vs two GEMMs21.296 vs cp14 CUDA12.176 us.
- m256: stacked17.296 vs two24.672 vs cp14 CUDA15.056 us.
- All normal/outlier/onehot checks pass. Reject stack path.

`top50_router_distributed.out`, Triton cooperative launch, fixed-order
distributed output reduction, private monotonic counters:
- m128:13.376 vs cp14 12.192 us; m256:15.296 vs15.264.
- FP64 correctness and graph capture/repeated replay pass.
- Six further configurations in `top50_router_sweep.out` also lose.
- `top50_router_sweep2.out` contains the final occupancy/split variants.

These are local paired same-run comparisons, not judge score estimates.
No subtraction of an assumed graph overhead is used to claim a speedup.

## Judge checks

- dfd482040f314655: cp14 deduplicated to9d8ae0d9547747ce; not an independent rerun.
- 03960c2e352c40e3: `top50_candidate.py`, cp14 plus latent TMA output
  descriptor repair. Status saved in `top50_candidate_status.json`.
  Active configurations do not exercise TMA_STORE; no speedup expected.
- After check03960c2e352c40e3 completed (9/9, no flags,1.426996), its exact
  candidate source was promoted to submission.py. Previous working source is
  preserved as checkpoints/pre_top50_submission_backup.py. All prior checkpoints
  are unchanged. This establishes the canonical baseline, not a new speedup.
- edfc418066374801: `top50_candidate_q2.py`, coherent Q2 ring (matching
  prefill/refill and retirement). Await judge evidence before accepting.
- 0a1d26b4cf674966: `top50_candidate_layout.py`, untimed physical-weight-layout
  selection versus the current selected kernel. FP64/probe/determinism checks,
  two paired timing rounds, >=0.35us and >=2.5% margin. Lossless original-precision
  transpose only; fresh outputs and timed input computation retained. Judge pending.

## External inspiration

https://github.com/lee101/openkernels (Apache-2.0).
README's projection1.149 figure is stale relative to docs/ideas.md's v22
1.459 check claim. Neither establishes a transferable speedup here.
Relevant mechanisms: single stacked-router GEMM (v11), distributed router
partial reduction (v22/wgr), asynchronous cluster partial pushes and relaxed
register caps. PDL code in that repository is excluded by this challenge's
rules. No external Python/CUDA was executed wholesale.

See `top50_research.md` for primary Triton references and corrected prior
assumptions. In particular the x-ingress/multicast diagnosis was falsified
by earlier local no-x experiments and must not be revived as established fact.

## Remaining work

Update: judge Q2 check edfc418066374801 passed9/9/no flags at1.418065;
no score gain established, do not integrate. Layout check0a1d26b4cf674966
passed9/9/no flags at1.431450. It selected only unpooler128 after repeated
paired setup timings ~23.14->22.24us raw (~0.9us). Overall score differences
are within confidence intervals. Exact validated layout source promoted to
submission.py and checkpoints/cp16_layout_check1.4314.py; preceding canonical
source preserved at checkpoints/cp15_descriptor_check1.427.py.

Next research opportunity is separating accumulation lifetime/register caps
from async cluster reduction, with inspection of generated code first.
Respect existing 1800-second Modal ledger; do not reset it silently.


## Clean-slate follow-up and experiment loop (2026-10-27)

All three named pending checks are done; their status snapshots were refreshed.
No new Modal jobs: ledger remains 1782.61/1800 seconds.

`experiment_loop.py`, `loop_correctness.py`, `loop_stats.py`, `loop_report.py`
provide source-pinned full-module experiments, cold captured-graph paired AB/BA
samples, fresh-input correctness checks, two confidence-gated timing cohorts,
and raw JSONL reconstruction. 30 CPU regression tests and manifest dry-run pass.
GPU execution of this new harness remains unvalidated; no local timings claimed.
See LOOP_ENGINEERING.md for commands and limitations.

Clean-slate research produced `clean_slate_weight_multicast.py`: share original
BF16 W tiles between CTAs that own different token partitions. This differs
from the rejected x-multicast mapping. API/source review and syntax pass only;
GPU lowering, synchronization, accuracy and speed remain unproven. Independent
compiler review and a separate candidate are in progress. submission.py remains
exact validated cp16. A shared-weight mechanism may still lose to L2 reuse and
its cluster barriers; neither the mechanism nor top50 is established.


New legitimate judge check: 83ac143714ef46b2, source
`top50_candidate_weight_multicast.py`, SHA256
388a04dccd70f77f4df2be261cc626ce5a087ac0e1882a13b0f6e835c18c299f.
Exact cp16 prefix plus isolated Triton/Gluon W-multicast. Tries unpooler256
then pooler256 with untimed correctness/resource gates and repeated cold graph
wins >=max(0.5us,2.5%); otherwise canonical fallback. Absolute setup deadline
260s, reserve before JIT; cannot interrupt ongoing compile. No new Modal work.
Do not promote until completed judge/log/per-shape evidence is reviewed.


Second independent architecture check: b1391aaca54b4ba9, source
`top50_candidate_token_warpgroups.py`, SHA256
70a9313951dcd487fedbcd8e3e2b4413f7ffc2481e6911d13a2a8b87e25ccddd.
Exact cp16 prefix; only unpooler256 eligible. One CTA has two token-partitioned
consumer warpgroups sharing W, rather than two clustered CTAs. Local barrier
protects stage reuse. Paired resource/correctness/time gates match prior
candidate; no performance result yet. No Modal job launched. The external loop
now has 30 passing CPU tests including input/accuracy-gate tests using torch CPU.


83ac143714ef46b2 completed: 9/9 tests, no flags, 1.425908x. Both new
multicast launches were rejected during compilation at final gl.store:
`'list' object has no attribute 'type'`. No wmc selections. This is canonical
fallback timing, NOT measured multicast performance. Saved profile/log under
bench/profiles/top50_candidate_weight_multicast.83ac143714ef46b2.*.
Pinned v3.8 Hopper API explicitly returns a scalar for one wait dependency;
do not blindly change acc=wait to tuple unpacking. Reviewers propose explicit
BlockedLayout epilogue with separate conversion/pointer/mask statements and full
traceback diagnostics. Original submitted candidate remains immutable.


b1391aaca54b4ba9 completed: passed/no flags, 1.419957x. The dual-token-WG
kernel DID compile: shared163872B,90registers/thread,zero spills; preparation
FP64/probe/outlier/zero/changed-input checks passed. Raw cold graph comparison:
34.272us new vs27.488us canonical (24.7% slower), selected=False. Reject this
implementation. Do NOT queue its v2 output-layout change: insufficient mechanism
to close6.8us gap. Existing submitted source remains preserved.

Wmc v2 SHA50eed811983aeaebacb8131658e0ac87ffe13afce5c75bddcdac82bbc6635e64
uses explicit blocked output layout with sameCTA token partition and full
tracebacks. Independent compiler review agrees pointer/mask/value partitions.
GPU compilation still required. This is a meaningful cluster-store lowering
correction, not a performance rerun. Canonical submission remains cp16.

Pending revised multicast judge check: 55c03961d2bc4f65.


55c03961d2bc4f65 completed 1.433611x canonicalfallback, no new selections.
Explicitblockedepilogue alone did NOT fix compilation. Full traceback isolates
layout equality during gl.store broadcast: a Triton tuple versus Python list
in cga_layout, not an accumulator/wait return. v3 constructs both MMA/epilogue
layout objects on host and passes as constexprs, matching upstream test pattern.
No new warp-specialization/performance claim follows this compiler correction.

New adaptive exact16x64 recordcodec prototype `adaptive_record_codec.py`:
independently reran CPU selftest: exhaustive BF16-bit roundtrip, exact rebase
guards; 1,048,576GaussianBF16 weights,146escapes,137/1024rawrecords,
12.56640625bits/weight includingmetadata. Novelty: recordrawfallback removes
sparsecorrection, but increasescompressedtraffic4.6%. Currentwinningz9 small-m
fixcost alreadyhidden (~20.2vs20.15us ablation); therefore prioritize m32's
currentlyuncompressedpooler/unpooler, not fictional6us smallmfixsavings.
Triton adaptive prototype is correctness-first bytegather and UNVALIDATED;
needs vector loads+decode before any fair performance candidate. No Modal work.

Pending host-layout-normalized multicast check: e24565548306424f, source
`top50_candidate_weight_multicast_v3.py`, SHA6b6ea84a7fd052a2e1b824bbb0c191b7e6b5c2f8199e1fa2235a732da111f911.
No new Modal jobs. Adaptive record format is the main new implementation direction;
source-only RS audit found register-source WGMMA already implemented historically,
so that mechanism is not presented as novel.


e24565548306424f completed: 1.446424x, passed/no flags; NOnewWMCselections.
Hostlayoutnormalization fixedcompile. BothWMCkernels used196640B shared,
90registers,0spills andpassedsetupcorrectness. Rawunpooler25671.296us
vs27.712canonical; pooler25671.520vs30.592. Reject conservativefullcluster-
barrier ring. Its scoreiscanonicalfallbackvariation, notnewarchitecturegain.
Do notqueueWMCagainwithoutdifferentconsumer-creditprotocolandpeerreview.

Vectorized torchpacker `adaptive_record_torch.py` accepts customphysical
fragmentorderandworksuntimedonCPU/GPU; CPUbyte-exactcomparisons tostdlib
passedmixed/raw/chunksizes1,3,1024/rebase/invalid/zerocases. GPUpreparetime
unmeasured. CurrentadaptiveGluonkernelstillneeds offlineDotAorderrepack
andvectorizedload/decode; compilerpeerisimplementing, notreadyforjudge.


New materially different adaptive candidate check: 2bf4dc23c8c742a6.
Source`top50_candidate_adaptive_record.py`, SHA
cf707451b027cf9f5c678cee6840c5e3aeb8f8467c39ac4172b1a46807109f85.
Exactcp16prefix+inlinedtorchpacker+customGlDotphysicalorder+predicatedvector
ASMdecode+MMA2. Threecompressedv4loads ORfourrawv4loads/thread,seveninteger
ops/eightweights, noformatbranch/CSRfix/decodedWsharedstaging. CPUmapping
proofpassed4096mixedfragmentvalues; preparationpackerbyteexactCPUtests
and30harnesstests pass. GPUcompileunvalidatedbeforejudge. Onlypooler/unpooler
m32 canselectaftercorrectness, fresh-inputgraphchecksandtwocold31repwins
>=max(.5us,2.5%). Elsecanonicalfallback. No newModaljobs.

Next independentimplementation task: rank3batchedMMA2intraCTAKsplit4, R32
rows, acc[4,32,m], finalFP32sumthenrescale+BF16. MoreCTAsandshorterKloops;
source-onlyframeworksupportidentified, exactkerneluncompiled. Do notclaim
performanceorresurrectpreviousRSmechanismasnewwork.


Offlinecompilerbreakthrough: existingModal`compile_only` functionhasNO GPU
resource; ranCPU-onlyofflineSM90compile9s (NO H200allocation,ledgerunchanged).
Fixedimageconfirmedtorch2.14.1+cu130,triton3.8.0. Bothadaptivekernelscompile
andptxaspass. Direct:76regs,0spill,0shared,0barriers;split:136regs,0spill,
2048Bshared,onebarrierID. VectorASMpreserved; noformatbranchorweightdecoded
sharedroundtrip. Splitsharedstoresareinfinalreductionepilogue, notdecoder.
Fullcompiler/PTXlog`adaptive_compile_only.out`. Numerical/timingstillUNPROVEN.

Splitcandidate source`top50_candidate_adaptive_record_split.py`, SHA
14dce9151046d2322503add9133e50d5ee13c9b4bac7cc4684a6795aacf07e32.
Exactdirectcandidateplusseparaterank3kernelandlaunchoverride; allsamecorrectness/
selectiongates. Materialmechanismchange:intraCTA4wayKpartitionandR32rows;
128pooler/256unpoolerCTAs vs64/128direct, oneFP32epiloguereduction.

Pending adaptive judges: direct 2bf4dc23c8c742a6; intra-CTA split 9a4165d0f3e948a2.


PTXinspectioncorrection: ptxas`used1barrier`meansonebarrierID, NOTone
syncinstruction. ActualsplitPTXcontains31bar.syncand16sharedstore/16load
instructions, allaftermainKloop (zeroWdecodedshared traffic). Genericgl.sum+
MMAoutputconversionserializesmany2048Bscratchphases. Proposedconcretefix:
explicit16KiBfullpartialstaging, onebarrier, loadbatchinregisterswithblocked
outputlayout, localFP32sumthenBF16store. Newsourcevariantmustcompileand
verifyactualinstructioncountbeforeGPUcheck. Originalqueuedsourcepreserved.


CPU-onlycompilercomparison`adaptive_epilogue_compile_only.out` CONFIRMS
31→1bar.syncinstruction reduction; shared2KiB→16KiB; unchanged136regs,
zerospills. Allmath/packedloadsunchanged. Newsource
`adaptive_record_split_epilogue.py`. Thisisprovedgeneratedcodeimprovement,
notyetGPUtimeimprovement. Specificsharedbankmappingreviewthenfoundplain
MMAstoresfourwayconflicts; rowxor((token//2)%4)*8 swizzle(vec8,phase2,max4)
makes32lanebanksdistinctandpreservesreaderbijection. Separatevariant
`adaptive_record_split_swizzle.py`awaitsCPUcompilercheck. NoH200allocation.


ADAPTIVEGPUnegativeevidence: direct2bf4dc23c8c742a6 passed,1.438884x,
NOselections. Rawpooler318.720usvs22.400;unpooler162.176vs21.856.
Split9a4165d0f3e948a2 passed,1.451458x,NOselections. Rawpooler109.344vs21.888;
unpooler58.240vs21.504. Bothsetupnumeric/freshgraphcheckspassed,alloutputs
canonicalfallbackafterlosing. Overallscorechangesarefallbackvariation,
NOTnewalgorithmspeedups; doNOTpromoteorclaim1.45guaranteedfromthisrun.

Singlepublication/swizzleepiloguescompile(136regs,16KiB,0spill,1actualbar.sync),
but epilogue-onlycannotcrediblyclose4–15xmainkernelgap. DoNOTjudgeepionly.
Potentialmaterialnextmechanism: compressed-bytecp.asyncring4+metadataadvance
prefetch, decodeinregs+intraCTAKsplit. Directmetadata→packedLDGdependency
chainandunoverlappedXloadsneedSASS/latencymodelauditfirst. Oldcompressed
TMA+RSpathwasalreadytried; newproposalneedsadaptiveMMA2specificproof.
No newModalGPUjobs. Canonicalsubmission.py remainsunchangedcp16.

Currentjobstate: all recorded judgejobs through9a4165d0f3e948a2 are complete;
NOjudgejobpending. Don'tpollcompletedjobsagain. Currentimplementationwork:
compressed-bytecp.asyncringwithpure7opregisterdecode, optionalXstagingstatic
control; requireCPUcompiler/layoutproofandpeerreviewbeforefuturejudge.


## Latest state: adaptive asynchronous pipeline

Machine-readable state: CAMPAIGN_STATE.json. All previous jobs are complete.
Only pending job: **83233f53d5ff4a84**, source
`top50_candidate_adaptive_async.py`, SHA256
`d2c3272138c899a9643d42c58b1291680bccf99cc54a2dbf8e7e3110b091b85f`.

This is a materially different load pipeline, not an epilogue-only retry:
compressed/raw records enter fixed shared slots through predicated cp.async;
a pure seven-operation decoder feeds MMA registers; no decoded-weight shared
round trip. W-only and W+x controls share the same safety schedule. Depth3,
intra-CTA four-way K split, one-publication swizzled reduction. Explicit final
CTA drain barrier protects allocator reuse. Both controls compile in the fixed
CPU-only image: 104/128 registers, 49,248/98,400 bytes shared, zero spills.
Actual PTX/SASS and mapping/lifetime peer review pass; GPU accuracy and timing
remain for the judge. Full evidence: adaptive_async_compile_final.out.

Preparation validates normal/outlier/probe rows, changed-input captured graph
replays and zero inputs for each control. It chooses the faster control, then
requires two paired cold rounds beating canonical by max(0.5us,2.5%). Only
pooler/unpooler m32 can dispatch it. No selection means fallback variation;
do not claim a speedup from its overall score alone. No Modal H200 job started.
Canonical source remains exact cp16. Top 50 remains unachieved.


## Verified per-shape improvement: cp17

83233f53d5ff4a84 passed with no flags. Both controls passed all setup numerical
and fresh-input graph checks. W+x staging is essential in this new path:
pooler W95.104us vs WX24.736; unpooler W55.936 vs WX20.672. Pooler loses
canonical22.368 vs WX24.448 and is rejected. Unpooler m32 wins twice:
21.760→20.480us, confirmation21.696→20.416us (1.280us,5.90%). Judge profile
unpooler m32 adjusted16.166us vs prior cp16 17.435us; raw20.256 vs21.408us.
Cross-run numbers support but do not replace the same-run paired evidence.

Exact passing source promoted to submission.py and
checkpoints/cp17_async_unpooler32_pass.py; preceding cp16 checkpoint preserved.
This retains one measured per-shape gain. Overall check1.427436xCI1.4060–1.4488
is within noise of earlier scores; NO overall score gain or top50 claimed.
Do not submit to leaderboard merely to chase that noise. No pending jobs.
No Modal H200 jobs; ledger1782.614647/1800s remains unchanged.


## ac2 (Claude Code takeover, 2026-10-28)

Board 1.433x rank 226; rank 50 now 1.545x. Hypothesis: cp17 async kernel at m32
streams x once per 32-row CTA => x L2->SM traffic = 2*N*K bytes (67 MB) > compressed W
(50 MB). ac2 adds RW=2 row-warps sharing one staged x (half x traffic) and cross-CTA
split-K (last-arriver, deterministic 2-way sum) so pooler gets unpooler's per-CTA shape.
All 11 variants compile CPU-only (Modal compile_only): 0 spills, 97-150 regs,
74-197 KB shared. Judge check 571959ca10f24e68 (bench/top50_candidate_ac2.py,
sha 23cd71cb...) pending; gates as cp17 (correctness + fresh graph + two paired
rounds >= max(0.5us,2.5%)).

### ac2 results (checks 571959ca, ef1b63eb, cecfa932, bdcbe951, dc389a40, f133c38e)

Raw paired cold-graph us in judge prepare, current -> best ac2:
- x-traffic halving (RW=2): unpooler32 20.19 -> 21.02 (loses). x is not the limit.
- 571959ca crashed (RW=1 row-group index bug, fixed: row = n0+(rgih*RW+rowhalf)*16).
- Consumption-ordered payload: no effect. Padded x stride: no effect.
- Metadata on issue path is the latency chain: consumption-ordered meta in global
  +10us; preloading the CTA's metadata into shared: pooler32 24.3 -> 22.8,
  qkv32 23.3 -> 21.7. With D4: pooler32 21.95 vs current 21.92 (parity),
  pooler16 19.84 vs z9 19.65, unpooler32 19.87-19.97 vs cp17 20.06-20.22 (below gate).
- Cross-CTA split-K (last-arriver) costs +5-8us at m32 (unexplained); qkv (72 CTAs
  without split) still loses 3-5us.
No selection; scores 1.426-1.436 are fallback noise. Canonical remains cp17. Line closed.

## CW8 pooler release check, 2026-10-28
Pending326978ecb91b416d, source bench/top50_candidate_cw8_pooler.py SHA dad17a60f386060cce3f66b7a34d36705aba1abc6269441d799ac521503ccf6d. Exact cp17 prefix preserved. Final source review PASS, CPU compiler+mapping proof PASS; no GPU proof yet. CW8 uses fixedR32, Kpartition8, halved serialloop; distinct from rejectedac2RW2. Correctness/freshgraph/two paired rounds required before selection. Canonicalcp17 unchanged; no ModalGPUjobs. Do not queue duplicate.

CW8 check326978ecb91b416d DONE PASS(no flags), no selection. BothCW4/8 numerical+freshgraph checks pass,128regs/0spills; shared98400/196800. ControlsCW4=24.096us,CW8=22.304us; canonical22.208vsCW822.336, so rejected. Overall1.428688 is fallbackvariation. Canonicalcp17 preserved. New CPU/source follow-up: CW8 +1KiB CTAmetadata preload, combining measuredorthogonal serialloop andmetadata latency mechanisms; requirecompiler/peerproof beforecheck. No pendingjobs.

## CW8+metadata check
Pending1dd77d0577fe4152, bench/top50_candidate_cw8_meta_pooler.py SHA8c943911bf27e71f7713466e18541fece2064aa2c7fc5f5cab96388e378f1b1c. Combines measured7.4%ownfamilyCW8loop reduction with measuredorthogonal metadataissuechain removal (ac2). CPUcompiler exacttorch2.14.1/triton3.8: GLOBAL128regs196800B, META158regs197824B,0spills; staticglobalLDG32→1,barriers6→7,MMA32/cpasync70unchanged. CPUlayout/tailproof and twopeersPASS; SASSmeta0x30000,flags0x30400nonoverlap. Exactcp17prefix. Onlypoolerm32eligible andrequiresbothcontrols correctness+freshgraph andtwo31rep pairedwinsmax.5us/2.5%. GPUproofpending. No ModalGPUjobs.

## Verified cp18 pooler m32 gain
1dd77d0577fe4152 DONEPASS noflags. GLOBAL andMETA both passnumerical+freshcapturedgraph tests. META controls22.272→21.536us; versuscanonical first22.400→21.536, confirm22.496→21.504 (0.992us/4.41%). SelectedTrue. Officialprofile runs _cwm_kernel forpoolerm32. Exactpassingsource SHA8c943911bf27e71f7713466e18541fece2064aa2c7fc5f5cab96388e378f1b1c promotedtosubmission.py andcheckpoints/cp18_cw8_meta_pooler_pass.py; cp17/cp16 retained. Score1.438075CI1.416504–1.459646, overallgainnotestablished, no1.45/top50claim andno scoredsubmissionyet. No pendingjobs/noModalGPUjobs. Raw sourcepinnedprofiles inbench/profiles/top50_candidate_cw8_meta_pooler.1dd77d0577fe4152.{json,log,txt}.

Router source audit follow-up: BT64 retaining existing cluster reduction is untried (previous BT64 trials used global reduction). High-m grids64→128 CTAs, accum64→32 FP32/thread, shared213024→139296B/CTA; Kloops8/16 and cluster phases unchanged; W requests/TMAcommands double. Requires new instantiation or Triton cluster port; honor Triton-only preference, no new C++ kernel silently. Hypothesis only; no source/GPU job. Estimated1us/2us router savings would raise1.4332 onlyto1.4421, not1.45.

## Router Gluon feasibility audit (CPU only)
router_cluster_compile_only.py now offline-compiles the isolated R32/T64 two-CTA epilogue for SM90: generic8192Bshared, async8200Bshared/8staticst.async, both0shfl and4staticclusterbarrierinstructions. Required explicit separate range layouts, sequentialexpanddims and Python-global nestedCGA constants to repair original syntax-only probe. See ROUTER_TRITON_FEASIBILITY.md androuter_cluster_compile_globals.out. This is notGPU correctness/timing, not4/8CTAproof, notfullrouter. No judgechecks orGPUModal allocations. Cp18preserved, no pendingjobs.

## Cluster sizes 4/8 compiler extension
router_cluster_sizes_compile_only.py and .out extend the isolated epilogue to2/4/8CTAs. Allsix generic/async variants compile onSM90: shared8192/8200B respectively,0shfl,4staticclusterbarrier instructions, async8staticst.async. NVMMADistributedLayout ininstalledTriton3.8exposes cga_layout. StillnotGPU numerical/timing orfullrouter proof. Nextboundedsourcework: router_cluster_bt64.py MMA2 hi/lo FP32acc mainloop plus localcp.asyncring andsameclusterepilogue; no newC++ orGPUallocation.

## Triton router cluster release check
Pending002617846e374c4c, sourcebench/top50_candidate_router_cluster_bt64.py SHAe98d58899ed2f04cf0dce36aff7fff610e3c9b9a301477f68e3fcbb933192494. Exactcp18prefix. FullGluonBT64rank3MMA2 mainloop +96KiBcp.asyncring, splitKcluster8form128/4form256,128physicalCTAs, FP32hi/loacc; genericvsasyncclusterepilogue controls, noglobalworkspace/tailkernel. CPUcompileall4pass147–149regs98304Bshared0spills64mma70cpasync0shfl; generic6vsasync4staticclusterbarrierinstructions. BoolmaskAPIissuefixedbeforeCPUrelease; rank3leadingstageNVsharedpatternconfirmedbycompiler. CPUownership/grid/K/tail proofs andactualdispatchregressionPASS; fresh-xpass-through andFalseSTOREmembershipcovered. Root+expertpeer wrapper/protocolreviewPASS. Bothcontrols mustpassFP32tol1e-4normal/outlier/probe+freshgraph/zero beforetiming; independentlyselectM128/256onlyaftertwo31rep pairedcoldwinsmax.5us/2.5%;outer260ssoftdeadline. GPUproofpending. Canonicalcp18unchanged. No newModalGPUjob; do notduplicatecheck.
