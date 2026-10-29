# Top-50 takeover log

Live account: `bharath03-a`. Scores below are live OKC board values; lower is better. The pasted external update is not this account’s current standing.

| Problem | Rank | Score (µs) | Speedup | Top-50 cutoff (µs) |
|---|---:|---:|---:|---:|
| muse_attention_decode | 784 | 3324.82 | 1.1328× | 1550.27 |
| muse_moe_w4a16 | 775 | 10712.04 | 1.2555× | 5968.81 |
| muse_decode_layer | 593 | 18147.76 | 1.2465× | 11059.14 |
| muse_prefill_layer | 484 | 386905.84 | 1.0013× | 230076.16 |

## Strategy
1. Prioritize whole decode layer: required for stated qualification. Components are independently scored; improvements must be explicitly ported.
2. Use shape-level profile evidence. Dense decode small batches and Marlin dominate sliding layers; full-context attention dominates some large-batch full layers.
3. Preserve submission.py while testing independent candidates. Spend a small initial check budget per idea. Promote only passing, repeatable improvements beyond timing uncertainty.
4. Use CUDA graph replay timing and cold L2 on Modal, matching OKC; OKC remains scoring authority.
5. Never cache input-derived schedules across calls, change precision beyond rules, skip keys, or move timed work outside kernel_fn.

## Initial experiments
- Whole layer pack_gqa=True: `candidate_pack_gqa.py`, check `3e54097dbc934eba`.
- Whole layer existing Gluon/Triton dense dispatch over older CUDA tiles: `candidate_gluon_dense.py`, check `8394363436bf45dd`.
- Attention fresh split16 schedule: `candidate_takeover_split16.py`, check `bc828eed08b14efe`.
- MoE compact local alignment: check `e37b8df3cb654c7a`.
- MoE single-token direct packed GEMV + SwiGLU: `candidate_direct_gemv.py`, check `7123ab6e42a945a7`.
- Prefill expert block M32: `test_prefill_m32.py`, check `5a552622d52f4677`.

## Research
- FlashAttention upstream split-KV/GQA tuning benchmark: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/benchmark_split_kv.py
- Existing Modal runner authenticated; bf16_proj/.venv/bin/modal available.
- No top-50 claim until live board verifies it.

## Implemented at takeover
- Attention: fused Q/K RMSNorm + RoPE + FP8 cache write, FA3, preallocated buffers, fused output RMSNorm/gate.
- MoE: prepacked NVFP4 Marlin experts, scratch/alignment buffers, separate SwiGLU, fused post-expert norm/weighted sum.
- Whole decode: full layer, older custom CUDA/Triton/Gluon projections, fused KV preparation, router/pooler stream overlap, Marlin experts, fused norm/residual boundaries.
- Prefill: full layer using SGLang/Marlin; expert GEMMs dominate sliding-prefill profiles.

## Modal findings
- FA3 split count 0/4/8/16/32/64/128 produces essentially identical large-batch runtime in synthetic scattered page1 cache sweep. Reject metadata-only dramatic speedup claim.
- Direct packed MoE GEMV passes 3 fresh seeds (~0.54–0.56% relative L2) but preliminary 309–570us two-GEMV timings are far slower than expected Marlin; do not promote.
- Whole-layer dense cp14 integration check `54f2a41cdd424d40`, source `muse_decode_layer/candidate_dense_cp14.py`; compile passes, numerical/judge and prepare-time validation pending.
- Additional grouped-head Triton attention backend Modal sweep running; no promotion until correctness and cold replay timing verified.

## Promotion policy
Run scored submissions only for a candidate with passing OKC correctness and a meaningful measured gain. Recheck borderline improvements relative to reported uncertainty. Keep source-to-job mapping and backups. Goal remains active; top50 not achieved.

## Follow-up evidence (second goal turn)
- Grouped-head Triton attention passed random-input numerical comparisons (~0.003–0.009 relative L2), but B32 cold replay was454–482us vsFA3~371us: rejected.
- Fresh generic index_select→contiguous FP8 attention: ~3318us atB32, bit-identical output: rejected.
- Fused byte-preserving Triton gather→contiguous attention: ~575us atB32 vspage1FA3~371us, bit-identical output: rejected. Page64 path~601us also loses. Gathering was included in graph timing.
- Installed FlashInfer Hopper auto dispatch selects XQA; documented page sizes16/32/64/128 exclude scored page1. Research continues; no unsafe direct call promoted.
- Independent Marlin header-wrapper candidate exposes tile_n/tile_k without patching installed modules. Initial compile include-path failure repaired; bounded Modal sweep v2 live, numerical/performance results pending.
- Whole-layer cp14 H200 probe generated full weights successfully and entered preparation; numerical result and300s prep limit remain unverified. Preparation duplicates old projection compile/warmup; v2 being prepared to remove redundant work.
- OKC compact alignment check e37b8df3cb654c7a confirmed running; remaining original eight-check set not completed at last poll. No new candidate promoted or scored by this team.

## Compilation and next-backend findings (third goal turn)
- Confirmed from installed tvm_ffi source: `TVM_FFI_CACHE_DIR` defaults to ephemeral `~/.cache/tvm-ffi`; persisting only TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR did not preserve Marlin builds. Future runners use `/cache/tvm-ffi` before imports.
- Native Marlin same-route comparison hit its bounded300s timeout while cold compiling; it produced no timing. Whole-layer H200 integration likewise observed active nvcc/ptxas compilation, not a stale-lock deadlock. Do not interpret these as numerical failures.
- `candidate_dense_cp14_v2.py` removes unused old dense CUDA compilation and warmups, retaining cp14 preparation/self-tests and full layer state. Python compile passes; not yet remote-validated.
- MoE agent preparing CPU precompile into persistent cache before future GPU benchmarks, preventing repeated expensive cold compile loops.
- Next bounded attention hypothesis: fused fresh gather into page64 plus XQA backend, which differs from rejected gather+FA3. Raw backend and gather costs must be separately measured; no direct page1 XQA call is valid.

## User-provided openkernels reference
See OPENKERNELS_RESEARCH.md for pinned revision, attribution, standalone and integrated candidate job IDs. Original submissions preserved. Compact alignment and packing-only checks passed but did not improve materially. Subagents hit usage limits; root continues.

## User scope clarification
Improve each of the five challenge scores and reach top50 overall, retaining whole-decode top50 as qualification target. Track per-challenge performance and actual overall rank separately; prefill has its own board and does not count toward the overall decode board. No component improvement implies an overall rank without live verification. BF16 projection work has independent candidates already pending; do not overwrite its submission or duplicate its checks.

## Expert-team coordination and positive attention evidence
See RESEARCH_TEAM.md for ownership, peer-review protocol and experiment proposal requirements. Retried all3 existing agents after user explicitly requested expert group collaboration; status was running at last query, output/availability still to verify.
Pinned attention H200 probe completed4 numerical checks and8 paired timing cases; rawB32full decode391.22→210.43us andverify396.37→240.99us. Official check pending; no rank claim. MoE CPUcompile launched session26255, log/tmp/openkernels_moe_cpu_compile.log, timeout1200s, persistentcache.

## Verified projection integration gain and scored rollout
Wholedecodecp14 check54f2a41cdd424d40 passed13/13/no flags:17275.39us,1.309394x,95%CI[1.289754,1.329035]. Accountbestprior18147.76us→4.81%lowerlatency. Exactnormalizedsourceverified by repeatcheck278b837b385c4a94 returningduplicate_of54f2a41cdd424d40/contenthash8703e2e72e292a74819ae4e6206fa663836c1cdc1f4273cc1831488f47916818. Plainfilesha256isbe77ac46... and differsOKCnormalizedhash; do notcomparethosealgorithmsdirectly. Scoredsubmit22316a25039248e1 launched for exactcandidate_dense_cp14.py; ranknotyetverified. Originalsubmission.py preserveduntilscoredresult. BF16ownerchatnotified.
MoECPUcompilationcompleteafteronepreemption; persistentcachecorrectlyresumedcompletedmoduleswithinseconds. H200originalpersistent6sizesnumericallypass; originalg1M1=36.98usvsoldpersistent56.54us,sameinput. Fullprecisionboundaryreviewstillrequired; notpromoted. New isolatedM1strictBF16g1candidate created: gate/upBF16, BF16downoutputbeforepostnorm, BF16effectivenormweights; otherloadsfall backoriginalMarlin. CPUcompileisolatedg1running session86724,/tmp/strict_g1_compile.log. GPUprobe_g1_strictfunctionpreparedbutNOTlaunchedyet.

## Active implementation queue after strict-boundary experiment
- `candidate_moe_g1_strict.py` standalone: all4normal/skewedfreshH200 numericalchecks passed,L2~0.0043–0.0048; pairedvsoriginalg1timingswithin0.12us. Downprojection isroundedBF16BEFOREnormweight, normweightroundedBF16, scaledgate/uproundedBF16; unchangednativeMarlinfallbackotherloads. Officialcheckb959d99adf194b71 queued.
- `candidate_cp14_open_attention.py`: checkedcp14projection + pinnedcustomattention, officialcheck086635c9d2d04846 queued.
- `candidate_cp14_attention_strict_g1.py`: addsstrictM1g1andremovesunneededM1routealignment; Pythoncompilepasses, NOTqueuedyet. Waitforcomponentofficialresults/preparebudget.
- `research_moe_full_bf16.py`: leanresearchbackendforEP8bankzero, roundsdownBF16beforepostnorm inbase/ring/g1 plusgate/up/effectivenormweights. Restrictsdispatchtofixedsmall/ring/g1,removesunusedWX/gk/fsrepack/compile. CPUcompilelive session18572,/tmp/full_bf16_compile.log. Reproduciblebuilder research/build_full_bf16.py. NOTjudgecandidateyet; GPUtestrequired.
- `candidate_attention_interior.py`: NEWfdecinteriorvalidblockunconditional16Bcopyfastpath, unchangedtail/newtokensentinelpath; shape/inputboundsbasedpredicateentirelytimed. CPUcompilelive session87269,/tmp/attention_interior_compile.log. Builderbuild_interior_fastpath.py. GPUcomparisonprepared `modal_research_compile.py::probe_attention --target interior --comparison attention` butNOTlauncheduntilCPUdone.
- RootGPUdriver::probe_moe --targetfullpreparedbutNOTlauncheduntilCPUdone.
- Expertagentsagainreportedaccountlimits; resumethroughscheduledheartbeatwhenavailable, donotduplicatelivejobs. Rootcontinues.

## Official attention breakthrough
Check59cbe07c5b5941a3:passedall16tests/noflags,2058.3184us,1.829795x,CI[1.802348,1.863283],vspriorbest3324.82us(38.09%lower). Localsourceidentityconfirmedbyrepeatcheckc3ed8a558b5c4c7f→duplicate_of59cbe07c5b5941a3/contenthash98cbbe3121fa0522602a5a67d2caf3e1d26c249541149af10e3aba4d0b831606. Exactscoredsubmitc81dff9234ee4ecdlaunched; leaderboardrankNOTupdateduntilresult. `VERIFIED_CANDIDATES.json`trackscheckedsourceSHAandscoredjobforattentionandwholecp14.
InteriorattentionCPUcompilecompleted113.16s; H200comparisonlive session53264,/tmp/attention_interior_probe.log. FullBF16leanMoECPUcompilecompleted155.37s; H200probe live session58968,/tmp/full_bf16_probe.log. Do notduplicate.

## Latest wholelayer and full-boundary results
Wholeattentionintegrationcheckf1e076dca5db4640:13/13pass,noflags,16527.87us,1.368615x,CI[1.348086,1.389144]. Exactsourceidentityrepeat8c646a8b8bba410e→duplicateoff1e076dca5db4640. Scoredsubmit40b62805457b4721 queued. Thisimprovesoveroldboard18147.76usby8.92%; officialscoredrankingstillpending.
LeanfullBF16MoEH200passed6scoredsizesandall5officialtestshapes(newinputseed,including4/33/257and128skewed),normalL2~0.0045–0.0047. Candidate_moe_full_strict.pyofficialcheck901f2ac35bfb49a5 queued. ThisisEP8localbankzerooptimizedresearchbackend; nofull256expertshardingclaim.
NewfullBF16shared-activationkernelvariant research_moe_full_bf16_gk.py completesW2partialreductionthenroundsBF16beforepostnorm,removingfactoredglobal-scaleassumption. CPUcompilelive47726,/tmp/full_bf16_gk_compile.log. GPUprobeplannedtargetfullgk--varied-scales testsnormal6+4extraedge/skewedcaseswithperexpertglobalscalevariation, comparesstrictv17vsstrictGKforsmallM. NOTlauncheduntilCPUdone.
Attentioninterior-copyH200prototypeall4numericalchecks passbutno convincing aggregatewin(M1regresses0.1–0.4us,B32fullsaves~1.5us<1%). Donotpromote; consideredrejectedpendingstrongermechanism.

## Loop engineering (2026-10-27)
Implemented research_harness: serialized bounded queue with source/settings fingerprints, OS lock, durable JSONL journal, skip/resume, explicit result-count gates, timeout process-group termination; no automatic scored submissions. Added paired CUDA graph comparison with read-based 256MiB cold-L2 flush, AB/BA sampling, raw samples, block-bootstrap intervals with minimum uncertainty, optional identical-input restoration, GPU/toolchain metadata, and weighted load aggregation. Six meaningful regression tests pass. W13 fused probe now uses this harness; prefill_queue.json runs strict ring plus three fresh W13 seeds. Live loop session47492; logs research_harness/results/prefill/. Do not duplicate.
Cache correction: TVM_FFI_CACHE_DIR alone does NOT preserve native SGLang Marlin. Installed SGLang uses SGLANG_JIT_CACHE_DIR. All future runner environments set /cache/sglang_jit. CPU compile v2 native cache copied from ephemeral /root/.cache/sglang/jit into this persistent path after successful build; subsequent function commits persist it. CPU native/glue/strictQ/oldW4 compiled; fusedW4 was compiling at last read.
Live overall board #663; whole cp14 scored22316... running, attentionc81.../wholeattention40b... queued. Improvements reported so far are checks, not live leaderboard gains.

## Verified scored progress and strict rollout
Liveboard (latest query): overall530, attention261/1.856x, whole507/1.3687x, MoE806/1.2555x, prefill501/1.0013x. Scoredattentionc81dff9234ee4ecd=2029.2727us; wholeattention40b62805457b4721=16527.0499us; wholecp1422316a25039248e1=17369.2225us. Allpassed. Attention default submission.py promoted from exactcheckedsource; olddefaultbackedup submission_before_attention_c81dff9234ee4ecd.py.
StrictfullGKMoEcheck6afd2a4709f54525=7763.2193us/1.73234x,allchecks/noflags; strictwholecombo56c0b49b63f542d9=14177.5205us/1.59550x,13tests/noflags. Recheckedexactsourcevia296fb29f4987458c and330ad7ef441646b4 yieldingduplicate_ofthesechecks. Scoredjobs81277f7dc1c04131(MoE),7e7f05d4f8ed46a2(whole) launched. Do not duplicate. VERIFIED_CANDIDATES.jsonupdated.
Prefillloop: oldW13probeincorrectlyused33CSRoffsets; backendexpects32inclusiveends, leavingunwrittenrows. ReferencecomparecanwronglypassNaN; explicitfinitenessadded. InvalidfirstprobeNOTevidence. Correctedprobecompleted18/18finitecomparisons,3seeds,size1024/3008/16384,wide0/1. Standardwide0~2.5–3.3%microkernelgainallseeds;wide1mixed/inconclusive. Newfusionnotyetfull-layer-enabled. Queue31957finishedsuccessfully; journal/logsunderresearch_harness/results/prefill. AllModaljobscomplete atlastquery. PrefillstrictQpassed4freshchecks(variedglobalscales,skewedrouting).
Harnesspeerreviewfixes: failed-taskresume returnsnonzero; inputresetalsoapplieswarmup/capture; aggregateusespaired-derivedcandidate_time; rejectsourcechangesduringrun. Sixregressiontests pass.

## Scored progress confirmed, research still active
Whole strictcombo scored7e7f05d4f8ed46a2 PASSED13940.1557us/1.6227x; livewhole#305, overall#307. Top50whole10950.5034us/2.0657x atquery; remaininglatencyreduction21.45%. MoE strictGK scored81277f7dc1c04131 PASSED7760.3553us/1.73297x, liveMoE#413. Promotedboth exactsources tosubmission.py; backups submission_before_strict_combo_7e7f05d4f8ed46a2.py and submission_before_strict_moe_81277f7dc1c04131.py. Attentionpreviouslypromoted2029.27us#261. VERIFIED_CANDIDATES recordsfilehashes/checks/scoredjobs.
Prefillstrictbaseline candidate_prefill_strict.py officialcheck4122e0950b134e3a RUNNING; novel fusion candidate_prefill_w13_mixed.py officialcheck3f290bf89ce445cb queued/running, exactsourcespreserved. StrictbaselineexcludesBF16half-KW2autotuning, eagerlypreparesnativebaselinefallbackforsmallT/headmismatch toavoidoldnon-strictsmallfallback/lazywork. Fusionuseslosslesspairedrowbank, modes0/2only, nativebankfallback; no pinschanged. Mixedskewed9/9finitechecks passed, modest2%mode2gain atlargecase.
Liveagentexperiments: MoEatomicqueueCPUcompiled/SASSnotconclusive, H200paired13caseprobe session79160 log/tmp/moe_queue_gpu.log; expertownsit, donotduplicate. Heavy-first variantCPUcompile92551 /tmp/moe_heavy_compile.log. Attention safeBF16PVCPUcompilepassed113.32s /tmp/attention_bf16pv_cpu_compile.log; expertcompilingmatchedFP16FS-disabledcontrol. NextGPUallocationattentionafterMoEqueuecomplete, thenMoEheavy. Preserveonline-softmaxstability, nofixed-offsetunshiftedexpassumption.
Researchharnessnow7regressiontestspass, source snapshots, workspace-widequeueOSlock, post-replayfiniteness/driftguard, durablejournal/resume/failuregates. NewguardbeingvalidatedbyMoEpairedGPUprobe. Do noteditmountedharnesswhilethatprobeislive.

QueueatomicfirstGPUprobeSTOPPEDbyexpertbeforebenchmark:GPUautoarch90mismatchedCPU90aandtriggeredcoldrebuild. Noresult/gainfromit. ExplicitTORCH_CUDA_ARCH_LIST=9.0afixed; CPUrestore91055/tmp/moe_queue_compile_restore.logpending. MoEheavyCPU92551pending. Combined17casebaseline/queue/heavyGPUprobe34pairedcomparisonsreadybutNOTlaunchedpendingattentionallocation. AttentionhasnextGPUallocationaftermatchedcontrolCPUcomplete; exacthandlesinagentmessages. Allfutureworkerscanuse research_harness.environment.configure_h200_cache beforeimports. Harnessnow8tests, addsconsistentCPU/GPUarch/cachesregressiontest.

## Latest wakeup recovery
Usageavailableafterreset; noModalcontainersactivebeforelaunch. AttentionBF16PVfinished45finitecanonicalchecks/8replaychecksallpassbutSLOWER: B16current133.84/control149.50/BF16156.94us; B32current243.89/control277.65/BF16293.94us. RejectsafeBF16variant(noOKC); preserveonline-maxstableoriginal. Logs /tmp/attention_bf16pv_gpu_probe.log.
MoEbaseline/queueCPUcachesrestoredandheavycompiled, all9.0a. Rootlaunchedcombined17freshcase/34comparisonH200probe session92867,/tmp/moe_queue_heavy_gpu.log,runner research/modal_queue_research.py::compare_queue timeout420s. Do notduplicate.
Prefillstrictcheck4122e0950b134e3a PASSED277460.6108us; fusedmixedcheck3f290bf89ce445cb PASSED273675.1173us(all5/noflags). Exactfusedsourceidentityrecheck469654ab7004465d→duplicate_of3f29. Scoredjobdcf056bc66d54bde launched. About29%lowerthanold386905.84usboard; relative1.36%strictbaseline→fusedwholegainiswithinnoise, donotclaimprecisefusionattributionfromseparatejudgechecks. Microkernelpairedgainsremainvalid. Source/provenance VERIFIED_CANDIDATES updated; defaultprefillNOTpromoteduntilscoredresult.

## Current checkpoint — loop harness and scored defaults

All four owned default submissions now match verified scored candidates, with previous defaults backed up. Latest snapshot is PROGRESS.json: overall #322; attention #275 at 2,029.27 us; MoE #420 at 7,760.36 us; whole decoder #320 at 13,940.16 us; prefill #252 at 274,396.63 us. Ranks drift as competitors submit. Top 50 is not achieved.

Prefill scored job dcf056bc66d54bde passed. VERIFIED_CANDIDATES.json records its exact source SHA and check 3f290bf89ce445cb. The total improvement over the old prefill board score is 29.1%; the isolated full-layer fusion difference versus the strict check is within uncertainty and is not proven by those separate runs.

MoE combined queue/heavy experiment ran 20 paired comparisons across scored sizes and M17. Neither change convincingly improved the scored sizes; queue regressed skewed M32. Do not submit either. The M33 replay guard rejected three one-ULP near-zero differences; this was an overly strict pointwise guard, not evidence of backend corruption. The harness now checks global relative L2 and row norms with a near-zero floor, while integer outputs remain exact. A targeted M33 rerun passed (/tmp/moe_replay_guard_validation.log). No reason to rerun neutral performance candidates.

Harness validation: eight CPU regression tests and three real H200 cases passed. H200 cases verify stable output, rejection of intentional replay corruption, and rejection of NaN. GPU app ap-dKcp7I7ynsGxoSXFQPXuTr; log /tmp/muse_harness_gpu_validation.log. Research harness is external to all submitted kernels.

Next research allocations: MoE expert is preparing an isolated const __grid_constant__ P parameter variant after SASS identified the entire 136-byte local stack as an address-taken parameter copy, rather than accumulator spills. It has the next GPU allocation after CPU/SASS checks. Attention expert has approval for an isolated fdec guarded fast-softmax prototype, preserving exact fallback, masks and all keys; CPU first, GPU after MoE. Prefill expert is reviewing the same parameter-copy mechanism before a larger producer-dequant rewrite. Agents must report concrete handles and avoid duplicate runs. BF16 projections remain owned by the other chat and must not be edited.

Latest CPU handles: MoE grid-constant compile session3404, /tmp/moe_grid_constant_compile_v2.log (v17 neutral; ring compiling); attention fdec fast-softmax compile session32145, /tmp/attention_fdecfast_cpu_compile.log, app ap-UGdiWdqZEIgAqXUzvJetyO; prefill grid-constant CPU session37674, /tmp/prefill_grid_constant_cpu.log (baseline compile failed, expert investigating). No new GPU allocations started for these. MoE has first GPU allocation once ring SASS is checked, then attention.

Harness now has nine CPU tests. Dispatch writes a durable STARTED journal record before invoking remote work; a crashed/interrupted coordinator cannot silently relaunch that fingerprint. Resume requires reviewing remote state before explicit retry. All scored sources remain unchanged by harness work.

## Latest completed experiments and pending validation

MoE grid-constant H200 comparison passed all 17 cases and replay guards. Ring normal M128 improves ~3.5%, M256 ~2.7–3.3%; skewed gains smaller. V17, G1 and GK do not have the parameter-copy mechanism, so no follow-on annotation work there. SASS ring STACK136→0, LDL106→0, STL17→0. Official standalone check 041ed31d40cf4296 passed 7,686.96 us (about 0.95% better than scored 7,760.36); whole integration check a56d540dabca47ed passed 13,958.83 us, essentially tied with scored 13,940.16 within uncertainty. No scored rollout yet for this small aggregate gain. Exact candidate research_moe_grid_constant.py and whole candidate_cp14_attention_grid_moe.py; builder build_grid_constant_combo.py.

Attention fdec fast-softmax app ap-HY2FvjIQNqmQ6VSI6lyGHs completed all correctness cases, but regressed full decode: B16 111.39→118.35 us, B32 201.57→218.40 us. Reject, no OKC. Its robust FP8 norm-overshoot overflow test and matched control remain useful. Raw log /tmp/attention_fdecfast_gpu_probe.log.

Prefill grid-constant CPU is not a stack-copy fix (baseline already zero local memory), but reduces static SASS wide instruction count 5,112→3,712, standard 3,560→2,808; runtime gain unproven. Instrumented W4PROF CPU compile completed90.86 s. Root launched phase diagnostic /tmp/prefill_w4_phase_gpu.log to measure consumer wait/dequant/MMA fractions before implementing producer-side decoded shared staging. No duplicate GPU jobs. Completed key raw experiment logs copied under research_harness/results/reviewed/ for durability.

Prefill W4 phase diagnostic completed18 finite/bitidentical cases, no watchdog errors; /tmp/prefill_w4_phase_gpu.log, app ap-OSp3tPrKiIoVId6CvXo6Fm. Wide16K skew consumer fractions: dequant27.3%, MMA issue/retire31.9%, wait14.1%; producer wait-empty759 cycles/stage, issue448. Diagnostic instrumentation perturbs execution, not a performance measurement. This justifies a bounded producer-dequant experiment but does not prove it wins (four producer warps vs eight consumers may move the bottleneck). Prefill expert has GPU allocation for grid-constant A/B first: session16379, /tmp/prefill_grid_constant_gpu.log,240s bound,18 fresh paired cases. Do not duplicate.

MoE grid-constant exact-source check1295591092534f60 returned duplicate_of041ed31d40cf4296. Scored job91e2025666e54aec launched to remeasure the small gain supported by three fresh paired seeds and SASS attribution; no claim of aggregate significance from the check alone. Prior default preserved pending result. Whole grid variant remains unscored because its check tied/slightly regressed within uncertainty.

New upstream research direction: public openkernels advanced from d2057db to47103fd9c1212868018eb615a4c6467e366e9ce0. Separate pinned checkout `/tmp/muse-openkernels-47103fd`; no scored source overwritten. Attention expert audits v75/S1K, MoE expert audits v52–55, prefill expert audits pf32. Root integrates only rules-compliant validated components, retaining current embedded projections and BF16 folder ownership. See OPENKERNELS_RESEARCH.md. GPU currently idle; do not launch before component audits/CPU builds.

## Claude Code takeover (2026-10-27 late, after Codex usage limit)
- Attention: submitted conservative v75 f16pv (check 86386fd15bdb45fe, 1.944x CI 1.914-1.980) as scored job 7da917a2ea564131 -> 1,942.8 us (1.939x), from 2,029.27. Promoted to submission.py; backup submission_before_v75_7da917a2ea564131.py. FP8-PV v75 check 7a50c7bf7ca043e0 = 1,910.5 us, CI overlaps f16pv and precision margin is thin; not submitted.
- Upstream openkernels advanced to e6b48c833d8c (checkout /tmp/muse-openkernels-e6b48c8): MoE v56 (8-op e2m1 dequant, wx UDESC), attention v81, decode_layer v35. Unmodified upstream v56 MoE check 24c4d0f8e8944e4c = 6,809.2 us (1.975x) vs our scored 7,675.9: d32 155, v16 224, v32 260 us. Its judge-GPU tune picks gk(KS2=1,MAXM=8) M<=8, gk(KS2=2) M<=16, r1 M24-64, wx NCW16 UDESC M96-192, wx NCW16 N8STEP M>=256.
- research_moe_v56_strict.py (builder research/build_v56_strict.py): v56 + our strict BF16 boundary edits replayed hunk-exactly (CUDA 6, CUDA_L 10, CUDA_G 5, CUDA_K 7 hunks; CUDA_X WX52 strict edits), BF16 effective norm weights, frozen tune table (TUNE=False). Check 97a2667c85984f5b pending.
- Whole decode: Codex whole-frontend route probe (app ap-uVa9aO0F17EtOr1fO6Ai04) finished: strict WX52 vs grid on actual decoder routes M32 1.47x (sliding+full), M256 1.06/1.13x, M128 0.96/0.99x. Built candidate_cp14_v75_f16_wx_moe.py (builder build_wx_route_combo.py; WX for M 17-64 and 192-512, grid otherwise; non-X MoE CUDA verified identical). Check 79366ecf17244d59 pending. Upstream dl v35 unmodified check 41d729f88b184002 pending (headroom reference only).
- RESULTS: strict v56 MoE check 97a2667c85984f5b 6,822.1 us (1.971x; strict boundaries cost ~0 vs upstream 6,809) -> scored 4a60fa38515e4581 = 6,827.3 us (was 7,675.9). Promoted; backup submission_before_v56strict_4a60fa38515e4581.py.
- Whole WX-route check 79366ecf17244d59 13,360 us -> scored 2f8b9a4afddc43e0 = 13,270 us (was 13,940). Promoted; backup submission_before_wx_2f8b9a4afddc43e0.py. Upstream unmodified dl v35 check 41d729f88b184002 = 12,540 us (1.804x) for reference.
- Whole + strict v56 MoE (candidate_cp14_v75_f16_v56moe.py, build_v56_combo.py; in-layer dispatch gk M<=16, wx NCW16 UDESC M17-192, N8STEP above) check cdf34cb4f1394eec 12,680 us (1.785x, CI 1.758-1.811); scored a6986c85974540a5 pending. B32 sliding decode now 1.358x (was 1.005x).


## Codex resumed checkpoint (2026-10-28 06:45 UTC)

Reconciled shared Claude takeover before launching any duplicate jobs. Attention conservative scored 7da917a2ea564131 verified 1942.7804 us and already promoted by shared worker. Native FP8 v75 check 7a50c7bf7ca043e0 passed 1910.4528 us; no scored submission from Codex. Targeted native FP8 app ap-ApdExlLDgwo1XSrD7Oxhel completed all 27 actual-branch inputs x3 variants (81 canonical finite/cache checks); worst row 0.1446 vs bound 0.150. Precision audit and raw logs persisted under attention research_artifacts/v75.

Prefill grid scored 0d00b0439c5d4a54 passed 268118.1033 us, 2.29% better than274396.6272; Codex promoted exact registry-hashed candidate with backup submission_before_grid_0d00b0439c5d4a54.py. PF32 hybrid rejected: 18 cases pass but current-grid comparison ties/regresses; no check. Distinct n64 33..64-expert tail kernel approved CPU resource proof only, expert owns it.

MoE actual frontend app ap-uVa9aO0F17EtOr1fO6Ai04 completed; surrogate router seed5353 explicitly not hidden judge seed. WX52 wins M32/M256, loses/ties M128; shared worker has integrated. Newer strict v56 official check97a2667c85984f5b=6822.0550us, scored4a60fa38515e4581=6827.2569us, already promoted by shared worker. Whole WX scored2f8b9a4afddc43e0=13269.3578us, already promoted. Whole strict-v56 checkcdf34cb4f1394eec=12675.5622us; scored a6986c85974540a5 RUNNING. Do not duplicate submit or overwrite default while that finishes.

Codex agents resumed after usage reset. Attention audits minimal v81 sliding FP8 delta; omit parked/deadlocking kernels. MoE audits new strict v56 + next mechanism and peer reviews n64. Prefill CPU-only n64 resource proof. Root coordinates source-to-job identity, scored promotions and statuses. BF16 folder remains untouched. Harness9 CPU regression tests pass. Whole builder --fp8-pv creates isolated candidate_cp14_v75_fp8_grid_moe.py, but not checked because newer v56 whole is ahead.
- Whole + strict v56 MoE scored a6986c85974540a5 = 12,710 us (1.779x). Promoted; backup submission_before_v56moe_a6986c85974540a5.py. Live ranks after: overall #228 (was #345), whole #227, MoE #263, attention #260. Top50 cutoffs ~10,638 us whole, ~5,759 us MoE, ~1,455 us attention; not reached. No Modal GPU used this session.
- Next: whole gap to upstream dl v35 is only d1/v1 (projection glue + s1k PDL launch); remaining top-50 gap ~16% needs new kernel mechanisms (MoE issue-efficiency, attention full-context B16/32).

## Whole decode: v35 base + wx M32 pin (Claude, 2026-10-28)
- Per-shape diff (a6986c85 vs upstream v35 41d729f8): upstream wins d1/d16/v1/v16, ours wins only d32 (MoE wx route at M32; upstream pins fs f1s3).
- candidate_v35_wxpin32.py = unmodified v35 + DL_PIN {32: wx NCW16 WXG2 UDESC}, margin 0. Check f8794cd650f845d5 12,463.6 us (1.815x); scored 5ab92e7e91d64475 12,484 us (1.812x), rank #227 -> #210. Promoted; backup submission_before_v35wx_5ab92e7e91d64475.py.
- Remaining vs our old cp14 path: v16 (300/411 vs 280/399, maybe run noise). Top-50 cutoff ~10,638 us still 15% away.
- Follow-up (Claude): normalised per-shape ratios (ours/SGLang, same run) show the old path's verify-b16 "advantage" was SGLang run variance (0.555 vs 0.560 sliding, 0.580 vs 0.575 full). Best-per-shape dispatch of old cp14 vs v35wx bounds at +0.5% (v16 glue swap alone +0.06%): below the 1% rule, no check spent. No verified MoE/attention win newer than v56-strict/v75 exists to integrate (MoE check 9224772201374f0a 7,066 us is worse than 6,822).

## Claude MoE/attention session (2026-10-28)
- Attention PARKED (scope moved to MoE; full-context B16/32 attention now owned by GEMV agent): muse_attention_decode/candidate_v75_fp8off4.py (builder build_fp8off.py) = v75 FP8-PV + FA3-style probability offset 2^4 (P below 2^-9 of row max no longer flushes; matters for dist=coherent tails). Built, NOT checked/validated. Unvalidated precision claim.
- MoE W2 L2-prefetch for gk M<=8 (research_moe_w2pf3.py, check 9224772201374f0a): LOST, d1 30.2->33.4, v1 73.2->82.7 (7,066 us total). Judge-GPU A/B of prefetch timing/policy variants in research_moe_kdiag.py (prints tune lines).
- Judge-GPU A/B (check ec07005dcea340ff, research_moe_kdiag.py, 3 passes x 3 judge-generator seeds, cold-L2 graph medians, repeatable to ~1 us): gk M1 base 35 / W2PF variants 38-41 / SG=3 35; M8 base 79 / W2PF 83-89 / SG=3 80. W2 L2 prefetch REJECTED in every timing/policy form: HBM is not idle during the M1/M8 W13 phase. Method note: prepare-time A/B on the judge GPU via the _tune harness is a cheap, reliable substitute for Modal (one check, many variants).
- moe_x timeline diag (check a9f80a16926c4d23, research_moe_xdiag.py, PROF in prepare): at M128/M256 the serial xgather pre-kernel took 9.8/11.3 us (O(R^2) per-warp rescans), the moe_x tail (max - mean consumer end) is 10-14 us, the post-W2 combine about 7 us, and setup 4-5 us.
- O(R) xgather (research_moe_xg2.py, builder research/build_xg2.py; output bytes identical): check 061e24b8ef934b9d PASSED at 6,756.3 us (1.991x, CI 1.961-2.020) vs strict v56 6,822. xgather 9.8->6.6 us (M128), 11.3->7.5 us (M256); v16 214.5, v32 252.1 us. The ~1% total gain is inside the CI, so it was NOT submitted. Nothing is in flight.
- NEXT: (1) launch xgather over all SMs (row segments; only 32-64 blocks copy now), (2) PDL xgather->moe_x: producer pre-issues the first NSL weight slots, then griddepcontrol.wait before the activation TMA, (3) K-split the last-wave units to cut the 10-14 us tail. All are local to CUDA_X; whole-decode moe_x (M17-512) benefits identically.

## Attention full-context b16/b32 (Claude, stopped on usage limit)
- Probe check 82ccf4fc000948b6 (muse_attention_decode/probe_gather_ceiling.py): random 128 B K+V row gather ceiling ~3.9 TB/s (B16 84 us raw, B32 163 us; sequential 4.1-4.3 TB/s). Current fdec: judge 110.7 / 201.5 us (~3.0-3.3 TB/s), so ~20-25% headroom. fdec already uses mma16 tensor cores; suspected bound is in-flight bytes (NS=8/4 -> 128 CTAs x 8 warps x 4 ring blocks = 128 KB/SM).
- IN FLIGHT: sweep check 36749dbf05214f06 (muse_attention_decode/probe_fdec_sweep.py): fdec variants base / RBF=6 / PFF=4 / PFF=8 / RBF6+PFF4 / balanced (fl|8) / balanced+PFF4 on synthetic full B16/B32; results print as "FSWEEP" lines in profiles/probe_fdec_sweep.36749dbf05214f06.log. Risk: 5 extension builds may hit the 10-min run limit.
- Next: if a variant beats base by >5%, set it in the attention candidate (RBF/PFF via _ext flags, DFL for balanced) -> standalone attention check -> port the same flags into whole decode candidate_v35_wxpin32 (its A__ v81 fdec) -> whole check; submit only at >=1% gain.


## Attention sweep recovery (Codex, 2026-10-28)
Authoritative sweep36749dbf05214f06 DONE FAILED return112, nottimeout. Firsterror ScatterGather indexoutofbounds: synthetic seq up to85999 but pt capacity80008; before fdec launch. No variant speed evidence. Attention expert repairs isolatedprobe, prefillpeerreviews, rootallocates oneOKCcheck. No active Modalapps orOKCjobs atresume. Userreports ModalGPU1782.6/1800s; HOLDALLNEWModalGPUlaunches (remaining17.4s not enough for controlledexperiment). CPU/sourceproof only. Exactlimits attention16checks,MoE14,whole14,prefill19;BF16reservedotherchat. Whole5ab92e7e91d64475 verified12484.0800us. Oldrootv81wholecheck2a18 passed12574.6707us but slowernewdefault, do notscore. RootlaunchfusionGPUtensorpointerfix remainsunvalidated, donotintegrate; local_start is GPUint32[1], nothostconstexpr. MoEroundstoreCPUcompleted225s, noGPUrunverified.


## Rules audit: v35wx withdrawn and strict default restored
Root/ MoE source audit confirms live embedded M_CUDA_K and M_CUDA_X in candidate_v35_wxpin32.py bypass BF16 gate/up/down intermediate boundaries required TASK103. GK factors W2 GS into norm; WX rounds only after normweight. nw effective weights remainFP32. Passingchecks is insufficient. Root withdrew scored5ab92e7e91d64475 and restored exacthash31d5bc39 candidate_cp14_v75_f16_v56moe.py (scored a6986c85974540a5=12714.4816us) tosubmission.py. Noncompliantdefaultbackedup submission_before_compliance_revert_5ab92e7e91d64475.py. Do not resubmit/promote oldv35wx. MoEexpert ports strictboundaries into v35glue in separatecandidate; sourceaudit and check required.

## BF16-chat takeover support review (2026-10-28)
No checks or ModalGPU jobs launched here; coordinating Challenge chat owns repaired attention probe. Independent peer review: `muse_attention_decode/research_artifacts/fdec_sweep_peer_review.md`. Fixed RBF5 geometry/reference/cache/ABBA pass source review. Before launch consider compact final summary+raw arrays (16KB logtail otherwise loses earlier cases) and compile-admission deadline (510s check currently only surrounds timing). RBF6 dormant source has omitted rpro[3] new-token repair hazard; fixed probe does not select it. MoE support builds a separate no-PDL vector-store gather from strict xg2; will leave mapping proof/review under research/XG_VECTOR_REVIEW.md. BF16 cp17 preserved.

Corrected attention diagnostic CHECK0653d51aad4c4281 launched after independentpeerreview; capacity86008, full-keycanonical/cachechecks andcoldL2ABBA, 3uniquemodulebuilds, controlledbudget. NoModalGPUused. Attentionagentchanged320/330s precompileguard nearupload; freezeallsource now, results/markerloggedversionauthoritative; no duplicatecheck. This diagnostic must neverbe scored/promoted. Strictv35 repair candidate ready8dc4c65e..., independentreviewpending thenonewholecheck.

### Support artifacts ready; BF16 check owned here
MoE no-PDL candidate `muse_moe_w4a16/research_moe_xg_vector.py` SHA c98d6791625b6a3587e169e6a2fc33090616fa2db06b665d0952aff974546eaf is source-only: CPU96row-byteproofs and independent peer review PASS; no CUDAcompile/GPU/timing. Builder research/build_xg_vector.py; notes research/XG_VECTOR_REVIEW.md. Has all-SM segment control XGSEG=1 vs4 and optional XGVECTOR; preserve smallM dispatch. Gather-only ceiling is~1.03%overall, so do not claim top50 from it; requires paired harness beforecheck. No new MoE jobs here.
BF16 own CW8 pooler check326978ecb91b416d RUNNING (CPU/layout/releasepeerreview cleared), canonicalcp17 unchanged. No ModalGPUjobs. Do not duplicate this check.


## Corrected sweep completed; no validated winning setting selected

Check **0653d51aad4c4281** passed all ordinary judge checks at1939.4813us (the normal kernel is unchanged). Diagnostics finished in257.98s,3modules. Job output API retains only the last16,000characters, so only five complete FSWEEP_RESULT rows survive; earlier B16 rows and some B32 rows are missing. Do not infer their performance or duplicate the run solely for observation loss. Visible B32 seeds16/113: PFF4/PFF8/balancedPFF4 speedup~0.565–0.569x, raw202→353–357us; balanced0.9923x,201.92→203.69us. These settings are rejected forB32; there is no evidence to build/promote a faster kernel from the retained results. Raw retained rows in research_harness/results/reviewed/fdec_sweep_fixed_results.json. Future diagnostic probes must print a compact final all-case summary after raw samples so the16KB service tail preserves decisions.

Strict v35 repair **474298298c394dd8** passed all checks/no flags at12,499.2808us, source8dc4c65e unchanged. Scored job **5648767e139d441b** is running; keep existing stricta698 default until a scored improvement is verified. No ModalGPU time spent this continuation.

Strict v35 scored **5648767e139d441b** PASSED/no flags at **12,501.6459us /1.80939x**. Promoted exactSHA8dc4c65e candidate_v35_wxpin32_strict.py to whole submission.py; prior strictdefault backedup submission_before_strict_v35_5648767e139d441b.py. The withdrawn noncompliant v35wx score remainswithdrawn. Relative to previousvalid12714.4816us, score improves1.67%; relative towithdrawn12484.08, strictrepaircosts17.57us/~0.14%. GoalstillnotTop50.


## Budget-aware research checkpoint (2026-10-29)
No live OKC jobs or active Modal tasks at authoritative inspection. Defaults unchanged. Completed cached fdec SASS proves both heads' QK work is already hoisted before first softmax; proposed source interleaving is a no-go. CPU k32 FP8 warp MMA also converts to FP16; no native throughput benefit. No GPU/check allocated to either. Evidence: attention research_artifacts/fdec_head_interleave/ and fdec_fp8_mma/.

MoE source model WHOLE_ROUTING_OVERLAP_MODEL.md confirms a potential side-stream metadata window behind pooler, but guaranteed slack and removable classification cost remain unmeasured. Root allocated preparation/peer review of one compact external-only diagnostic, not its launch. Preserve exact BF16 math and current strict v35. Do not implement metadata ownership/consumer changes until phase evidence supports them. Future probe summaries must stay below2.8KB at the end of logs. Modal GPU hold remains in force.
