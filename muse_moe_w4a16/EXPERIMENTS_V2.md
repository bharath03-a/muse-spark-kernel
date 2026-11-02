# MoE EXPERIMENTS_V2
ideas informed by vjeux/okc-2026, alygg77 (Apache-2.0)

- baseline analysis: current submission = upstream v56-v60 lineage (persistent gk/wx/ring kernels). Last scored runs 6,750-6,815 us (096b6e4e67804be3: d1 30.2, d16 112.0, d32 145.0, v1 79.2, v16 215.9, v32 264.5 vs floors 11.4/52.5/70.8/37.3/101.3/108.9). Farthest from floor: d1 2.6x, v32 2.4x, v16 2.1x, d32 2.05x.
- probe_moe_top50_v1 (3530b14afb4942cf, 6,713 us): PDL variants -9 us on v16/v32 only; all other knobs flat. Knob space in v56 lineage is exhausted.
- scored 8943fb06044340de: still queued for GPU (no result at time of report).
- alygg77 file benchmark-only check (alygg KPERM3, 2.40x claimed): launched, GPU queue backlog, result pending (stdout /private/tmp/claude-501/chk_alygg.json). Copy deleted per rule; never submitted.
- reimplementation (in-CTA routing, fat/skinny + stream-K tail, K-permuted scale load): NOT started.

## FOR WHOLE-DECODE
(nothing verified yet)

## 2026-11-01 update
- candidate_kperm.py (ours): wx WXG2 consumer, KPERM = slot-internal repack in prepare (_repackXK): [4cc x 8rec x 512B weights][8rec x 32lane x 4cc x 2B scales]; one ld.shared.v2 serves 4 chunks (replaces 4 lds16). PDLX added to the M>=128 wx configs. Compiles on docker sm_90a: 96 regs, 20B spill stores (baseline 16B). Check pending: /private/tmp/claude-501/chk_kperm.json (GPU queue stalled; 8943fb06044340de still queued).
- candidate_alygg.py (copy-then-modify now allowed): re-copied, header intact; benchmark check pending: /private/tmp/claude-501/chk_alygg.json. Not yet promoted.
- candidate_v2_pdl.py = alygg + PDL=True (kernel already has griddepcontrol wait/launch_dependents). check running, out /private/tmp/claude-501/chk_v2pdl.json. base 478c0e9ca2214849 = 5,676 us.
- candidate_v2_ns4.py = v2_pdl + WGS_NS 3->4, WGS_NX 3->2 (smem 144+64=208KB vs 204KB; barrier idx parametrized). Rationale: v16/v32 verify at 1.7-1.9x SOL, weight-bound; deeper weight ring = more bytes in flight. Expect up to ~3-5% on v16/v32 if latency-bound. Check out: /private/tmp/claude-501/chk_v2ns4.json
- v2_ns4 check 9496923fcafb436f: FAILED at runtime (CUDA invalid argument at first generate_input; likely dynamic smem NS4*36KB+NX2*32KB+sizeof(Smem) > 227KB limit). Not submitted. v2_pdl ae3cab0c2a884aec 5,682 us (no gain).
- ours_v1 (keell0renz + wsi issuer warp on bs1/bs16 builds via PS_SHAPES): judge 5,651.7 us (2.380x), flat vs 5,640.7; correct (rel<1e-5). No Modal.
- ours_v2 (v1 + OKC_NS=4 at bs1/bs16, main WGSFCI=6 at bs32): judge 5,690.3 us, slower (+0.9%); correct. Dead.
- EVIDENCE (job 958bd92a752e4369; floors from earlier notes, not re-derived; all 6 cases weight 60): case us/floor/gap/eff: d1 25.5/11.4/14.1/45%; d16 90.7/52.5/38.2/58%; d32 122.0/70.8/51.2/58%; v1(bs8) 66.9/37.3/29.6/56%; v16(bs128) 176.4/101.3/75.1/57%; v32(bs256) 212.2/108.9/103.3/51%. Largest gap: v32 (WGS main build, bs256).
- CAVEAT: first-principles byte count (32 experts x 28.3MB = 906MB) gives >=215us at 4.2TB/s for v16/v32, contradicting measured 176/212 and the stored floors, so the expert-hit count / byte model is unresolved; floors unreliable until the harness's real byte count is known.
- DIAGNOSIS not completed: needs per-SM timeline (%globaltimer/%smid) on H200 to separate ramp, A->B dip and tail; none available without GPU time. No structural change built (blind edit of the 6.9k-line WGS stream-K kernel judged <10% odds per judge check).
- GROUND TRUTH: floor = 0.75*distinct*28.3MB/4.8TB/s (reference.py), so TASK.md floors are conservative. Real distinct local experts (reference.generate_input, Modal): bs256=28, bs128=23, bs16=9. bs256: 792MB -> 189us at 4.2TB/s vs 222 measured graph (85% of achievable); bs128 ~85%.
- TIMELINE bs256 (main WGS build, Modal H200, modal_tl.py; us from first CTA start): route done 3.3; gather done/producers start 7.7 (ramp); A done 133-152 (med 141); B done 200-214 (med 207); final 206-219 (med 214). Steady BW (792MB/206us)=3.84TB/s. Handoff dip: none visible in aggregate (B duration 66us = 1/3 of A). Tail: last B-done to end ~5us; B-done spread +-3us, systematic (CTA idx>=59 finish A ~8us later, B ~5us later: cost-model skew). bs128: ramp 5.4, B med 171 max 175, final 174-178.
- Ramp (7.7) largest. ours_v3: weight producer pre-issues first 2 ring stages right after stage ranges (bar 8), before gather join. Modal A/B same container, 60 interleaved cold-L2 reps, graph-median us base vs v3: bs1 31.3/31.1, bs16 114.1/114.5, bs32 168.2/168.4, bs8 80.3/79.9, bs128 192.1/192.8, bs256 222.6/223.1; correct (rel L2 <=3.3e-3). FLAT -> ramp hypothesis falsified (weights are not the limiter during the gather). No judge check spent.
