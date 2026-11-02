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
