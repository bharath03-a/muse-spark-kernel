# Strict GK round-at-completed-W2-writer

Candidate `research_moe_v56_round_store.py`; reproducible builder `research/build_gk_round_store.py`; original scored v56 unchanged.

For KS2=1 each W2 callback receives v after all K pieces and the same SW reduction; its active expert index a satisfies rexp[aoff[a]+slot]=a. The callback now writes FP32(BF16(v*gs2[a])) to the shared down-output region. For KS2>1, the existing ordered FP32 sum of every K partial t is unchanged; its final shared store now writes FP32(BF16(t*gs2[rexp[r]])) for the same valid output rows. Invalid padded shared rows retain the prior value. Multiplication and BF16 conversion remain in their original order.

The former shared read→scale→BF16→shared write loop is removed. Exactly one redundant CTA barrier is removed. The previous barrier, immediately after the complete W2 shared writers and before any norm warp reads another writer's rows, remains. No partial K accumulator is rounded. Norm multiplication, sumsquares and final route combination are unchanged. This preserves the full BF16 down-output boundary with no extra or omitted rounding.

Independent attention specialist review passed callback indexing, complete-K/SW reduction ordering, KS2 sum order, and required writer→norm barrier. Empty individual experts and tokens are covered in the planned test; the official busiest-rank contract ensures at least one local route. No all-empty-bank behavior claim is made.

CPU sm90a cache/SASS audit includes original and candidate for KS2=1/MAXM8 and KS2=2. Planned GPU probe is17 cases,24balanced AB/BA samples each, fresh seeds103/217/331 on M1/8/16, nearby2/3/7/9/12/15, and M8/M16 sparse-local cases with one active expert and most zero-output tokens. Both implementations use the same varied per-expert global scales and independent mutable scratch. FP32-reference/finite checks and replay guards precede all timing.

This is a small structural savings; a major speedup is not assumed. Existing GK already reuses each expert's loaded weights across its routed tokens. More scale/unpack ALU tuning has limited scope after v56. A distinct later hypothesis is testing128rather than132 persistent CTAs: W2 partitions become exactly2tiles/CTA for KS2=1 and4for KS2=2, avoiding the132-grid's last-wave imbalance, at the cost of leaving4SMs idle. That needs phase-profile evidence and is not implemented here.

CPU audit complete225.35seconds: KS2=1 registers82→80,BAR17→16,LDS201→183,STS54→49; KS2=2 registers117→117,BAR23→22,LDS336→340,STS88→83. Both variants STACK0,LOCAL0,LDL0,STL0. Cached binaries persisted in muse-ext-cache. No GPU run was launched; later GPU hard-budget exhaustion halted this experiment before measurement. Do not claim speedup or promote. Durable CPU log profiles/research_gk_round_store_cpu.log.
