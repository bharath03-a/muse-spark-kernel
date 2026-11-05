# Final results — Nov 4 2026 (submissions closed 11:59 pm PT)

## Standings

| Board | Best | Entry | Code |
|---|---|---|---|
| muse_decode_layer (= overall rank) | 2.258x, #237 | `candidate_last_v5.py`, job 4f006c820954441e | credited decoder + OURS changes |
| muse_bf16_proj | 1.4814x | own lineage | ours |
| muse_moe_w4a16 | 2.384x | credited (openkernels-derived) | credited |

Own MoE (`muse_moe_w4a16/own_moe_v5.py`, hand-written CUDA): 1.61x on judge check 2c1a0498b43a4f9b (v3 was 0.68x). Not submitted: below the credited entry.

## What landed in the best decoder (`candidate_last_v5.py`)

Base `candidate_hc_v1.py`, plus:
- Sliding attention B32 q1 `n_splits=12`: decode32 sliding 148 -> 143.5 us.
- Router reads only the BF16 hi half (T<=8 and the fused pooler+router path at T16/T32): ~0.4-1 us per load, expert selection still passes 13/13.
- MoE M=8 dispatched to the MID build (`MOE_TINY2_MAX_M=4`): verify1 ~113 -> ~100-109 us (part of the best reading was run noise).

## Measured no-go (Nov 3-4)

Details and job IDs in `HILLCLIMB_LOG.md`.
- Own BF16: zip configs, fused hi+lo router (v3), persistent multicast swap-AB wgmma (v2, 2-7x slower).
- Own MoE v5 hybrid in decoder: 1.5-2.2x slower than the in-decoder fused MoE at every T.
- Launch-gap / second-stream audit: no idle time; ceiling +0.64%.
- Knob sweeps: decode T16/T32 MoE + attention, verify T8/T128 MoE + attention, projection PCX/GLE/EARLY, T1 MoE dispatch, MID at T128/256 (1.94x), B4FIN at T16.
- FP8 activations / FP8 wgmma: weight-read bound; lossy weights rejected (tolerance + review risk).
- Glue op-merge: kernels already at bandwidth limit.
- Attention mask structure: already exploited by the credited kernel.
- Public repos (AutoclickerI, alygg77, Ranoobaba, lilaclilac09, rohanarun): best 2.206x, none usable.

## Lessons

- Judge noise is about ±1.5% per run; single-check wins under 1% are not real. Judge per-load us, not the total.
- Byte-identical resubmits dedupe; cosmetic re-rolls risk review flags, so we never used them.
- The remaining gap is inside the fused MoE kernel (decode32 sliding: 146 us vs 84.5 us roofline). Beating 2.3x needs a new MoE kernel, not tuning.
- Check budget (40/day) ran out on the last day; plan checks per agent up front.
