# Can doubled splits create two resident fdec CTAs per SM?

No, with the current kernel. This is a source/resource audit, not a GPU result.

## Resource bounds

The launch requests `max(8*RBF*4096 +256 +8*2048 +2048, (512+256*64)*4)` dynamic shared bytes per CTA. At RBF5 that is182,528B. Two CTAs require365,056B before static/reserved shared memory, beyond Hopper's228KiB/SM. Current compiled fdec<false,true> uses157 registers/thread, no local/stack spills (persisted resource log). At256 threads this is40,192 registers/CTA before allocation rounding; two exceed Hopper's65,536 registers/SM. Two resident CTAs would require at most128 registers/thread as well as a smaller shared allocation.

RBF3 still requests116,992B/CTA; twice233,984B already exceeds228KiB=233,472B before static/reserved memory. RBF2 requests84,224B but halves the outstanding useful KV stages: twoCTAs ×(RBF-1)×8warps×4096B=65,536B, versus oneRBF5 CTA's131,072B. Reducing RBF3's gate buffer could approach the shared limit but would require a resource/occupancy proof and register reduction; it is not obtained by changing NS. Forcing FDMINB2 can cause register spills; it is not evidence of a faster consumer.

Primary resource limits: [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), occupancy/shared-memory sections. Actual residency must ultimately be queried with the CUDA occupancy API.

## Merge liveness

The current global merge is not last-CTA-only. Every request's last nmg arrivers (nmg largestpoweroftwo≤min(nsp,16)) stays resident and spins on its request counter until allnsp splits arrive. Earlier non-mergers return. Doubling B16 splits8→16 or B32 splits4→8 means nmg=nsp: every CTA can spin. Grid size becomes256 while existing occupancy only permits132residentCTAs. There is no scheduling-independent guarantee that each request's remaining split can launch. A possible schedule with fewer than nsp waitingCTAs per request can fill residency and block its own missing producers. Contiguous block-index scheduling often completes a request, but CUDA does not provide that liveness contract.

Current B16/B32 grids contain128CTAs and fit one wave; runtime balanced fl8 explicitly caps its grid at CUDA occupancy. Cluster mode is disabled for these enlarged grids and cannot supply a co-scheduling guarantee. Existing whole-layer PDL does not repair this merge dependency cycle.

Therefore a larger grid needs either proven twoCTA residency or a merger that does not retain multiple waiting CTAs (e.g. a genuine last-arriver writer or a separate merge launch). Those alter resource/scheduling tradeoffs and need their own correctness/performance evidence.

## Existing negative evidence

Pinned upstream e6b48c8 docs/ideas.md reports RBF4/5/6 tied on H100 full decode (B16~138–139us,B32~260–263us), and balanced fdec fl13 had no gain. Neither experiment actually establishes twoCTA residency. An earlier3090 read+merge model reports B32/80K NS4~870GB/s vsNS8~760GB/s; this excludes attention math and is not an H200 kernel verdict. No direct successful H200 two-resident-fdec result was found. Hence there is strong source evidence against the simple NS-only proposal, and limited measured evidence against deeper rings/more splits, rather than a claimed measurement of the unbuilt twoCTA redesign.

Practical bandwidth headroom remains small: B32/80K minimumKV+table traffic~665.6MB/3.87TB/s≈172us versus~201us, before prologue/merge costs. Root's actual whole-layer weighted model bounds20–25%fdecB16/B32 reduction at~1.24–1.56%whole improvement. No GPU/check launched for this audit.
