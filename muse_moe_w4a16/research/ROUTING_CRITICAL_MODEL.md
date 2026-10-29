# Source-only routing/tail model after xg2

No GPU/check allocation. Preserve current scored standalone v56 and strict whole repair. The all-SM row-segment gather idea is parked, not implemented/promoted.

## Aggregate ceiling

TASK42 uses the geometric mean of six equally weighted loads. Check061e24b8ef934b9d medians are214.497us/M128 and252.145us/M256; profiles attribute6.6and7.5us to xgather respectively. Removing those gathers completely would improve aggregate by only `(214.497/(214.497-6.6)*252.145/(252.145-7.5))**(1/6)`, approximately1.0103x. A nonzero replacement has less headroom. Profile/timing noise makes a gather-only >1%gate especially unconvincing.

Segmenting each8192byte routed BF16 row into eight independent1024byte segments would launch enough tasks to use all132SMs at M128/M256. Each segment covers64of512uint4 loads, retains the existing global q4 index in kperm/xaddr, and writes the identical four32bit words per load. Segment q4 ranges are disjoint and their union equals the original[0,512), so no additional or omitted bytes, no rounding, and no inter-segment synchronization is needed. Read+write traffic remains16384bytes per local route. Actual whole-router surrogate loads167/326localroutes therefore require2.736/5.341MB of gather traffic. Each CTA would still repeat the stable-ID O(R) classification, so copying faster does not remove the setup floor. No official check is justified by this bound.

## Structural candidate worth profiling first

Produce one stable route table per timed invocation and reuse it across all SMs and phases, rather than independently recomputing match/prefix/scatter in every gather and GK/WX CTA. Table fields can be localID[R]u8, stable position[R]u16, sorted routeID[R]u16, plus <=1KiB of counts/offsets/padding. Total<=1KiB+5Rbytes (R=M*8), at M256 <=11KiB. Every entry depends on fresh routing and must be produced inside timed work. No activation or route output may be cached in prepare.

Whole integration could extend an existing routing kernel's timed last-arriver epilogue, avoiding a separate launch. Standalone requires a timed producer or a proven resident-grid synchronization scheme; an extra metadata launch/serial phase can erase savings. The consumer must acquire the readiness publication before reading any field, and release must follow all metadata writes. A PDL early launch signal is not data readiness. Same-stream next-invocation ordering and counter reset must remain correct. Do not use a CPU-read histogram or route-dependent host branch.

This mechanism affects all six MoE loads, unlike gather-only changes. It does not change BF16 gate/up/inter/down boundaries or partial-K accumulation order. GK can rebuild its local shared route ownership from the same stable route positions; WX can use the same list/padded offsets as its gathered activation bank. Any common table needs exact ownership proofs for absent experts, zero-output tokens, all nearby token counts and arbitrary legal local_start.

To clear a3%aggregate gate, an example required net savings vector is about[2,2,3,4,5,6]us on M[1,8,16,32,128,256]. This is a threshold illustration, not a performance prediction. Existing PROF setup timestamps must first demonstrate at least that much headroom after accounting for the producer, synchronization, and metadata loads. Without measured phase timings, a >3%prediction is unsupported.

## Other limits

GK already reuses each loaded expert weight tile across its routed tokens. Latest upstream evidence puts small-load weight-stream floors around17/54/79us for M1/8/16 at4.35TB/s. Further unpack instruction reductions are unlikely to produce a large factor after v56. The completed round-store CPU variant saves one CTA barrier with zero spills but remains unmeasured.

WX final combination waits for all W2 work and reads BF16 per-route down outputs. Per-route readiness could overlap combination with stragglers, but token outputs depend on every local route; heavy-first W2 ordering reduces predictable overlap. Summing separate BF16 partial-route outputs would add a forbidden extra boundary. A hybrid must retain FP32 route accumulation and one final BF16 rounding.

Moving a full norm reduction to one owner instead of duplicating it on all GK CTAs may reduce traffic/instructions but adds publication/serialization. It needs measured norm-tail duration first. No assumption that less total work means lower latency is justified.
