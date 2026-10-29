"""Judge-GPU timeline diagnostic of moe_x (wx): per-CTA consumer end times -> tail (load-imbalance) loss.

Hypothesis: ~uniform-cost units (256 rows x full K, ~19 us each at v32) pulled from one queue quantize the makespan to
ceil(U / 132) units, so the last wave idles most SMs. PROF build of the scored wx configs at M32 / M128 / M256 on
judge-generator routings; prints quantiles of per-CTA consumer-loop end, combine end and the graph-replay time.
Kernel numerics and the scored dispatch are unchanged (prepare-only diagnostic).
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / 'research_moe_v56_strict.py').read_text()
text = text.replace('"moe_v56strict_v5', '"moe_xdiag_v5')
text += '''

# ---- Research diagnostic (build_xdiag.py): moe_x per-CTA timeline on the judge GPU, printed from prepare().
_prepare_scored = prepare


def _xdiag(st):
    dev = st["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    nth = 17  # NCW 16 consumer warps + producer
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda c: _extX(c), [("-DNCW=16", "-DWXG2", "-DUDESC", "-DPROF"), ("-DNCW=16", "-DN8STEP", "-DWXG2", "-DUDESC", "-DPROF")]))
    for M, cfg in ((32, ("-DNCW=16", "-DWXG2", "-DUDESC")), (128, ("-DNCW=16", "-DWXG2", "-DUDESC")),
                   (256, ("-DNCW=16", "-DN8STEP", "-DWXG2", "-DUDESC"))):
        for seed in (21, 22, 23):
            data = _synth(M, seed + M, dev)
            ids = data["topk_ids"]
            cnt = torch.bincount(ids[(ids >= 0) & (ids < 32)].flatten().long(), minlength=32)
            na = int((cnt > 0).sum()); npas = ((cnt + 63) // 64)
            units = int((npas * 32).sum() + (npas * 16).sum())
            prof = torch.zeros((nsm * nth * 16,), dtype=torch.int64, device=dev)
            st["profW"] = torch.empty(0, dtype=torch.int64, device=dev)
            _wx(data, st, cfg); torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr): _wx(data, st, cfg)
            ts = []
            for _ in range(9):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                sink.copy_(flush.sum()); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
            del gr
            st["profW"] = prof
            pcfg = cfg + ("-DPROF",)
            _wx(data, st, pcfg); torch.cuda.synchronize(); prof.zero_()
            sink.copy_(flush.sum()); torch.cuda.synchronize()
            _wx(data, st, pcfg); torch.cuda.synchronize()
            st["profW"] = torch.empty(0, dtype=torch.int64, device=dev)
            pr = prof.view(nsm, nth, 16).double()
            t0 = pr[:, :, 0][pr[:, :, 0] > 0].min()
            cend = (pr[:, :16, 2].max(1).values - t0) / 1e3
            fin = (pr[:, :, 7].max() - t0) / 1e3
            setup = (pr[:, :, 1].max() - t0) / 1e3
            q = torch.quantile(cend, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64, device=dev)).tolist()
            print("XD M%d seed%d experts %d units %d (%.2f/SM) replay %.1f us | setup %.1f | consumer end p0 %.1f p10 %.1f p50 %.1f p90 %.1f max %.1f mean %.1f | combine end %.1f"
                  % (M, seed, na, units, units / nsm, sorted(ts)[len(ts) // 2], setup, *q, cend.mean().item(), fin), flush=True)


def prepare(weights):
    st = _prepare_scored(weights)
    try:
        _xdiag(st)
    except Exception as ex:
        print("XD failed", repr(ex)[:300], flush=True)
    torch.cuda.empty_cache()
    return st
'''
out = root / 'research_moe_xdiag.py'
out.write_text(text)
compile(text, str(out), 'exec')
print(out)
