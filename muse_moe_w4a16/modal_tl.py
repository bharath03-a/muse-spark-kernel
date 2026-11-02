import os
from pathlib import Path
import modal

PROJECT_DIR = Path(__file__).parent
REMOTE_DIR = "/root/proj"
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    .run_commands(
        "echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
        "echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
    )
    .entrypoint([])
    .add_local_dir(PROJECT_DIR, remote_path=REMOTE_DIR, ignore=[".venv", "__pycache__", "profiles", "*.pyc", ".git"])
)
app = modal.App("muse-moe-tl", image=image)


@app.function(gpu="H200", timeout=900)
def run(cand: str = "submission", shapes: str = "256,128"):
    import sys, time, importlib, statistics
    sys.path.insert(0, REMOTE_DIR)
    import torch
    t0 = time.time()
    import reference as R
    S = importlib.import_module(cand)
    S.TINY_DEFINES = S.TINY2_DEFINES = S.MID_DEFINES = ()
    S.PS_SHAPES = {}
    S.CUDA_SRC = S.CUDA_SRC.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif', 1)
    w = R.generate_weights(1)
    st = S.prepare(w)
    dev = "cuda"
    G = st["grid"]
    dbgdef = tuple(S._main_defines(st["sfx"])) + st["sfxm"] + ("-DOKC_DEBUG=1",)
    ext = S._ext(dbgdef)
    base = S._ext(tuple(S._main_defines(st["sfx"])) + st["sfxm"])
    print(f"[tl] prepared {time.time()-t0:.0f}s", flush=True)
    for bs in [int(s) for s in shapes.split(",")]:
        d = R.generate_input(bs, "ep8", 4242 + bs)
        ids = d["topk_ids"]
        print(f"[tl] bs {bs} distinct local {int(torch.unique(ids[ids<32]).numel())} routes {int((ids<32).sum())}", flush=True)
        x = d["hidden_states"]; idt = ids.to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
        def f(): base.okc_run(st["w"], st["scratch"], x, idt, tw, out, 0, 32, G)
        g = torch.cuda.CUDAGraph()
        f(); torch.cuda.synchronize()
        with torch.cuda.graph(g): f()
        ts = []
        for _ in range(30):
            flush.add_(1)
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
        print(f"[tl] bs {bs} graph median us {sorted(ts)[15]:.1f}", flush=True)
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(st["scratch"]) + [dbg]
        for rep in range(3):
            flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
            ext.okc_run(st["w"], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        tt = dbg[:G * 8].view(G, 8).cpu().double()
        t00 = tt[:, 0][tt[:, 0] > 0].min()
        print('[tl] bs',bs,'perCTA A,B,fin:',' '.join(f'{int(i)}:{(tt[i,2]-t00)/1000:.0f}/{(tt[i,3]-t00)/1000:.0f}/{(tt[i,5]-t00)/1000:.0f}' for i in range(G)),flush=True)
        for k in range(8):
            col = tt[:, k][tt[:, k] > 0]
            if col.numel():
                c = (col - t00) / 1000
                print(f"[tl] bs {bs} slot{k} n={col.numel()} min {c.min():.1f} med {c.median():.1f} max {c.max():.1f}", flush=True)


@app.local_entrypoint()
def main(cand: str = "submission", shapes: str = "256,128"):
    run.remote(cand, shapes)
