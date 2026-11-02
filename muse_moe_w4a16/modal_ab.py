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
app = modal.App("muse-moe-ab", image=image)


@app.function(gpu="H200", timeout=1500)
def run(cands: str = "submission,candidate_ours_v3", shapes: str = "1,16,32,8,128,256"):
    import sys, time, importlib
    sys.path.insert(0, REMOTE_DIR)
    import torch
    import reference as R
    w = R.generate_weights(1)
    mods = []
    for c in cands.split(","):
        S = importlib.import_module(c)
        t0 = time.time()
        st = S.prepare(w)
        print(f"[ab] {c} prepared {time.time()-t0:.0f}s", flush=True)
        mods.append((c, S, st))
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int32, device="cuda")
    for bs in [int(s) for s in shapes.split(",")]:
        d = R.generate_input(bs, "ep8", 777 + bs)
        graphs = []
        for c, S, st in mods:
            o = S.kernel_fn(d, st); torch.cuda.synchronize()
            err = R.check_implementation(d, o, w)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                S.kernel_fn(d, st)
            graphs.append((c, g, err))
        ts = {c: [] for c, _, _ in graphs}
        for rep in range(60):
            order = graphs if rep % 2 == 0 else graphs[::-1]
            for c, g, _ in order:
                flush.add_(1)
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); b.synchronize(); ts[c].append(a.elapsed_time(b) * 1000)
        print(f"[ab] bs {bs} " + "  ".join(f"{c}: {sorted(v)[30]:.1f}us" for c, v in ts.items()) + "  check " + str([str(e)[:60] for _, _, e in graphs]), flush=True)


@app.local_entrypoint()
def main(cands: str = "submission,candidate_ours_v3", shapes: str = "1,16,32,8,128,256"):
    run.remote(cands, shapes)
