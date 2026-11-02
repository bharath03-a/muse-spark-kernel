from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).parent
REMOTE_DIR = "/root/own"
FILES = ["reference.py", "checks.py", "nvfp4.py", "own_moe_v3.py", "own_moe_v4.py"]

image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    .run_commands(
        "echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
        "echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
    )
    .entrypoint([])
)
for f in FILES:
    image = image.add_local_file(PROJECT_DIR / f, remote_path=f"{REMOTE_DIR}/{f}")

app = modal.App("muse-moe-own-ab", image=image)

SCORED = [1, 16, 32, 8, 128, 256]
TESTS = [(4, "normal"), (33, "normal"), (257, "normal"), (128, "skewed"), (1, "normal")]
REPS = 40


@app.function(gpu="H200", timeout=1500)
def run(cands: str):
    import importlib
    import sys
    import time

    t_start = time.time()
    sys.path.insert(0, REMOTE_DIR)
    import torch

    import reference as R
    from checks import relative_l2

    w = R.generate_weights(R.WEIGHTS_SEED)
    mods = []
    for c in cands.split(","):
        mod = importlib.import_module(c)
        t0 = time.time()
        st = mod.prepare(w)
        print(f"[own] {c} prepare {time.time() - t0:.1f}s", flush=True)
        mods.append((c, mod, st))

    def check(mod, st, d):
        out = mod.kernel_fn(d, st)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, out, w)
        return ok, relative_l2(out, R.ref_kernel(d, w)), msg

    for bs, dist in TESTS:
        d = R.generate_input(bs, "ep8", 4242 + bs, dist=dist)
        for c, mod, st in mods:
            ok, err, msg = check(mod, st, d)
            print(f"[own] test bs={bs} {dist} {c}: ok={ok} relL2={err:.3e} {msg}", flush=True)

    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int32, device="cuda")
    for bs in SCORED:
        d = R.generate_input(bs, "ep8", 777 + bs)
        graphs = []
        for c, mod, st in mods:
            ok, err, _ = check(mod, st, d)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = mod.kernel_fn(d, st)
            graphs.append((c, g, out, ok, err))
        ts = {c: [] for c, *_ in graphs}
        for rep in range(REPS):
            for c, g, *_ in (graphs if rep % 2 == 0 else graphs[::-1]):
                flush.add_(1)
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                ts[c].append(a.elapsed_time(b) * 1000)
        replay_err = {c: relative_l2(out, R.ref_kernel(d, w)) for c, _, out, _, _ in graphs}
        print(f"[own] scored bs={bs} " + "  ".join(
            f"{c}: {sorted(ts[c])[REPS // 2]:.1f}us ok={ok} relL2={err:.3e} replay_relL2={replay_err[c]:.3e}"
            for c, _, _, ok, err in graphs), flush=True)
    print(f"[own] gpu wall {time.time() - t_start:.0f}s", flush=True)


@app.local_entrypoint()
def main(cands: str = "own_moe_v3,own_moe_v4"):
    run.remote(cands)
