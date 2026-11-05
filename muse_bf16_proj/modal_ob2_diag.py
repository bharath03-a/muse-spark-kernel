"""OURS: ob2 diag (occupancy + raw timing per config)."""
import os, time
from pathlib import Path
import modal
PROJECT_DIR = Path(__file__).parent
REMOTE_DIR = "/root/proj"
GPU = "H200"
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    .run_commands(
        "echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
        "echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
        "python -c 'import grpclib, google.protobuf, torch, sglang, triton; print(grpclib.__file__,"
        " google.protobuf.__version__, torch.__version__, triton.__version__)'",
    )
    .entrypoint([])
    .add_local_dir(
        PROJECT_DIR,
        remote_path=REMOTE_DIR,
        ignore=[".venv", "__pycache__", "profiles", "*.pyc", ".git", ".ruff_cache", "uv.lock"],
    )
)
EXT_CACHE = modal.Volume.from_name("muse-ext-cache", create_if_missing=True)
app = modal.App("muse-ob2-diag", image=image)

@app.function(gpu="H200", timeout=900, volumes={"/cache": EXT_CACHE})
def diag():
    import sys, importlib.util, torch
    t0 = time.time()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/cache/torch_ext")
    sys.path[:0] = [REMOTE_DIR]
    spec = importlib.util.spec_from_file_location("cand", f"{REMOTE_DIR}/candidate_ownbf_v2.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ext = mod._build_ob2()
    for n, cfgs in ((4096, mod.OB2_PLAN[("pooler", 128)]), (2304, mod.OB2_PLAN[("qkv_gate", 128)])):
        w = torch.randn(n, 8192, device="cuda").to(torch.bfloat16)
        x = torch.randn(128, 8192, device="cuda").to(torch.bfloat16)
        for c in cfgs:
            o = mod._ob2_run(ext, x, w, c); torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(20): mod._ob2_run(ext, x, w, c)
            b.record(); b.synchronize()
            print(n, c, "occ_clusters", ext.ob2_occ(*c), "warm_us", round(a.elapsed_time(b) * 50, 2), flush=True)
    EXT_CACHE.commit()
    print("MODAL_SECONDS", round(time.time() - t0), flush=True)

@app.local_entrypoint()
def main():
    diag.remote()
