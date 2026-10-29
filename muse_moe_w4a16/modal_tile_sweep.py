"""Dev runner: executes bench/ scripts on a Modal H200 inside the judge's SGLang image.

    modal run modal_app.py::env
    modal run modal_app.py::correct
    modal run modal_app.py::bench
    modal run modal_app.py::micro --name <script>
"""

import os
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).parent
REMOTE_DIR = "/root/proj"
# Judge image (lmsysorg/sglang:dev-muse-spark-open); pin by digest once confirmed.
JUDGE_IMAGE = "lmsysorg/sglang:dev-muse-spark-open"
# H200 needs a Modal payment method; MUSE_GPU="H100!" is a stand-in (timings relative only).
GPU = os.environ.get("MUSE_GPU", "H200") or None  # MUSE_GPU="" for CPU-only ::inspect

# PATH python is the /opt/sglang venv, but Modal installs its runtime deps into system /usr; a .pth
# appends the system dirs after the venv's site-packages so sglang's own versions still win.
image = (
    modal.Image.from_registry(JUDGE_IMAGE)
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

app = modal.App("muse-moe-direct-probe", image=image)
# load_inline build cache shared by CPU compile containers and GPU runs (same image, same paths).
EXT_CACHE = modal.Volume.from_name("muse-ext-cache", create_if_missing=True)
VOLS = {"/cache": EXT_CACHE}


def _run(script: str, *args: str) -> None:
    import runpy
    import sys

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/cache/torch_ext")
    os.environ.setdefault("TRITON_CACHE_DIR", "/cache/triton")
    sys.path[:0] = [REMOTE_DIR, f"{REMOTE_DIR}/bench"]
    sys.argv = [script, *args]
    runpy.run_path(f"{REMOTE_DIR}/{script}", run_name="__main__")


@app.function(gpu=GPU, timeout=300, volumes=VOLS)
def probe():
    import sys, importlib, torch
    sys.path.insert(0, REMOTE_DIR)
    import reference as ref
    import candidate_direct_gemv as cand
    import candidate_marlin_tiles as baseline
    from nvfp4 import random_weights
    gen = torch.Generator(device="cuda").manual_seed(5252)
    w = {}
    for name, n in [("w13",8192),("w2",4096)]:
        packed, sf, gs = random_weights(n,4096,generator=gen,device="cuda",leading=(32,))
        w[name],w[name+"_sf"],w[name+"_gs"] = packed,sf,gs.expand(32).contiguous()
    w["post_norm_w"] = torch.randn(4096,device="cuda",dtype=torch.bfloat16)*0.1
    state = baseline.prepare(w)
    for bs in [1,32,256]:
        d=ref.generate_input(bs,"ep8",7)
        reference=ref.ref_kernel(d,w)
        for tile_k,tile_n in [(-1,-1),(128,128),(64,128),(64,256)]:
            baseline.TILE_K,baseline.TILE_N=tile_k,tile_n
            try:
                out=baseline.kernel_fn(d,state)
                print("CHECK",bs,tile_k,tile_n,ref.compare(out,reference,ref.REL_L2_TOL),flush=True)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):baseline.kernel_fn(d,state)
                flush=torch.empty(64*1024*1024,device="cuda",dtype=torch.uint8)
                times=[]
                for _ in range(20):
                    flush.zero_()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000)
                print("TILES_GRAPH_US",bs,tile_k,tile_n,sorted(times)[10],flush=True)
            except Exception as exc:
                print("FAIL",bs,tile_k,tile_n,type(exc).__name__,str(exc)[:1000],flush=True)
