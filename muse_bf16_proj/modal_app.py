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

app = modal.App("muse-bf16-proj", image=image)
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


@app.function(gpu=GPU, timeout=1200, volumes=VOLS)
def env() -> None:
    _run("bench/env.py")


@app.function(gpu=GPU, timeout=1200, volumes=VOLS)
def correct(args: str = "") -> None:
    _run("bench/correct.py", *args.split())


@app.function(gpu=GPU, timeout=1200, volumes=VOLS)
def bench(args: str = "") -> None:
    _run("bench/harness.py", *args.split())


@app.function(gpu=GPU, timeout=1200, volumes=VOLS)
def micro(name: str, args: str = "") -> None:
    _run(f"bench/{name}.py", *args.split())


@app.function(gpu=GPU, timeout=1200, volumes=VOLS)
def session(cmds: str) -> None:
    """Several bench scripts in one container (one cold start): "env;correct --quick;harness"."""
    import shlex
    import time

    for cmd in filter(None, (c.strip() for c in cmds.split(";"))):
        name, *args = shlex.split(cmd)
        t0 = time.time()
        print(f"\n######## {cmd}", flush=True)
        try:
            _run(f"bench/{name}.py", *args)
        except SystemExit:
            pass
        except Exception:
            import traceback

            traceback.print_exc()
        print(f"######## {cmd} took {time.time() - t0:.0f}s", flush=True)
    EXT_CACHE.commit()


@app.function(timeout=1200, volumes=VOLS, cpu=8)
def compile_only(cmd: str = "cluster_dev --compile") -> None:
    """CPU-only nvcc/triton build of one or more bench scripts (;-separated) into the shared cache."""
    import shlex
    import time

    t0 = time.time()
    try:
        for subcmd in filter(None, (c.strip() for c in cmd.split(";"))):
            name, *args = shlex.split(subcmd)
            ts = time.time()
            print(f"\n######## compile_only: {subcmd}", flush=True)
            try:
                _run(f"bench/{name}.py", *args)
            except SystemExit:
                pass
            except Exception:
                import traceback

                traceback.print_exc()
            print(f"######## {subcmd} took {time.time() - ts:.0f}s", flush=True)
    finally:
        EXT_CACHE.commit()
        print(f"compile_only total {time.time() - t0:.0f}s", flush=True)


@app.function(timeout=900)
def inspect(path: str = "") -> None:
    """CPU-only: versions and source of image modules (no GPU needed)."""
    import importlib
    import importlib.metadata as md

    for pkg in ("torch", "triton", "sglang", "flashinfer-python", "sgl-kernel", "nvidia-cublas-cu12",
                "nvidia-cublas"):
        try:
            print(pkg, md.version(pkg))
        except md.PackageNotFoundError:
            print(pkg, "not installed")
    import torch

    print("torch cuda", torch.version.cuda)
    for mod in path.split(",") if path else ():
        try:
            m = importlib.import_module(mod)
            print(f"===== {mod} {m.__file__}")
            print(open(m.__file__).read())
        except Exception as exc:
            print("missing", mod, type(exc).__name__, exc)
