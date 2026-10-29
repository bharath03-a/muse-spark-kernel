"""Print the GPU and library versions of the judge image."""

import importlib

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
props = torch.cuda.get_device_properties(0)
print("gpu", props.name, "sms", props.multi_processor_count, "mem GB", props.total_memory / 2**30,
      "L2 MB", props.L2_cache_size / 2**20)
for mod in ("triton", "sglang", "flashinfer", "cutlass", "deep_gemm", "tilelang"):
    try:
        m = importlib.import_module(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as exc:
        print(mod, "unavailable:", type(exc).__name__, exc)

for path in ("sglang.kernels.ops.gemm.router_gemv", "sglang.kernels.ops.gemm.hopper_bf16_gemv",
             "sglang.srt.layers.quantization.unquant"):
    try:
        m = importlib.import_module(path)
        print("ok", path, m.__file__)
    except Exception as exc:
        print("missing", path, type(exc).__name__, exc)

try:
    import flashinfer.gemm as fg
    print("flashinfer tinygemm_bf16:", hasattr(fg, "tinygemm_bf16"))
except Exception as exc:
    print("flashinfer.gemm unavailable:", exc)

import sys  # noqa: E402

for mod in sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[1] == "--src" else ():
    try:
        m = importlib.import_module(mod)
        print(f"===== {mod} {m.__file__}")
        print(open(m.__file__).read())
    except Exception as exc:
        print("missing", mod, type(exc).__name__, exc)
