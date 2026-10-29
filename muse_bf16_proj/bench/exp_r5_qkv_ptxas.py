from common import load_module
m = load_module("bench/cand_qkv.py", "cand")
m.CUDA_FLAGS = m.CUDA_FLAGS + ["-Xptxas", "-v"]
try:
    m._build_cuda()
except ImportError as e:
    print("import err (expected on CPU)", e)
