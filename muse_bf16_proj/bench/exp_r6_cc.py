"""CPU: build cand_zip1's CUDA extension into the shared cache."""
from common import load_module
m = load_module("bench/cand_zip1.py", "cand")
m._build_cuda()
print("built")
