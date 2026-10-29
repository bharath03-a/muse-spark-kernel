import time
from common import load_module
t = time.time()
m = load_module("bench/cand_qkv.py", "cand")
m._build_cuda()
print(f"built in {time.time() - t:.0f}s")
