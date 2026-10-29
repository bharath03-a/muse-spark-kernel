"""CPU prebuild of a candidate's load_inline extensions into the shared cache: exp_r7_build_cand.py [file]."""
import sys
import time

from common import load_module

t0 = time.time()
mod = load_module(sys.argv[1] if len(sys.argv) > 1 else "bench/cand_zip.py", "cand")
for fn in (mod._build_cuda, mod._build_zip):
    try:
        fn()
    except ImportError as exc:  # CPU container: .so built and cached, only the libcuda import fails
        print(f"{fn.__name__}: built (import skipped: {exc})")
print(f"built in {time.time() - t0:.0f}s")
