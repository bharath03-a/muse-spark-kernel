"""Splice exp_r16_os_src.py into cp10 -> bench/cand_r16_os.py."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
base = (ROOT / "checkpoints/cp10_qz_check1.407.py").read_text()
src = (ROOT / "bench/exp_r16_os_src.py").read_text()
head = 'import hashlib\n\nimport torch\n\nCUDA_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]\n'
assert head in src
src = src.replace(head, "")
anchor = "def prepare(weights: Weights) -> dict:"
assert base.count(anchor) == 1
base = base.replace(anchor, "# ===== r16: one-shot small-matrix kernels =====\n" + src + "\n\n" + anchor)
old = "    _r9_prepare(state)\n    torch.cuda.synchronize()\n    return state\n"
assert base.count(old) == 1
base = base.replace(old, "    _r9_prepare(state)\n    _os_prepare(state)\n    torch.cuda.synchronize()\n    return state\n")
old = '''    if cfg["layer"] == "router":
        return _router(x, state)
'''
assert base.count(old) == 1
base = base.replace(old, '''    if state.get("os"):
        layer = cfg["layer"]
        key = "router" if layer == "router" else _bf16_key(layer, cfg["n"], state[layer].shape[0])
        out = _os_try(x, key, cfg["n"], state)
        if out is not None:
            return out
''' + old)
(ROOT / "bench/cand_r16_os.py").write_text(base)
print("ok", len(base.splitlines()))
