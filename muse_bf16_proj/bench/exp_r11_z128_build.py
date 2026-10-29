"""Build bench/cand_r11_z128*.py = cand_r10_big.py + an r11 block (m=128 12-bit zip register-A wgmma GEMM).

    .venv/bin/python bench/exp_r11_z128_build.py [--src exp_r11_z128_k1] [--out bench/cand_r11_z128.py]
"""
import argparse
import ast
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
BASE = PROJ / "bench/cand_r10_big.py"


def block(src_path: Path) -> str:
    text = src_path.read_text()
    tree = ast.parse(text)
    first = tree.body[0]
    start = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else 0
    return "\n".join(text.splitlines()[start:]).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="exp_r11_z128_k1")
    ap.add_argument("--out", default="bench/cand_r11_z128.py")
    args = ap.parse_args()
    base = BASE.read_text()
    blk = block(PROJ / "bench" / f"{args.src}.py")
    anchor = "\ndef prepare(weights: Weights) -> dict:\n"
    assert base.count(anchor) == 1
    out = base.replace(anchor, "\n# ===== r11: m=128 12-bit zip register-A wgmma GEMM =====\n" + blk + "\n" + anchor)
    old = "    _r9_prepare(state)\n    torch.cuda.synchronize()\n    return state\n"
    assert out.count(old) == 1
    out = out.replace(old, "    _r9_prepare(state)\n    _z11_prepare(state)\n    torch.cuda.synchronize()\n    return state\n")
    old = '    out = _r9_try(x, cfg["layer"], cfg["n"], state)\n'
    assert out.count(old) == 1
    out = out.replace(old, '    out = _z11_try(x, cfg["layer"], cfg["n"], state)\n'
                           '    if out is not None:\n'
                           '        return out\n' + old)
    ast.parse(out)
    for bad in ("subprocess", "os.system", "accessPolicyWindow", "cudaGraphKernelNodeAttribute"):
        assert bad not in out, bad
    (PROJ / args.out).write_text(out)
    print("wrote", args.out, len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
