"""Build bench/cand_r9_cluster*.py = cp8 checkpoint + an r9 block (cluster split-K DSMEM GEMM for m=128/256).

    .venv/bin/python bench/exp_r9_cl_build.py [--src exp_r9_cl_k1] [--out bench/cand_r9_cluster.py]
"""
import argparse
import ast
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
BASE = PROJ / "checkpoints/cp8_zip_robust_check1.399_f29380ec.py"


def block(src_path: Path) -> str:
    text = src_path.read_text()
    tree = ast.parse(text)
    first = tree.body[0]
    start = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else 0
    return "\n".join(text.splitlines()[start:]).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="exp_r9_cl_k1")
    ap.add_argument("--out", default="bench/cand_r9_cluster.py")
    args = ap.parse_args()
    base = BASE.read_text()
    r9 = block(PROJ / "bench" / f"{args.src}.py")
    anchor = "\ndef prepare(weights: Weights) -> dict:\n"
    assert base.count(anchor) == 1
    out = base.replace(anchor, "\n# ===== r9: m=128/256 cluster split-K DSMEM-reduced wgmma GEMM =====\n" + r9 + "\n" + anchor)
    old = "    _zip_prepare(state)\n    torch.cuda.synchronize()\n    return state\n"
    assert out.count(old) == 1
    out = out.replace(old, "    _zip_prepare(state)\n    _r9_prepare(state)\n    torch.cuda.synchronize()\n    return state\n")
    old = '    if cfg["layer"] in state.get("zip", {}):\n'
    assert out.count(old) == 1
    out = out.replace(old, '    out = _r9_try(x, cfg["layer"], cfg["n"], state)\n'
                           '    if out is not None:\n'
                           '        return out\n' + old)
    ast.parse(out)
    for bad in ("subprocess", "os.system", "accessPolicyWindow", "cudaGraphKernelNodeAttribute"):
        assert bad not in out, bad
    (PROJ / args.out).write_text(out)
    print("wrote", args.out, len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
