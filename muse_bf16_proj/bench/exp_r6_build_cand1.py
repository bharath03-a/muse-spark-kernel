"""Build bench/cand_zip1.py = submission.py + 12-bit lossless pooler m=1 kernel (v1 register-direct)."""
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from unittest import mock  # noqa: E402
sys.modules['torch'] = mock.MagicMock()
import exp_r6_zip_v1 as V1  # noqa: E402
import exp_r6_zip_v2 as V2  # noqa: E402

root = Path(__file__).resolve().parent.parent
s = (root / "submission.py").read_text()

src = V1.ZIP_SRC
for line in ("ZIP_CASE(1, 4, 2, 8, 3)", "ZIP_CASE(1, 4, 4, 8, 2)", "ZIP_CASE(1, 4, 4, 8, 3)",
             "ZIP_CASE(1, 2, 2, 8, 2)", "ZIP_CASE(1, 2, 4, 8, 2)", "ZIP_CASE(1, 2, 1, 8, 2)"):
    assert line in src
    src = src.replace("  " + line + "\n", "")
assert src.count("  ZIP_CASE(") == 1 and "ZIP_CASE(1, 4, 2, 8, 2)" in src.split("void launch_zip")[1]
assert '"""' not in src

a = '"""\n\nCUDA_CPP = ('
assert s.count(a) == 1
s = s.replace(a, src + a)

py = "\n\nZIP_CPP = (" + repr(V1.ZIP_CPP) + ")\n"
py += "REBASE = 112\n\n"
for fn in (V2.pick_eb, V2.pack, V2.unpack):
    py += inspect.getsource(fn) + "\n\n"
py += '''def _zip_pack_pooler(w: torch.Tensor, ext) -> dict | None:
    """Pack the pooler (12-bit lossless) and verify bit-exactness and the kernel; None -> uncompressed path."""
    try:
        n, k = w.shape
        p = pack(w, 4)
        dec = unpack(p, n, k) * p["scale"]
        bad = (dec != w.float()).nonzero()
        if bad.shape[0] != p["n_esc"]:
            raise AssertionError("zip round trip: unexpected mismatch count")
        if p["n_esc"]:
            want = w[bad[:, 0], bad[:, 1]].float()
            rp = p["rowptr"].long()
            cnt = rp[1:] - rp[:-1]
            rows = torch.repeat_interleave(torch.arange(n, device=w.device), cnt)
            cols = p["esck"][: rows.numel()].long()
            order = torch.argsort(bad[:, 0] * k + bad[:, 1])
            if not (torch.equal(bad[order][:, 0], rows) and torch.equal(bad[order][:, 1], cols)):
                raise AssertionError("zip escape list mismatch")
            if not torch.equal(dec[rows, cols] + p["escd"][: rows.numel()], w[rows, cols].float()):
                raise AssertionError("zip escape fix is not exact")
        del dec, bad
        x = torch.randn(1, k, device=w.device).to(torch.bfloat16)
        out = torch.empty(1, n, device=w.device, dtype=torch.bfloat16)
        ext.launch_zip(x, p["buf"], out, p["rowptr"], p["esck"], p["escd"], p["scale"], 0, 1, 4, 2, 8, 2)
        ref = x.double() @ w.double().t()
        if float((out.double() - ref).norm() / ref.norm()) > 5e-3:
            raise AssertionError("zip kernel self-check failed")
        return p
    except Exception as exc:
        print(f"zip pooler disabled: {type(exc).__name__}: {exc}")
        return None


def _zip_pooler_m1(x: torch.Tensor, state: dict) -> torch.Tensor:
    p = state["zip_pooler"]
    out = torch.empty(1, p["buf"].shape[0] * 64, device=x.device, dtype=torch.bfloat16)
    state["ext"].launch_zip(x, p["buf"], out, p["rowptr"], p["esck"], p["escd"], p["scale"], 0, 1, 4, 2, 8, 2)
    return out
'''
b = "\nCUDA_CPP = ("
assert s.count(b) == 1
s = s.replace(b, py + b)

c = '" int64_t per);")'
assert s.count(c) == 1
s = s.replace(c, '" int64_t per);\\n" + ZIP_CPP)')
d = 'functions=["launch_wgmma", "launch_gemv"]'
assert s.count(d) == 1
s = s.replace(d, 'functions=["launch_wgmma", "launch_gemv", "launch_zip"]')

e = '    state["unpooler_slice"] = state["unpooler"][:UNPOOL_SLICE]\n'
assert s.count(e) == 1
s = s.replace(e, e + '    state["zip_pooler"] = _zip_pack_pooler(state["pooler"], state["ext"]) if state.get("ext") is not None else None\n')

f = '    if cfg["layer"] == "router":\n        return _router(x, state)\n'
assert s.count(f) == 1
s = s.replace(f, f + '    if cfg["layer"] == "pooler" and x.shape[0] == 1 and state.get("zip_pooler") is not None:\n        return _zip_pooler_m1(x, state)\n')
s = s.replace('"""Dense projections of Muse Spark Open', '"""[cand_zip1: pooler m=1 12-bit lossless] Dense projections of Muse Spark Open', 1)
(root / "bench" / "cand_zip1.py").write_text(s)
print("wrote cand_zip1.py", len(s))
