"""Integrate measured W13 epilogue fusion while retaining native/fallback banks.

No pins are changed. Mixed mode2 remains native until separately validated.
"""
from pathlib import Path
import ast
ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'research_prefill_w13_fused.py').read_text()

def change(old, new):
    global source
    assert source.count(old) == 1, (old[:90], source.count(old))
    source = source.replace(old, new)

old = '''                st["w4"] = {"ext": ext, "w13": w13, "g13": g13, "w2": w2, "g2": g2, "ctr": torch.zeros(24, dtype=torch.int32, device=dev)}'''
change(old, old + '''
                # Keep native gate-first W13 for diagnostics, mixed mode and fallback.
                # Pairing is a lossless row permutation of both codes and scales.
                try:
                    paired, paired_g = _pack_paired_w13(weights)
                    st["w4"]["w13_fused"] = paired
                    st["w4"]["g13_fused"] = paired_g
                except Exception as exc:
                    print(f"paired W13 disabled, native path retained: {exc}", flush=True)''')
old = '''    Hh = _w4gemm(A, offs, w["w13"], w["g13"], w["ctr"][0:8], wide=wide)
    del A
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act2(Hh, offs, I)
    del Hh'''
new = '''    if wide in FUSED_W13_MODES and "w13_fused" in w:
        # W13 always uses its full K reduction; W2 split policy is unchanged.
        I = _w4gemm(A, offs, w["w13_fused"], w["g13_fused"],
                    w["ctr"][0:8], ksplit=1, wide=wide, fuse=True)
        del A
    else:
        Hh = _w4gemm(A, offs, w["w13"], w["g13"], w["ctr"][0:8], wide=wide)
        del A
        I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
        g.act2(Hh, offs, I)
        del Hh'''
change(old,new)
# Reuse the independently built strict native fallback, without mutating it.
strict_source = (ROOT / 'candidate_prefill_strict.py').read_text()
strict_marker = '# Native fallback embedded from the local challenge baseline, no sibling imports.'
assert strict_source.count(strict_marker) == 1
native_suffix = strict_source[strict_source.index(strict_marker):]
assert source.count('for k2 in (1, 2)') == 2
source = source.replace('for k2 in (1, 2)', 'for k2 in (1,)')
guard_marker = '    """gathered rms-normed rows -> w4g W13 (H, gate|up) -> SwiGLU -> w4g W2 (ks2 K halves) -> post-norm weighted combine (+ add)"""'
assert source.count(guard_marker) == 1
source = source.replace(guard_marker, guard_marker + '\n    if ks2 != 1:\n        raise ValueError("strict W2 requires complete FP32 reduction before BF16 output")')
lazy = '\n'.join([
    '    if T <= 256 or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2:',
    '        if st.get("lean"):',
    '            if "dl" not in st:',
    '                st["dl"] = _prepare_dl(st["weights"])',
    '            return _kernel_fn_dl(data, st["dl"])',
    '        return _kernel_fn_dl(data, st)',
]) + '\n'
assert source.count(lazy) == 1
source = source.replace(lazy, '    # Untimed native fallback is dispatched by the final strict wrapper.\n')

source += '''

# Correctness and paired microbenchmarks covered these two modes only.
# Mode2 uses two different big-tile schedules plus a remainder and remains native.
FUSED_W13_MODES = (0, 1)

def _pack_paired_w13(weights):
    raw, scales, gs = weights["w13"], weights["w13_sf"], weights["w13_gs"]
    half = raw.shape[1] // 2
    if raw.shape[1] != 2 * half or half % 8:
        raise ValueError("paired W13 requires complete groups of eight gate/up rows")
    gate = torch.arange(half, device=raw.device).view(-1, 8)
    perm = torch.stack((gate, half + gate), dim=1).reshape(-1)
    return _w4pack(raw.index_select(1, perm), scales.index_select(1, perm), gs)
'''
source += '\n' + native_suffix
out=ROOT/'candidate_prefill_w13_integrated.py'
out.write_text(source)
compile(source,str(out),'exec')
tree=ast.parse(source)
# Native diagnostic call sites must remain native; only production dispatch branches.
moe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_moe_w4')
text=ast.unparse(moe)
assert text.count('fuse=True')==1
assert 'g.act2' in text
assert 'w13_fused' in text and "w['w13']" in text
print(out)
