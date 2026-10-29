"""Port upstream v56 (8-op e2m1 dequant, wx UDESC) onto our strict BF16 boundaries, frozen dispatch.

Strict edits are replayed from the verified diff (pinned d2057 original -> scored strict/grid source)
as exact block replacements that must match v56 exactly once. CUDA_X gets the WX52 strict edits.
Dispatch is a frozen copy of v56's own judge-GPU tune table (job 24c4d0f8e8944e4c log).
"""
import ast
import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
UP = Path('/tmp/muse-openkernels-e6b48c8/muse_moe_w4a16/submission_v56.py')
up = UP.read_text()
orig = (root / 'candidate_openkernels_moe.py').read_text()
strict = (root / 'research_moe_grid_constant.py').read_text()


def strings(text):
    out = {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            out[getattr(n.targets[0], 'id', '')] = (n, n.value.value)
    return out


def change(s, a, b, n=1):
    assert s.count(a) == n, (a[:120], s.count(a), n)
    return s.replace(a, b)


def replay(old_src, new_src, target):
    """Apply each old->new hunk (with one context line on each side) to target exactly once."""
    a, b = old_src.splitlines(True), new_src.splitlines(True)
    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        lo, hi = max(i1 - 1, 0), min(i2 + 1, len(a))
        before = ''.join(a[lo:hi])
        after = ''.join(a[lo:i1]) + ''.join(b[j1:j2]) + ''.join(a[i2:hi])
        edits.append((before, after))
    for before, after in edits:
        target = change(target, before, after)
    return target, len(edits)


U, O, S = strings(up), strings(orig), strings(strict)
patched = {}
for name in ('CUDA', 'CUDA_L', 'CUDA_G', 'CUDA_K'):
    patched[name], k = replay(O[name][1], S[name][1], U[name][1])
    print(name, 'hunks', k)

c = U['CUDA_X'][1]
c = change(c, '#include <cuda_bf16.h>', '''#include <cuda_bf16.h>
// Strict SGLang boundaries: BF16 gate/up -> FP32 SiLU/product -> BF16 inter.
__device__ __forceinline__ float strict_swiglu(float gate, float up) {
  gate=__bfloat162float(__float2bfloat16_rn(gate));
  up=__bfloat162float(__float2bfloat16_rn(up));
  return gate/(1.f+__expf(-gate))*up;
}
''')
c = change(c, 'gt / (1.f + __expf(-gt)) * up', 'strict_swiglu(gt, up)')
c = change(c, 'const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;',
           'const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];')
c = change(c, 'const float v0 = acc[i * 4] * n0, v1 = acc[i * 4 + 1] * n0, v2 = acc[i * 4 + 2] * n1, v3 = acc[i * 4 + 3] * n1;',
           '''// Complete W2, apply its own expert's global scale, then round BEFORE norm.
      const float q0=__bfloat162float(__float2bfloat16(acc[i*4]*gs)), q1=__bfloat162float(__float2bfloat16(acc[i*4+1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[i*4+2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[i*4+3]*gs));
      const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;''')
for i in range(4):
    c = change(c, f'__float2bfloat16(v{i})', f'__float2bfloat16(q{i})')
c = change(c, '__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])',
           '__floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1])')
patched['CUDA_X'] = c

text = up
for name, src in patched.items():
    node = U[name][0]
    seg = ast.get_source_segment(up, node.value)
    assert text.count(seg) == 1, name
    text = text.replace(seg, repr(src), 1)

# Unique extension names so no cached non-strict build is reused.
text = text.replace('"moe_w4a16_v5', '"moe_v56strict_v5')
assert '"moe_w4a16_v' not in text
text = change(text, '"nw": 1.0 + weights["post_norm_w"].float()',
              '"nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float()')

text += '''

# ---- Research team: strict BF16 boundaries (see header) and frozen judge-GPU dispatch.
# Source: lee101/openkernels e6b48c833d8c4c1ebbca16b80d784b9e3d5bb86b submission_v56.py, Apache-2.0.
TUNE = False
G_CFG = ()
FS_CFG = ()
K_CFG = ("-DSW=4 -DKS2=1 -DMAXM=8", "-DSW=4 -DKS2=2")
WX_CFG = (("-DNCW=16", "-DWXG2", "-DUDESC"), ("-DNCW=16", "-DN8STEP", "-DWXG2", "-DUDESC"))
_FROZEN = (
    (8, ("gk", 0, 0, "-DSW=4 -DKS2=1 -DMAXM=8")),
    (16, ("gk", 0, 0, "-DSW=4 -DKS2=2")),
    (64, ("ring", 1, 0)),
    (192, ("wx", 0, 0, "-DNCW=16 -DWXG2 -DUDESC")),
    (1 << 30, ("wx", 0, 0, "-DNCW=16 -DN8STEP -DWXG2 -DUDESC")),
)


def kernel_fn(data, state):
    M = data["hidden_states"].shape[0]
    cfg = next(c for hi, c in _FROZEN if M <= hi)
    if data["config"]["local_start"] != 0 and cfg[0] in ("gk", "wx"):
        cfg = _default(M)
    return _run(cfg, data, state)
'''
out = root / 'research_moe_v56_strict.py'
out.write_text(text)
compile(text, str(out), 'exec')
src_dir = root / 'research' / 'kernel_sources'
for name, src in patched.items():
    (src_dir / f'v56strict_{name}.cu').write_text(src)
print(out)
