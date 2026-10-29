"""CPU-only SM90 compile audit for ac2 variants. No GPU."""
import json, subprocess, tempfile, time, traceback
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.experimental.gluon._runtime import GluonASTSource
from adaptive_record_async_kernel import DECODE_ASM, DECODE_CONSTRAINTS
from ac2_kernel import _ac2_kernel

VARIANTS = [
    ('pool32_d6', 4096, 8192, 32, 1, 1, 6),
    ('qkv32_d4', 2304, 8192, 32, 1, 1, 4),
    ('pool16_d6', 4096, 8192, 16, 1, 1, 6),
]
fn = _ac2_kernel
names = list(fn.arg_names)
ptrs = {'payload': '*u32', 'metadata': '*u32', 'x': '*bf16', 'out': '*bf16', 'ws': '*fp32', 'cnt': '*i32'}
for label, N, K, M, RW, SPLIT, DEPTH in VARIANTS:
    consts = dict(N=N, K=K, M=M, SCALE=2.0 ** -4, DEPTH=DEPTH, ASM=DECODE_ASM,
                  CONSTRAINTS=DECODE_CONSTRAINTS, RW=RW, SPLIT=SPLIT, XS=K, META_CO=2)
    sig = {nm: ptrs.get(nm, 'constexpr') for nm in names}
    attrs = {(names.index(nm),): [['tt.divisibility', 16]] for nm in ptrs}
    src = GluonASTSource(fn=fn, signature=sig, constexprs={(names.index(nm),): v for nm, v in consts.items()}, attrs=attrs)
    t0 = time.monotonic()
    try:
        k = triton.compile(src, target=GPUTarget('cuda', 90, 32), options={'num_warps': 4 * RW})
        ptx = k.asm['ptx']
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'k.ptx'; p.write_text(ptx)
            r = subprocess.run([get_ptxas(90).path, '-arch=sm_90a', '-v', str(p), '-o', str(Path(td) / 'k.cubin')],
                               capture_output=True, text=True)
        info = r.stderr.strip().splitlines()[-2:] if r.stderr else []
        print(json.dumps({'variant': label, 'ok': True, 's': round(time.monotonic() - t0, 1),
                          'shared': k.metadata.shared, 'mma': ptx.count('mma.sync'), 'bar': ptx.count('bar.sync'),
                          'cp_async': ptx.count('cp.async.'), 'st_shared': ptx.count('st.shared'),
                          'ld_shared': ptx.count('ld.shared'), 'ldmatrix': ptx.count('ldmatrix'),
                          'atom': ptx.count('atom.'), 'ptxas': [l for l in r.stderr.splitlines() if 'spill' in l or 'registers' in l]}), flush=True)
    except Exception:
        print(label, 'COMPILE_FAILED\n' + traceback.format_exc()[-3000:], flush=True)
