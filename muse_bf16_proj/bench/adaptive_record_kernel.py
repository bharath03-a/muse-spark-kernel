"""GPU-UNVALIDATED adaptive 16x64 exact-record MMAv2 prototype.

No timed format branch, sparse fix, decoded-W shared memory, or ninth warp.
Packed records use a custom offline MMA-fragment order. Each thread loads
three compressed128-bit vectors or four raw128-bit vectors; decoding costs
seven integer operations per eight BF16 values. PTX must still be inspected.
--syntax runs without torch/Triton; no GPU execution was used to author this.
"""
from __future__ import annotations
import argparse
import ast
import pathlib
import sys
import statistics

if '--syntax' in sys.argv:
    ast.parse(pathlib.Path(__file__).read_text())
    print('syntax PASS; GPU compile/correctness/performance UNVALIDATED')
    raise SystemExit(0)

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from adaptive_record_codec import pack_records, unpack_records
from adaptive_record_torch import pack_weights

def dot_record_order():
    """[p2,lane32,inner16] -> logical16x64 indices, standard DotA layout."""
    return [((lane >> 2) + ((inner >> 1) & 1) * 8) * 64
            + (inner & 1) + (lane & 3) * 2 + p * 8
            + ((inner >> 2) & 1) * 16 + ((inner >> 3) & 1) * 32
            for p in range(2) for lane in range(32) for inner in range(16)]


def vector_asm():
    # pack32 BF16 ->16 output registers. Inputs pack32 pointer +32flag values.
    # Within a physical thread all32 input values are IDENTICAL; use first.
    # Pointer operand$16, flag$48. Remaining repeated operands are compiler CSE.
    lines = ["{", ".reg .pred P;", ".reg .b32 a<4>, b<4>, c<4>, h0, h1, tmp;",
             "setp.ne.u32 P, $48, 0;",
             "@!P ld.global.v4.u32 {a0,a1,a2,a3}, [$16];",
             "@!P ld.global.v4.u32 {b0,b1,b2,b3}, [$16+512];",
             "@!P ld.global.v4.u32 {c0,c1,c2,c3}, [$16+1024];"]
    for j in range(4):
        o = [f"${4*j+i}" for i in range(4)]
        lines.append(f"@P ld.global.v4.u32 {{{','.join(o)}}}, [$16+{512*j}];")
        lines += [f"@!P lop3.b32 h0, c{j}, 0x87878787, 0x38383838, 0xEA;",
                  f"@!P shf.l.wrap.b32 tmp, c{j}, c{j}, 4;",
                  "@!P lop3.b32 h1, tmp, 0x87878787, 0x38383838, 0xEA;",
                  f"@!P prmt.b32 {o[0]}, a{j}, h0, 0x5140;",
                  f"@!P prmt.b32 {o[1]}, a{j}, h0, 0x7362;",
                  f"@!P prmt.b32 {o[2]}, b{j}, h1, 0x5140;",
                  f"@!P prmt.b32 {o[3]}, b{j}, h1, 0x7362;"]
    return "\n".join(lines + ["}"])


VECTOR_ASM = vector_asm()
VECTOR_CONSTRAINTS = ','.join(['=r'] * 16 + ['l'] * 32 + ['r'] * 32)


@gluon.jit
def adaptive_mm(payload, metadata, x, out, N: gl.constexpr, K: gl.constexpr,
                M: gl.constexpr, SCALE: gl.constexpr,
                ASM: gl.constexpr, CONSTRAINTS: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [4, 1], [16, 8])
    adot: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bdot: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    dll: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32]],
        [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        [[16, 0], [32, 0]], [], [64, 64])
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, dll))[:, None]
    ks = gl.arange(0, 64, layout=gl.SliceLayout(0, dll))[None, :]
    lane = (rows & 7) * 4 + ((ks >> 1) & 3)
    n0 = gl.program_id(0) * 64
    acc = gl.zeros([64, M], gl.float32, mma)
    for it in range(K // 64):
        rid = ((n0 + rows) // 16) * (K // 64) + it
        meta = gl.load(metadata + rid)
        raw = ((meta & 0x80000000) != 0).to(gl.uint32)
        raw = raw + gl.zeros([64, 64], gl.uint32, dll)
        base = (meta & 0x7fffffff).to(gl.int64) * 16
        ptr = payload.to(gl.pointer_type(gl.uint8)) + base + lane * 16
        # pack32 produces exactly one thread's complete A register fragment.
        # Source addresses are16B aligned; each warp's32vectorloads coalesce.
        a = gl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS,
                                     args=[ptr, raw], dtype=gl.bfloat16,
                                     is_pure=False, pack=32)
        a = gl.convert_layout(a, adot, assert_trivial=True)
        kk = it * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, bdot))[:, None]
        tt = gl.arange(0, M, layout=gl.SliceLayout(0, bdot))[None, :]
        b = gl.load(x + tt * K + kk)
        acc = mma_v2(a, b, acc)
    ns = n0 + gl.arange(0, 64, layout=gl.SliceLayout(1, mma))[:, None]
    ts = gl.arange(0, M, layout=gl.SliceLayout(0, mma))[None, :]
    gl.store(out + ts * N + ns, (acc * SCALE).to(gl.bfloat16))


def encode(w, *, raw=False):
    packed = pack_weights(w, force_raw=raw, order=dot_record_order())
    return packed['payload'], packed['metadata'], packed['scale']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int, choices=[8, 16, 32], default=16)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--reps', type=int, default=30)
    args = parser.parse_args()
    assert args.n % 64 == args.k % 64 == 0
    torch.manual_seed(20261027)
    w = torch.randn(args.n, args.k, device='cuda').to(torch.bfloat16)
    x = torch.randn(args.m, args.k, device='cuda').to(torch.bfloat16)
    packed = {key: encode(w, raw=raw) for key, raw in [('adaptive', False), ('allraw', True)]}
    outs = {key: torch.empty(args.m, args.n, dtype=torch.bfloat16, device='cuda') for key in packed}
    def run(key):
        payload, meta, scale = packed[key]
        ker = adaptive_mm[(args.n // 64,)](payload, meta, x, outs[key], args.n, args.k, args.m, scale, VECTOR_ASM, VECTOR_CONSTRAINTS, num_warps=4)
        return ker
    for key in packed:
        ker = run(key)
        print('COMPILE', key, ker.n_regs, ker.n_spills, ker.metadata.shared, flush=True)
    graphs = {}
    for key in packed:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(key)
        graphs[key] = g
    previous = {}
    for dist in ['normal', 'outliers', 'onehot', 'changed']:
        x.normal_()
        if dist == 'outliers':
            x[:, ::97] *= 32
        if dist == 'onehot':
            x.zero_(); x[torch.arange(args.m), torch.arange(args.m) % args.k] = 2
        ref = x.double() @ w.double().T
        denom = ref.norm(dim=1).clamp_min(ref.norm(dim=1).mean() / 10)
        for key in packed:
            for _ in range(3):
                graphs[key].replay()
                value = outs[key].double()
                error = value - ref
                assert torch.isfinite(value).all()
                assert error.norm() / ref.norm() <= (0.001 if dist == 'onehot' else 0.01)
                assert (error.norm(dim=1) / denom).max() <= (0.001 if dist == 'onehot' else 0.05)
            if dist == 'changed':
                assert not torch.equal(previous[key], outs[key])
            previous[key] = outs[key].clone()
            print('CHECK', key, dist, flush=True)
    flush = torch.randn(256 * 2**20 // 4, device='cuda', dtype=torch.float32)
    times = {key: [] for key in packed}
    for rep in range(args.reps):
        for key in (list(packed) if rep % 2 == 0 else list(packed)[::-1]):
            flush.sum()  # Read eviction; do not dirty256MiB immediately before replay.
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graphs[key].replay(); b.record(); b.synchronize()
            times[key].append(a.elapsed_time(b) * 1000)
    print('RAW_COLD_GRAPH_US', {key: statistics.median(t) for key, t in times.items()})

if __name__ == '__main__':
    main()
