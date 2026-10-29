"""GPU-UNVALIDATED rank3 K split with single-publication epilogue.

Four warps own distinct K/4 partitions and the same32 output rows. Fused
FP32 gl.sum(axis=0) replaces global split-K tail. CPU/source prototype only.
Original queued split kernel is untouched. No CUDA work at import. Use --syntax without torch/Triton.
"""
from __future__ import annotations
import ast
import pathlib
import sys
if '--syntax' in sys.argv:
    ast.parse(pathlib.Path(__file__).read_text())
    print('syntax PASS; GPU compile/correctness/performance UNVALIDATED')
    raise SystemExit(0)

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from adaptive_record_kernel import VECTOR_ASM, VECTOR_CONSTRAINTS, dot_record_order
from adaptive_record_torch import pack_weights

@gluon.jit
def split_mm_epilogue(payload, metadata, x, out, N: gl.constexpr, K: gl.constexpr,
             M: gl.constexpr, SCALE: gl.constexpr,
             ASM: gl.constexpr, CONSTRAINTS: gl.constexpr):
    W: gl.constexpr = 4
    R: gl.constexpr = 32
    gl.static_assert(K % (W * 64) == 0)
    gl.static_assert(N % R == 0)
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [4, 1, 1], [1, 16, 8])
    adot: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bdot: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    dll: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 8, 0], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 16, 0]],
        [[0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[1, 0, 0], [2, 0, 0]], [], [W, R, 64])
    # Expand axes in the exact inverse order of nested SliceLayouts.
    batch = gl.arange(0, W, gl.SliceLayout(1, gl.SliceLayout(2, dll)))
    batch = batch[:, None][:, :, None]
    row = gl.arange(0, R, gl.SliceLayout(0, gl.SliceLayout(2, dll)))
    row = row[None, :][:, :, None]
    kk = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(0, dll)))
    kk = kk[None, :][None, :, :]
    lane = (row & 7) * 4 + ((kk >> 1) & 3)
    n0 = gl.program_id(0) * R
    acc = gl.zeros([W, R, M], gl.float32, mma)
    for it in range(K // (W * 64)):
        record_k = batch * (K // (W * 64)) + it
        rid = ((n0 + row) // 16) * (K // 64) + record_k
        meta = gl.load(metadata + rid)
        raw = ((meta & 0x80000000) != 0).to(gl.uint32)
        raw = raw + gl.zeros([W, R, 64], gl.uint32, dll)
        base = (meta & 0x7fffffff).to(gl.int64) * 16
        ptr = payload.to(gl.pointer_type(gl.uint8)) + base + lane * 16
        a = gl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS,
                                     args=[ptr, raw], dtype=gl.bfloat16,
                                     is_pure=False, pack=32)
        a = gl.convert_layout(a, adot, assert_trivial=True)
        bb = gl.arange(0, W, gl.SliceLayout(1, gl.SliceLayout(2, bdot)))
        bb = bb[:, None][:, :, None]
        bk = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, bdot)))
        bk = bk[None, :][:, :, None]
        bt = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(0, bdot)))
        bt = bt[None, :][None, :, :]
        kpos = bb * (K // W) + it * 64 + bk
        b = gl.load(x + bt * K + kpos)
        acc = mma_v2(a, b, acc)
    # One shared publication instead of generic interwarp-reduction chunks.
    scratch = gl.allocate_shared_memory(
        gl.float32, [W, R, M], gl.SwizzledSharedLayout(1, 1, 1, [1, 2, 0]))
    scratch.store(acc)
    gl.barrier()
    # Batch is register-only here: every outputowner reads all4 Kpartials.
    red: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 32, 1], [1, 1, 4], [1, 2, 0])
    partials = scratch.load(red)
    total = gl.sum(partials, axis=0)
    epi: gl.constexpr = gl.SliceLayout(0, red)
    rows = n0 + gl.arange(0, R, gl.SliceLayout(1, epi))[:, None]
    tokens = gl.arange(0, M, gl.SliceLayout(0, epi))[None, :]
    gl.store(out + tokens * N + rows, (total * SCALE).to(gl.bfloat16))


def make(x, w, *, raw=False):
    m, k = x.shape
    n = w.shape[0]
    if m not in (8, 16, 32) or n % 32 or k % 256:
        raise ValueError('prototype requires M8/16/32,N%32=0,K%256=0')
    p = pack_weights(w, force_raw=raw, order=dot_record_order())
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    def run():
        return split_mm_epilogue[(n // 32,)](p['payload'], p['metadata'], x, out,
                                   n, k, m, p['scale'], VECTOR_ASM,
                                   VECTOR_CONSTRAINTS, num_warps=4)
    return run, out
