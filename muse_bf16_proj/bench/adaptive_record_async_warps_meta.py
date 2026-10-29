"""GPU-UNVALIDATED compressed-byte cp.async ring plus shared X staging.

CW4/8 R32 with optional CTA-local metadata preload.
CW8 M32 depth3:196800B rings/flags +1024B metadata.
No decoded-weight shared stores, sparse correction, or format control branch.
The safety baseline uses two CTA barriers/iteration for X cross-lane reuse.
CPU/source only; --syntax does not import torch or Triton.
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
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy
from adaptive_record_kernel import dot_record_order
from adaptive_record_torch import pack_weights


def decode_asm():
    # pack16uint32 in/out. Rawinput registerorder already equals MMAorder.
    # Compressed input: a0..3,b0..3,code0..3,unused0..3.
    # Sourceflags packed16 values after16outputs+16data inputs =>operand32.
    lines = ['{', '.reg .pred P;', '.reg .b32 h0,h1,tmp;',
             'setp.ne.u32 P, $32, 0;']
    for j in range(4):
        out = [f'${4*j+i}' for i in range(4)]
        for i in range(4):
            lines.append(f'@P mov.b32 {out[i]}, ${16+4*j+i};')
        lines += [f'@!P lop3.b32 h0, ${24+j}, 0x87878787, 0x38383838, 0xEA;',
                  f'@!P shf.l.wrap.b32 tmp, ${24+j}, ${24+j}, 4;',
                  '@!P lop3.b32 h1, tmp, 0x87878787, 0x38383838, 0xEA;',
                  f'@!P prmt.b32 {out[0]}, ${16+j}, h0, 0x5140;',
                  f'@!P prmt.b32 {out[1]}, ${16+j}, h0, 0x7362;',
                  f'@!P prmt.b32 {out[2]}, ${20+j}, h1, 0x5140;',
                  f'@!P prmt.b32 {out[3]}, ${20+j}, h1, 0x7362;']
    return '\n'.join(lines + ['}'])


DECODE_ASM = decode_asm()
DECODE_CONSTRAINTS = ','.join(['=r'] * 16 + ['r'] * 16 + ['r'] * 16)

@gluon.jit
def issue_warps_meta(payload, metadata, x, wstage, xstage, flags, ms, n0,
          step, K: gl.constexpr, M: gl.constexpr, STAGE_X: gl.constexpr, CW: gl.constexpr, STAGE_META: gl.constexpr):
    cl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [CW, 1], [1, 0])
    record = gl.arange(0, 2 * CW, gl.SliceLayout(1, cl))[:, None]
    word = gl.arange(0, 512, gl.SliceLayout(0, cl))[None, :]
    batch = record % CW
    row = n0 + record // CW * 16
    ki: gl.constexpr = K // (CW * 64)
    pred = step < ki
    rid = (row // 16) * (K // 64) + batch * ki + step
    if STAGE_META:
        # One preloaded [2*CW,1] row, matching cl's record ownership.
        # Tail commits remain in the queue; clamp before reading shared.
        cached = ms.index(gl.minimum(step, ki - 1)).load(cl)
        meta = gl.where(pred, cached, 0)
    else:
        meta = gl.load(metadata + rid, mask=pred, other=0)
    raw = ((meta & 0x80000000) != 0).to(gl.uint32)
    flags.store(raw)
    offset = (meta & 0x7fffffff).to(gl.int64) * 4
    src = payload + offset + word
    async_copy.async_load(wstage, src, mask=pred & ((word < 384) | (raw != 0)))
    if STAGE_X:
        xl: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 4, 8], [CW, 1, 1], [2, 1, 0])
        b = gl.arange(0, CW, gl.SliceLayout(1, gl.SliceLayout(2, xl)))
        b = b[:, None][:, :, None]
        t = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, xl)))
        t = t[None, :][:, :, None]
        kk = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(0, xl)))
        kk = kk[None, :][None, :, :]
        xp = x + t * K + b * (K // CW) + step * 64 + kk
        async_copy.async_load(xstage, xp, mask=pred)
    async_copy.commit_group()


@gluon.jit
def async_mm_warps_meta(payload, metadata, x, out, N: gl.constexpr, K: gl.constexpr,
             M: gl.constexpr, SCALE: gl.constexpr, DEPTH: gl.constexpr,
             ASM: gl.constexpr, CONSTRAINTS: gl.constexpr, STAGE_X: gl.constexpr, CW: gl.constexpr, STAGE_META: gl.constexpr):
    gl.static_assert(CW == 4 or CW == 8)
    gl.static_assert(K % (CW * 64) == 0)
    gl.static_assert(N % 32 == 0)
    gl.static_assert(DEPTH >= 2 and DEPTH <= 4)
    gl.static_assert(K // (CW * 64) >= DEPTH)
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [CW, 1, 1], [1, 16, 8])
    adot: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bdot: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    cl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [CW, 1], [1, 0])
    wb = gl.allocate_shared_memory(gl.uint32, [DEPTH, 2 * CW, 512], gl.SwizzledSharedLayout(4, 1, 1, [1, 0]))
    # Flags are ordinary shared values, published by subsequent CTA barriers.
    fb = gl.allocate_shared_memory(gl.uint32, [DEPTH, 2 * CW, 1], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    if STAGE_X:
        xlayout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([CW, M, 64], gl.bfloat16)
        xb = gl.allocate_shared_memory(gl.bfloat16, [DEPTH, CW, M, 64], xlayout)
    else:
        xb = wb  # Unused descriptor argument in constexpr-disabled Xcopy path.
    n0 = gl.program_id(0) * 32
    KI: gl.constexpr = K // (CW * 64)
    NR: gl.constexpr = 2 * CW
    if STAGE_META:
        gl.static_assert((KI & (KI - 1)) == 0)
        m2 = gl.allocate_shared_memory(gl.uint32, [KI, NR], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        # Global metadata stays in original row16-major order. Adjacent step
        # lanes and then adjacent K partitions form contiguous global loads.
        STEP_LANES: gl.constexpr = min(KI, 32)
        ml: gl.constexpr = gl.BlockedLayout([1, 1], [STEP_LANES, 32 // STEP_LANES], [1, CW], [0, 1])
        mi = gl.arange(0, KI, gl.SliceLayout(1, ml))[:, None]
        mr = gl.arange(0, NR, gl.SliceLayout(0, ml))[None, :]
        mid = (n0 // 16 + mr // CW) * (K // 64) + (mr % CW) * KI + mi
        m2.store(gl.load(metadata + mid))
        # Unswizzled byte-identical view, same proven pattern as ac2 ms/fb.
        ms = m2._reinterpret(gl.uint32, [KI, NR, 1], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        gl.barrier()  # Publish all CTA metadata before any issue consumes it.
    else:
        ms = fb  # Unused descriptor in constexpr-disabled shared-meta path.
    for pre_s in gl.static_range(DEPTH):
        issue_warps_meta(payload, metadata, x, wb.index(pre_s), xb.index(pre_s), fb.index(pre_s), ms,
              n0, pre_s, K, M, STAGE_X, CW, STAGE_META)
    acc = gl.zeros([CW, 32, M], gl.float32, mma)
    for it in range(K // (CW * 64)):
        s = it % DEPTH
        async_copy.wait_group(DEPTH - 1)
        # Publish flags in BOTH controls; staged X may cross lane/warp ownership.
        gl.barrier()
        packed = wb.index(s).load(cl)
        raw = fb.index(s).load(cl)
        raw = raw + gl.zeros([2 * CW, 512], gl.uint32, cl)
        values = gl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS,
                                          args=[packed, raw], dtype=gl.uint32,
                                          is_pure=True, pack=16)
        lo = (values & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        hi = (values >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        physical = gl.join(lo, hi)
        # Physicalbits -> batch,row,K; all permutations are register-local.
        a = physical.reshape([2, CW, 2, 2, 8, 4, 2, 2, 2])
        a = a.permute((1, 0, 7, 4, 2, 3, 6, 5, 8)).reshape([CW, 32, 64])
        a = gl.convert_layout(a, adot, assert_trivial=True)
        if STAGE_X:
            b = xb.index(s).permute((0, 2, 1)).load(bdot)
        else:
            bb = gl.arange(0, CW, gl.SliceLayout(1, gl.SliceLayout(2, bdot)))
            bb = bb[:, None][:, :, None]
            bk = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, bdot)))
            bk = bk[None, :][:, :, None]
            bt = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(0, bdot)))
            bt = bt[None, :][None, :, :]
            b = gl.load(x + bt * K + bb * (K // CW) + it * 64 + bk)
        # Do not overwrite a slot until EVERY consumer has loaded its registers.
        gl.barrier()
        issue_warps_meta(payload, metadata, x, wb.index(s), xb.index(s), fb.index(s), ms,
              n0, it + DEPTH, K, M, STAGE_X, CW, STAGE_META)
        acc = mma_v2(a, b, acc)
    async_copy.wait_group(0)
    # All threads drain before the allocator may reuse ring storage.
    gl.barrier()
    scratch = gl.allocate_shared_memory(gl.float32, [CW, 32, M],
                                       gl.SwizzledSharedLayout(8, 2, 4, [1, 2, 0]))
    scratch.store(acc)
    gl.barrier()
    red: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 32, 1], [1, 1, CW], [1, 2, 0])
    total = gl.sum(scratch.load(red), axis=0)
    epi: gl.constexpr = gl.SliceLayout(0, red)
    nr = n0 + gl.arange(0, 32, gl.SliceLayout(1, epi))[:, None]
    mt = gl.arange(0, M, gl.SliceLayout(0, epi))[None, :]
    gl.store(out + mt * N + nr, (total * SCALE).to(gl.bfloat16))


def make(x, w, *, raw=False, depth=3, stage_x=True, cw=8, stage_meta=True):
    m, k = x.shape
    n = w.shape[0]
    if cw not in (4, 8) or m not in (8, 16, 32) or n % 32 or k % (cw * 64) or k // (cw * 64) < depth:
        raise ValueError('requires CW4/8,M8/16/32,N%32=0,K%(CW64)=0,KI>=depth')
    if stage_meta and ((k // (cw * 64)) & ((k // (cw * 64)) - 1)):
        raise ValueError("shared metadata requires power-of-two K partition steps")
    shared = depth * (cw * 4096 + (cw * m * 128 if stage_x else 0) + cw * 8)
    shared += 2 * (k // 64) * 4 if stage_meta else 0
    if shared > 220 * 1024:
        raise ValueError('conservative shared-memory limit exceeded')
    p = pack_weights(w, force_raw=raw, order=dot_record_order())
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    def run():
        return async_mm_warps_meta[(n // 32,)](p['payload'], p['metadata'], x, out,
                                   n, k, m, p['scale'], depth, DECODE_ASM,
                                   DECODE_CONSTRAINTS, stage_x, cw, stage_meta, num_warps=cw)
    return run, out
