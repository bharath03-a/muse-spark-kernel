"""ac2: adaptive compressed-record ring, RW row-warps share one staged x, cross-CTA split-K.

Same record format/decoder as cp17 _ac_kernel. RW=1,SPLIT=1 reproduces cp17's mapping.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy


@gluon.jit
def _ac2_issue(payload, metadata, x, wstage, xstage, flags, ms, n0, kb0, xk0, cta,
               step, K: gl.constexpr, KC: gl.constexpr, M: gl.constexpr, RW: gl.constexpr,
               XS: gl.constexpr, META_CO: gl.constexpr):
    NW: gl.constexpr = 4 * RW
    NR: gl.constexpr = 8 * RW
    cl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NW, 1], [1, 0])
    r = gl.arange(0, NR, gl.SliceLayout(1, cl))[:, None]
    word = gl.arange(0, 512, gl.SliceLayout(0, cl))[None, :]
    # record r -> warp r % NW = rowhalf + RW*batch; rgih = r // NW
    rowhalf = r % RW
    batch = (r // RW) % 4
    rgih = r // NW
    row = n0 + (rgih * RW + rowhalf) * 16
    ki: gl.constexpr = KC // 256
    pred = step < ki
    if META_CO == 2:
        # CTA's metadata preloaded to shared once; no global load on the issue path.
        mrow = ms.index(gl.minimum(step, ki - 1)).load(cl)
        meta = gl.where(pred, mrow, 0)
    else:
        if META_CO == 1:
            rid = (cta * ki + step) * NR + r
        else:
            rid = (row // 16) * (K // 64) + kb0 + batch * ki + step
        meta = gl.load(metadata + rid, mask=pred, other=0)
    raw = ((meta & 0x80000000) != 0).to(gl.uint32)
    flags.store(raw)
    offset = (meta & 0x7fffffff).to(gl.int64) * 4
    async_copy.async_load(wstage, payload + offset + word, mask=pred & ((word < 384) | (raw != 0)))
    xl: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 4, 8], [4, RW, 1], [2, 1, 0])
    b = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, xl)))[:, None][:, :, None]
    t = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, xl)))[None, :][:, :, None]
    kk = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(0, xl)))[None, :][None, :, :]
    xp = x + t * XS + xk0 + b * (KC // 4) + step * 64 + kk
    async_copy.async_load(xstage, xp, mask=pred)
    async_copy.commit_group()


@gluon.jit
def _ac2_kernel(payload, metadata, x, out, ws, cnt, N: gl.constexpr, K: gl.constexpr,
                M: gl.constexpr, SCALE: gl.constexpr, DEPTH: gl.constexpr,
                ASM: gl.constexpr, CONSTRAINTS: gl.constexpr, RW: gl.constexpr, SPLIT: gl.constexpr,
                XS: gl.constexpr, META_CO: gl.constexpr):
    KC: gl.constexpr = K // SPLIT
    NW: gl.constexpr = 4 * RW
    NR: gl.constexpr = 8 * RW
    ROWS: gl.constexpr = 32 * RW
    gl.static_assert(KC % 256 == 0)
    gl.static_assert(N % ROWS == 0)
    gl.static_assert(DEPTH >= 2 and DEPTH <= 8)
    gl.static_assert(KC // 256 >= DEPTH)
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [4, RW, 1], [1, 16, 8])
    adot: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bdot: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    cl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NW, 1], [1, 0])
    wb = gl.allocate_shared_memory(gl.uint32, [DEPTH, NR, 512], gl.SwizzledSharedLayout(4, 1, 1, [1, 0]))
    fb = gl.allocate_shared_memory(gl.uint32, [DEPTH, NR, 1], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    xlayout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([4, M, 64], gl.bfloat16)
    xb = gl.allocate_shared_memory(gl.bfloat16, [DEPTH, 4, M, 64], xlayout)
    pid_n = gl.program_id(0)
    pid_s = gl.program_id(1)
    n0 = pid_n * ROWS
    kb0 = pid_s * (KC // 64)
    xk0 = pid_s * KC
    cta = pid_n * SPLIT + pid_s
    KI: gl.constexpr = KC // 256
    if META_CO == 2:
        m2 = gl.allocate_shared_memory(gl.uint32, [KI, NR], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        ml: gl.constexpr = gl.BlockedLayout([1, 1], [32 // NR, NR], [NW, 1], [1, 0])
        mi = gl.arange(0, KI, gl.SliceLayout(1, ml))[:, None]
        mr = gl.arange(0, NR, gl.SliceLayout(0, ml))[None, :]
        m2.store(gl.load(metadata + cta * (KI * NR) + mi * NR + mr))
        # Unswizzled, so the [KI, NR, 1] view is byte-identical (same convention as fb).
        ms = m2._reinterpret(gl.uint32, [KI, NR, 1], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        gl.barrier()
    else:
        ms = fb
    for pre_s in gl.static_range(DEPTH):
        _ac2_issue(payload, metadata, x, wb.index(pre_s), xb.index(pre_s), fb.index(pre_s), ms,
                   n0, kb0, xk0, cta, pre_s, K, KC, M, RW, XS, META_CO)
    acc = gl.zeros([4, ROWS, M], gl.float32, mma)
    for it in range(KC // 256):
        s = it % DEPTH
        async_copy.wait_group(DEPTH - 1)
        gl.barrier()
        packed = wb.index(s).load(cl)
        raw = fb.index(s).load(cl)
        raw = raw + gl.zeros([NR, 512], gl.uint32, cl)
        values = gl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS,
                                          args=[packed, raw], dtype=gl.uint32,
                                          is_pure=True, pack=16)
        lo = (values & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        hi = (values >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        physical = gl.join(lo, hi)
        if RW == 1:
            a = physical.reshape([2, 4, 2, 2, 8, 4, 2, 2, 2])
            a = a.permute((1, 0, 7, 4, 2, 3, 6, 5, 8)).reshape([4, 32, 64])
        else:
            a = physical.reshape([2, 4, 2, 2, 2, 8, 4, 2, 2, 2])
            a = a.permute((1, 0, 2, 8, 5, 3, 4, 7, 6, 9)).reshape([4, ROWS, 64])
        a = gl.convert_layout(a, adot, assert_trivial=True)
        b = xb.index(s).permute((0, 2, 1)).load(bdot)
        gl.barrier()
        _ac2_issue(payload, metadata, x, wb.index(s), xb.index(s), fb.index(s), ms,
                   n0, kb0, xk0, cta, it + DEPTH, K, KC, M, RW, XS, META_CO)
        acc = mma_v2(a, b, acc)
    async_copy.wait_group(0)
    gl.barrier()
    scratch = gl.allocate_shared_memory(gl.float32, [4, ROWS, M],
                                       gl.SwizzledSharedLayout(8, 2, 4, [1, 2, 0]))
    scratch.store(acc)
    gl.barrier()
    red: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 32, 1], [1, 1, NW], [1, 2, 0])
    total = gl.sum(scratch.load(red), axis=0)
    epi: gl.constexpr = gl.SliceLayout(0, red)
    rl = gl.arange(0, ROWS, gl.SliceLayout(1, epi))[:, None]
    mt = gl.arange(0, M, gl.SliceLayout(0, epi))[None, :]
    out_ptrs = out + mt * N + (n0 + rl)
    if SPLIT == 1:
        gl.store(out_ptrs, (total * SCALE).to(gl.bfloat16))
    else:
        ws_base = ws + pid_n * (SPLIT * ROWS * M) + rl * M + mt
        gl.store(ws_base + pid_s * (ROWS * M), total)
        gl.barrier()
        done = gl.atomic_add(cnt + pid_n, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                total = total + gl.load(ws_base + (1 - pid_s) * (ROWS * M), cache_modifier=".cg")
            else:
                total = gl.zeros([ROWS, M], gl.float32, epi)
                for sp in gl.static_range(SPLIT):
                    total += gl.load(ws_base + sp * (ROWS * M), cache_modifier=".cg")
            gl.store(out_ptrs, (total * SCALE).to(gl.bfloat16))
            gl.atomic_xchg(cnt + pid_n, 0)
