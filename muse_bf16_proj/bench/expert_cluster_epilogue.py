"""GPU-UNVALIDATED cluster epilogue control; no projection mainloop yet.
Derived mechanism/layout contracts from upstream Triton v3.8.0:
python/test/gluon/test_core.py async_shared_store_kernel, test_async_shared_store.
Compares generic distributed shared store + barrier with register-source async
store + mbarrier; actual shuffled/remote instruction costs must be inspected.
--syntax is CPU only. Run other modes only with authorized GPU budget.
"""
import argparse
import ast
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--syntax', action='store_true')
    args = ap.parse_args()
    if args.syntax:
        ast.parse(Path(__file__).read_text())
        print('Syntax valid; GPU lowering, correctness, and timing UNVALIDATED')
        return
    run_gpu()


def run_gpu():
    global torch, triton, gluon, gl, mbarrier, async_store
    import torch
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier, async_store
    from top50_qkv_mma import graph, paired

    @gluon.jit
    def reduce_partials(P, O, R: gl.constexpr, T: gl.constexpr, ASYNC: gl.constexpr):
        # Each source CTA holds one K-split plane. Destination CTAs own tokens.
        src: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 8, 4], [1, 4, 1], [2, 1, 0], cga_layout=[[1, 0, 0]])
        dst: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 8, 4], [1, 4, 1], [2, 1, 0], cga_layout=[[0, 0, 1]])
        shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[2, 1, 0], cga_layout=[[0, 0, 1]])
        sf = gl.arange(0, 2)
        rr = gl.arange(0, R)
        tt = gl.arange(0, T)
        offs = sf[:, None, None] * (R*T) + rr[None, :, None] * T + tt[None, None, :]
        offs = gl.set_auto_layout(offs, src)
        values = gl.load(P + gl.program_id(0)*(2*R*T) + offs)
        smem = gl.allocate_shared_memory(gl.float32, [2, R, T], shared)
        if ASYNC:
            bar = mbarrier.allocate_mbarrier()
            mbarrier.init(bar, count=1)
            mbarrier.fence_init_release_cluster()
            gl.barrier(cluster=True)
            mbarrier.expect(bar, smem.nbytes_per_cta)
            async_store(smem, values, bar)
            mbarrier.wait(bar, phase=0, deps=[smem])
            local = smem.load(dst)
            # Keep each peer's destination and barrier alive until all peers finish.
            gl.barrier(cluster=True)
            mbarrier.invalidate(bar)
        else:
            smem.store(values)
            gl.barrier(cluster=True)
            local = smem.load(dst)
            gl.barrier(cluster=True)
        total = gl.sum(local, axis=0)
        outoffs = rr[:, None]*T + tt[None, :]
        outoffs = gl.set_auto_layout(outoffs, gl.SliceLayout(0, dst))
        gl.store(O + gl.program_id(0)*(R*T) + outoffs, total)

    torch.manual_seed(20261026)
    r,t,tiles = 64,32,64
    partials = torch.randn(tiles,2,r,t,device='cuda',dtype=torch.float32)
    outs = {mode:torch.empty(tiles,r,t,device='cuda') for mode in [False,True]}
    functions = {}
    for mode in [False,True]:
        def launch(mode=mode):
            return reduce_partials[(tiles,)](partials,outs[mode],r,t,mode,num_warps=4,num_ctas=2)
        kernel = launch()
        ptx = kernel.asm['ptx']
        print('MODE',mode,'regs',kernel.n_regs,'shared',kernel.metadata.shared,
              'st.async',ptx.count('st.async'),'shuffle',ptx.count('shfl'),flush=True)
        if mode:
            assert 'st.async' in ptx, 'Requested register-source operation absent'
        functions[mode] = launch
    graphs = {mode:graph(fn)[0] for mode,fn in functions.items()}
    for changed in range(3):
        partials.normal_()
        reference = partials.sum(1)
        for mode,g in graphs.items():
            g.replay()
            torch.testing.assert_close(outs[mode],reference,rtol=0,atol=0)
    flush = torch.empty(256*2**20//4,device='cuda')
    print('RAW_COLD_US',paired(graphs,flush,30),flush=True)


if __name__ == '__main__':
    main()
