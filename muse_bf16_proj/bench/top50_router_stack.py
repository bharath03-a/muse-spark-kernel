"""Router one cuBLAS call for [hi;lo] rows, FP32 output, versus two calls.
Inspiration: lee101/openkernels submission_v11.py stacked router GEMM.
No prepare; cached cp14 CUDA extension only. Cold graphs, exact precision.
"""
import hashlib
import os
from pathlib import Path
import time
import torch
from common import load_module
from top50_qkv_mma import graph, paired


def main():
    start = time.monotonic()
    C = load_module('checkpoints/cp14_D_slice_check1.426.py', 'cp14_router_stack')
    tag = hashlib.sha1((C.CUDA_SRC + ' '.join(C.CUDA_FLAGS + C.CUDA_LDFLAGS)).encode()).hexdigest()[:10]
    name = f'muse_proj_cuda_cp4_{tag}'
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', '/cache/torch_ext')) / name
    ext = None
    if list(cache.glob('*.so')):
        ext = C._build_cuda()
    else:
        print('SKIP cp14_cuda: no cached extension; refusing GPU compile', flush=True)
    torch.manual_seed(27)
    w = torch.randn(256, 8192, device='cuda') / 8192**0.5
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    ht, lt = hi.T, lo.T
    stacked_t = torch.cat([hi, lo], dim=0).T
    hl = torch.stack([hi, lo]).contiguous()
    state = {'ext': ext}
    flush = torch.empty(256 * 2**20 // 4, device='cuda')
    for m in (128, 256):
        x = torch.randn(m, 8192, device='cuda').to(torch.bfloat16)
        def stacked():
            both = torch.mm(x, stacked_t, out_dtype=torch.float32)
            return both[:, :256] + both[:, 256:]
        def two():
            acc = torch.mm(x, ht, out_dtype=torch.float32)
            acc += torch.mm(x, lt, out_dtype=torch.float32)
            return acc
        funcs = {'stacked512_mm_sum': stacked, 'two256_mm_sum': two}
        pick = C._cuda_pick(state, 'router', m)
        if pick:
            kind, params = pick
            funcs['cp14_cuda'] = lambda: C._cuda_run(state, x, hl, 256, torch.float32, kind, params)
        for dist in ('normal', 'outliers', 'onehot'):
            x.normal_()
            if dist == 'outliers':
                x[:, ::97] *= 30
            if dist == 'onehot':
                x.zero_()
                x[torch.arange(m), torch.arange(m) * 31] = 2
            ref = x.double() @ w.double().T
            rn = ref.norm(dim=1)
            denom = rn.clamp_min(rn.mean() / 10)
            for key, fn in funcs.items():
                out = fn()
                delta = out.double() - ref
                rel = (delta.norm() / ref.norm()).item()
                row = (delta.norm(dim=1) / denom).max().item()
                print(f'CHECK {key} m{m} {dist} rel{rel:.4g} row{row:.4g}', flush=True)
                assert torch.isfinite(out).all() and rel <= 1e-4 and row <= 5e-4
        x.normal_()
        captured = {key: graph(fn) for key, fn in funcs.items()}
        graphs = {key: val[0] for key, val in captured.items()}
        print(f'ROUTER_RAW_COLD_US m{m} {paired(graphs, flush, reps=40)}', flush=True)
    print(f'router_stack total {time.monotonic()-start:.2f}s', flush=True)

if __name__ == '__main__':
    main()
