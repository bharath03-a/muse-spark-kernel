"""External microprobe: measure exact decoder indexing/copy launch fusion.

No submitted kernel imports this file. Includes full operations under graph
replay; graph-owned temporaries exclude Python allocation costs. These timings
do not predict whole-layer gains.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent.parent
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
         .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth',
                       'echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
         .entrypoint([]))
if modal.is_local():
    image = image.add_local_dir(ROOT / 'research_harness', '/root/research_harness', ignore=['results/**', '__pycache__/**'])
app = modal.App('muse-decoder-launch-fusions', image=image)
cache = modal.Volume.from_name('muse-ext-cache', create_if_missing=True)


@app.function(gpu='H200', timeout=120, volumes={'/cache': cache})
def compare():
    import sys
    sys.path.insert(0, '/root')
    from research_harness.environment import configure_h200_cache
    configure_h200_cache('/cache')
    import json
    import torch
    import triton
    import triton.language as tl
    from research_harness.gpu import compare_graphs

    @triton.jit
    def localize(ids, out, Start, count: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        start = tl.load(Start)
        v = tl.load(ids + i, i < count, other=0).to(tl.int32) - start
        # Legal ids and starts are both in [0,255], so one addition is exact.
        v = tl.where(v < 0, v + 256, v)
        tl.store(out + i, tl.where(v < 32, v, -1), i < count)

    @triton.jit
    def sliced_output(src, dst, count: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = i // 8192, i % 8192
        v = tl.load(src + row * 1024 + col, (i < count) & (col < 1024), other=0)
        tl.store(dst + i, v, i < count)

    torch.manual_seed(271828)
    for m in [1, 8, 16, 32, 128, 256, 257]:
        ids = torch.randint(0, 256, (m, 8), device='cuda', dtype=torch.int32)
        out = torch.empty_like(ids)
        start_tensor = torch.empty((1,), device='cuda', dtype=torch.int32)
        for start in [0, 1, 31, 32, 224, 255]:
            start_tensor.fill_(start)
            def old():
                v = (ids - start_tensor) % 256
                return torch.where(v < 32, v, -1)
            def new():
                localize[(triton.cdiv(ids.numel(), 256),)](ids, out, start_tensor, ids.numel(), 256)
                return out
            assert torch.equal(old(), new()), (m, start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            new()
        for start in [0, 255, 32, 224, 1, 31]:
            start_tensor.fill_(start)
            graph.replay()
            assert torch.equal(old(), out), ('fresh_start_graph', m, start)
        report = compare_graphs(old, new, repetitions=40)
        print('LAUNCH_FUSION_RESULT', json.dumps({'kind': 'local_ids', 'tokens': m, **report}), flush=True)
    for m in [1, 2, 3, 5, 8]:
        src = torch.randn(m, 1024, device='cuda', dtype=torch.bfloat16)
        a = torch.empty(m, 8192, device='cuda', dtype=torch.bfloat16)
        b = torch.empty_like(a)
        def old():
            a[:, :1024].copy_(src)
            a[:, 1024:].zero_()
            return a
        def new():
            sliced_output[(triton.cdiv(b.numel(), 256),)](src, b, b.numel(), 256)
            return b
        assert torch.equal(old(), new()), m
        report = compare_graphs(old, new, repetitions=40)
        print('LAUNCH_FUSION_RESULT', json.dumps({'kind': 'slice_copy_zero', 'tokens': m, **report}), flush=True)
    cache.commit()
    print('LAUNCH_FUSION_DONE', flush=True)
