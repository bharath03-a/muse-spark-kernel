"""GPU regression tests for actual replay corruption and NaN rejection."""
from pathlib import Path
import modal

image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
         .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth',
                       'echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
         .entrypoint([]))
if modal.is_local():
    image = image.add_local_dir(Path(__file__).resolve().parent, '/root/research_harness',
                                ignore=['results/**', '__pycache__/**'])
app = modal.App('muse-replay-harness-validation', image=image)


@app.function(gpu='H200', timeout=90)
def validate():
    import sys
    sys.path.insert(0, '/root')
    import torch
    from research_harness.gpu import compare_graphs
    x = torch.ones((4, 256), dtype=torch.bfloat16, device='cuda')
    report = compare_graphs(lambda: x.clone(), lambda: x.clone(), repetitions=8)
    assert report['replay_stability']['passed']
    print('HARNESS_GPU_PASS stable', flush=True)
    counter = torch.zeros_like(x)

    def corrupt():
        counter.add_(1)
        return counter

    try:
        compare_graphs(lambda: x.clone(), corrupt, repetitions=8)
    except AssertionError as exc:
        assert 'drifted' in str(exc), str(exc)
    else:
        raise AssertionError('Replay corruption was accepted')
    print('HARNESS_GPU_PASS corruption_rejected', flush=True)
    try:
        compare_graphs(lambda: x.clone(), lambda: torch.full_like(x, float('nan')), repetitions=8)
    except AssertionError as exc:
        assert 'Nonfinite' in str(exc), str(exc)
    else:
        raise AssertionError('NaN was accepted')
    print('HARNESS_GPU_PASS nonfinite_rejected', flush=True)
