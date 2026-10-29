"""Research-only paired timing. Import this on the Modal H200 worker."""
from .stats import paired_summary


def compare_graphs(baseline, candidate, *, repetitions=40, warmup=3,
                   flush_bytes=256 * 1024 * 1024, replay_overhead_us=0.0,
                   reset_inputs=None, stability_rtol=0.01, stability_atol=1e-5):
    """Callers validate fresh seeds first. Mutable state must be reset to identical
    values by reset_inputs() before each replay, or be provably idempotent.
    reset_inputs must only restore inputs, never compute candidate intermediates.
    """
    import torch
    if repetitions < 8 or repetitions % 2 or warmup < 1:
        raise ValueError('Need >=8 even repetitions and >=1 warmup')
    if flush_bytes < 128 * 1024 * 1024:
        raise ValueError('Cold-L2 buffer must exceed H200 L2')
    fns = {'baseline': baseline, 'candidate': candidate}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs, snapshots = {}, {}, {}
    def map_tensors(value, fn):
        if isinstance(value, torch.Tensor):
            return fn(value)
        if isinstance(value, dict):
            return {k: map_tensors(v, fn) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [map_tensors(v, fn) for v in value]
        return value
    def finite(tensor):
        if not torch.isfinite(tensor).all().item():
            raise AssertionError('Nonfinite graph output')
        return tensor
    with torch.cuda.stream(stream):
        for name, fn in fns.items():
            for _ in range(warmup):
                if reset_inputs is not None:
                    reset_inputs()
                outputs[name] = fn()
            snapshots[name] = map_tensors(outputs[name], lambda t: t.clone())
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for value in snapshots.values():
        map_tensors(value, finite)
    # Retain graph outputs and separate pools throughout timing.
    for name, fn in fns.items():
        # This reset must be outside capture, on the stream being captured.
        if reset_inputs is not None:
            with torch.cuda.stream(stream):
                reset_inputs()
            stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs[name] = fn()
        graphs[name] = graph
    torch.cuda.current_stream().wait_stream(stream)
    flush = torch.ones(flush_bytes, device='cuda', dtype=torch.uint8)
    sink = torch.empty((), device='cuda', dtype=torch.int64)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for rep in range(repetitions):
        order = ['baseline', 'candidate'] if rep % 2 == 0 else ['candidate', 'baseline']
        row = {'order': order}
        for name in order:
            if reset_inputs is not None:
                reset_inputs()
            # Read an oversized buffer, as the judge does; outside the events.
            torch.sum(flush, dim=0, dtype=torch.int64, out=sink)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            row[name + '_us'] = start.elapsed_time(end) * 1000
        samples.append(row)
    # A persistent scratch counter can work once and corrupt later replays.
    # Detect drift before returning any timing evidence. Reference correctness
    # is still the caller's responsibility; this is only a replay guard.
    def check_replay(actual, expected):
        if isinstance(actual, torch.Tensor):
            finite(actual)
            if not actual.is_floating_point() and not actual.is_complex():
                assert torch.equal(actual, expected), 'Discrete graph output drifted'
                return
            # Parallel reductions may move a few near-zero BF16 values by one
            # ULP. Use norm error, like the challenge, instead of pointwise
            # relative error which diverges near cancellation.
            delta = (actual.float() - expected.float()).reshape(-1)
            ref = expected.float().reshape(-1)
            bound = stability_rtol * torch.linalg.vector_norm(ref) + stability_atol * ref.numel() ** .5
            assert (torch.linalg.vector_norm(delta) <= bound).item(), 'Graph output norm drifted'
            if actual.ndim >= 2:
                width = actual.shape[-1]
                error_rows = torch.linalg.vector_norm(delta.reshape(-1, width), dim=1)
                ref_rows = torch.linalg.vector_norm(ref.reshape(-1, width), dim=1)
                row_bound = stability_rtol * ref_rows.clamp_min(ref_rows.mean() * .1) + stability_atol * width ** .5
                assert (error_rows <= 5 * row_bound).all().item(), 'Graph output row drifted'
        elif isinstance(actual, dict):
            for key in actual:
                check_replay(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            for a, b in zip(actual, expected, strict=True):
                check_replay(a, b)
    for name in fns:
        check_replay(outputs[name], snapshots[name])
    report = paired_summary(samples, replay_overhead_us=replay_overhead_us)
    report['replay_stability'] = dict(passed=True, metric='relative_l2_and_rows', rtol=stability_rtol, atol=stability_atol)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report['environment'] = dict(gpu=props.name, capability=[props.major, props.minor],
                                 total_memory=props.total_memory, torch=torch.__version__,
                                 cuda=torch.version.cuda)
    return report
