"""Minimal audit wrapper around EXISTING register-A compressed WGMMA source.

This does not introduce a new kernel or claim a speedup. Import/prepare/run only
when execution is authorized. File can be imported with bench on sys.path.
"""
import torch
import zip_gluon_dev as Z


def prepare_probe(w, m=32):
    if m not in (1, 8, 16, 32):
        raise ValueError('small-token audit only')
    bt = max(8, m)
    compressed = Z.encode(w, warps=4, dense=False)
    dense = Z.encode(w, warps=4, dense=True)
    Z.round_trip(w, compressed)
    return dict(
        weight=w, m=m,
        compressed=Z.Runner(compressed, mma=3, kg=1, stages=8, bt=bt, q=1),
        dense=Z.Runner(dense, mma=3, kg=1, stages=8, bt=bt, q=1),
        metadata=dict(mma=3, warps=4, kg=1, stages=8, bt=bt, q=1,
                      packed_bytes=compressed.nbytes(), original_bytes=w.numel()*2,
                      note='existing zip_gluon_dev.zgemv source; not a new mechanism'))


def run(x, state, variant='compressed'):
    if x.shape[0] != state['m'] or x.shape[1] != state['weight'].shape[1]:
        raise ValueError('unexpected input shape')
    out = torch.empty((x.shape[0], state['weight'].shape[0]),
                      device=x.device, dtype=torch.bfloat16)
    state[variant](x, out)
    return out


def expected_codegen_checks():
    return (
        'wgmma.mma_async.*.f32.bf16.bf16 uses A register braces, not A descriptor',
        'DLL -> DotOperandLayout0 conversion passes assert_trivial',
        'compressed planes load to registers; no decoded BF16 W store/reload in mainloop',
        'separate escape-fix shared/layout activity from mainloop when counting stores',
        'record registers, shared bytes, spills, commit/wait counts and cold paired timings',
    )
