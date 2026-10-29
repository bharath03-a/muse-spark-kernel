# okc: problem=muse_prefill_layer
"""Probe flashinfer SM90 CUTLASS prepare_cutlass_w4a16_weights full source."""

import inspect
import torch

def prepare(weights):
    import flashinfer.fused_moe.prepare as fmp
    src = inspect.getsource(fmp.prepare_cutlass_w4a16_weights)
    print("=== prepare_cutlass_w4a16_weights lines 40-140 ===")
    lines = src.split("\n")
    for i, line in enumerate(lines[40:140], start=40):
        print(f"{i}: {line}")

    src_scale = inspect.getsource(fmp.interleave_moe_scales_for_sm90_mixed_gemm)
    print("\n=== interleave_moe_scales_for_sm90_mixed_gemm ===")
    for i, line in enumerate(src_scale.split("\n")[:50]):
        print(f"{i}: {line}")

    return {}

def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    device = data["stream"].device
    return (
        torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=device),
        data["stream"],
        torch.zeros((tokens, 8), dtype=torch.int32, device=device),
    )
