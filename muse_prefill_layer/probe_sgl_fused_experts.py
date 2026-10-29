# okc: problem=muse_prefill_layer
"""Probe sglang fused_experts with dequantized weights."""

import torch
import types
from sglang.srt.layers.moe.fused_moe_triton import fused_experts, moe_align_block_size

def prepare(weights):
    return {}

def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    device = data["stream"].device

    try:
        import sglang.srt.layers.moe.fused_moe_triton as fmt
        import inspect
        print("fused_moe_triton source file:", fmt.__file__)
        lines = inspect.getsource(fmt.fused_experts)
        print("fused_experts first 40 lines:")
        for line in lines.split("\n")[:40]:
            print(" ", line)
    except Exception as e:
        print("error inspect fused_experts:", e)

    return torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=device), data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=device)
