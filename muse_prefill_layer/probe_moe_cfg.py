# okc: problem=muse_prefill_layer
"""Test sglang fused_experts on 16K prefill tokens."""

import torch
import types
from sglang.srt.layers.moe.fused_moe_triton import fused_experts, MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput

def prepare(weights):
    return {}

def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    device = data["stream"].device

    try:
        import inspect
        print("MoeRunnerConfig fields:", [x for x in dir(MoeRunnerConfig) if not x.startswith('_')])
        print("MoeRunnerConfig init sig:", inspect.signature(MoeRunnerConfig.__init__))
    except Exception as e:
        print("error inspect MoeRunnerConfig:", e)

    return torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=device), data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=device)
