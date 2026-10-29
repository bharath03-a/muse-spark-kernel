# okc: problem=muse_prefill_layer
"""Probe Marlin supported moe_block_size values."""

import torch
import types
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm, _jit_moe_wna16_marlin_module
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.kernels.ops.moe import moe_align_block_size

def prepare(weights):
    device = weights["qkv_gate"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "layer": layer,
        "ws": marlin_make_workspace(device, max_blocks_per_sm=32),
    }

def kernel_fn(data, state):
    device = data["stream"].device
    tokens = data["config"]["tokens"]
    
    # Test different block_size_m
    layer = state["layer"]
    ws = state["ws"]
    
    for bs_m in [8, 16, 32, 48, 64, 128, 256, 512]:
        try:
            local_ids = torch.zeros((tokens, 8), dtype=torch.int32, device=device)
            max_padded = tokens * 8 + (256 + 1) * bs_m
            max_padded = (max_padded + 3) // 4 * 4
            sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
            block_experts = torch.empty((max_padded // bs_m,), dtype=torch.int32, device=device)
            n_padded = torch.empty((1,), dtype=torch.int32, device=device)
            cumsum = torch.empty((258,), dtype=torch.int32, device=device)
            
            moe_align_block_size(local_ids, 257, bs_m, sorted_ids, block_experts, n_padded, cumsum, True, False)
            c = torch.empty((tokens * 8, 8192), dtype=torch.bfloat16, device=device)
            x = torch.randn((tokens, 4096), dtype=torch.bfloat16, device=device)
            topk_w = torch.ones((tokens, 8), dtype=torch.float32, device=device)
            
            common = dict(moe_block_size=bs_m, mul_topk_weights=False, is_ep=True,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                          use_fp32_reduce=True, is_zp_float=False)
            
            c = moe_wna16_marlin_gemm(
                x, c, layer.w13_weight, None, layer.w13_weight_scale, layer.w13_weight_scale_2,
                None, None, None, ws, sorted_ids, block_experts, n_padded, topk_w,
                top_k=8, size_m=tokens, size_n=8192, size_k=4096, **common
            )
            print(f"block_size_m={bs_m}: SUCCESS!")
        except Exception as e:
            print(f"block_size_m={bs_m}: FAILED ({e})")
            
    return torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=device), data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=device)
