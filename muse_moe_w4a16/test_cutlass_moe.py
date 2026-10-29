# okc: problem=muse_moe_w4a16
import torch
import flashinfer.fused_moe as fi_moe

def round_up(x, y):
    return (x + y - 1) // y * y

def prepare(weights):
    device = weights["w13"].device
    # Prepare CUTLASS NVFP4 weights
    # w13: [32, 8192, 2048] uint8
    # w2: [32, 4096, 2048] uint8
    # w13_sf: [32, 8192, 256] fp8
    # w2_sf: [32, 4096, 256] fp8
    # w13_gs: [32] float32
    # w2_gs: [32] float32
    w1 = weights["w13"].contiguous()
    w2 = weights["w2"].contiguous()
    
    # Pad scales if needed: expected s1 shape is (32, round_up(8192, 128), round_up(256, 4))
    s1 = weights["w13_sf"].view(torch.uint8).contiguous()
    s2 = weights["w2_sf"].view(torch.uint8).contiguous()
    
    num_exp = w1.shape[0]
    a1_gs = torch.ones(num_exp, dtype=torch.float32, device=device)
    a2_gs = torch.ones(num_exp, dtype=torch.float32, device=device)
    fc1_dequant = weights["w13_gs"].float().contiguous()
    fc2_dequant = weights["w2_gs"].float().contiguous()
    
    return {
        "w1": w1.view(torch.int64),
        "w2": w2.view(torch.int64),
        "quant_scales": [a1_gs, s1.view(torch.int32), fc1_dequant, a2_gs, s2.view(torch.int32), fc2_dequant],
        "device": device,
    }

def kernel_fn(data, state):
    x = data["hidden_states"]
    topk_ids = data["topk_ids"].int().contiguous()
    topk_weights = data["topk_weights"].contiguous()
    
    print(f"Testing cutlass_fused_moe with x={x.shape}, topk_ids={topk_ids.shape}")
    try:
        out = fi_moe.cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=state["w1"],
            fc2_expert_weights=state["w2"],
            output_dtype=torch.bfloat16,
            quant_scales=state["quant_scales"],
        )
        print("cutlass_fused_moe succeeded! out shape:", out.shape)
        return out
    except Exception as e:
        print("cutlass_fused_moe error:", e)
        return x
