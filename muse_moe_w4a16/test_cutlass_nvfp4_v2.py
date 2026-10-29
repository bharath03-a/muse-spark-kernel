# okc: problem=muse_moe_w4a16
import torch
import flashinfer.fused_moe as fi_moe
from flashinfer.tllm_enums import ActivationType

def prepare(weights):
    device = weights["w13"].device
    sl = slice(0, 32)
    w13 = weights["w13"][sl].contiguous()
    w2 = weights["w2"][sl].contiguous()
    w13_sf = weights["w13_sf"][sl].contiguous()
    w2_sf = weights["w2_sf"][sl].contiguous()
    w13_gs = weights["w13_gs"][sl].float().contiguous()
    w2_gs = weights["w2_gs"][sl].float().contiguous()

    a1_gs = torch.ones(32, dtype=torch.float32, device=device)
    a2_gs = torch.ones(32, dtype=torch.float32, device=device)

    quant_scales = [a1_gs, w13_sf, w13_gs, a2_gs, w2_sf, w2_gs]

    return {
        "w13": w13,
        "w2": w2,
        "quant_scales": quant_scales,
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }

def kernel_fn(data, state):
    x = data["hidden_states"]
    topk_ids = data["topk_ids"].int().contiguous()
    topk_weights = data["topk_weights"].float().contiguous()
    cfg = data["config"]

    print(f"=== Testing CUTLASS NVFP4 MoE: bs={cfg['bs']}, topk_ids={topk_ids.shape} ===")

    # Test 1: use_w4_group_scaling=False, use_packed_weights=True
    try:
        out = fi_moe.cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=state["w13"],
            fc2_expert_weights=state["w2"],
            output_dtype=torch.bfloat16,
            quant_scales=state["quant_scales"],
            ep_size=8,
            ep_rank=cfg["local_start"] // 32,
            use_w4_group_scaling=False,
            use_packed_weights=True,
            activation_type=ActivationType.Swiglu,
            use_fused_finalize=False,
        )
        print("Success with use_fused_finalize=False! shape:", out.shape)
        return out
    except Exception as e:
        print("Error with use_fused_finalize=False:", e)

    # Test 2: use_packed_weights=False
    try:
        out = fi_moe.cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=state["w13"],
            fc2_expert_weights=state["w2"],
            output_dtype=torch.bfloat16,
            quant_scales=state["quant_scales"],
            ep_size=8,
            ep_rank=cfg["local_start"] // 32,
            use_w4_group_scaling=False,
            use_packed_weights=False,
            activation_type=ActivationType.Swiglu,
            use_fused_finalize=False,
        )
        print("Success with use_packed_weights=False! shape:", out.shape)
        return out
    except Exception as e:
        print("Error with use_packed_weights=False:", e)

    return x
