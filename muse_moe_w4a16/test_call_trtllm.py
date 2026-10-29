# okc: problem=muse_moe_w4a16
import torch
import traceback
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    device = weights["w13"].device
    print("Testing trtllm_fp4_block_scale_routed_moe...")
    try:
        # Check signature and call
        fn = fi_moe.trtllm_fp4_block_scale_routed_moe
        print("Function found:", fn)
        # Create minimal dummy inputs on device
        num_tokens = 1
        num_experts = 256
        local_num_experts = 32
        local_expert_offset = 0
        hidden_size = 4096
        intermediate_size = 4096
        top_k = 8

        x = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

        w1 = torch.zeros(local_num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.uint8, device=device)
        w1_s = torch.ones(local_num_experts, 2 * intermediate_size, hidden_size // 16, dtype=torch.float8_e4m3fn, device=device)
        w2 = torch.zeros(local_num_experts, hidden_size, intermediate_size // 2, dtype=torch.uint8, device=device)
        w2_s = torch.ones(local_num_experts, hidden_size, intermediate_size // 16, dtype=torch.float8_e4m3fn, device=device)

        out = fn(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=None,
            gemm1_weights=w1,
            gemm1_weights_scale=w1_s,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_s,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=0,
            do_finalize=False,
        )
        print("SUCCESS! Output type:", type(out))
        if isinstance(out, (list, tuple)):
            print("Output shapes:", [t.shape for t in out if isinstance(t, torch.Tensor)])
        elif isinstance(out, torch.Tensor):
            print("Output shape:", out.shape)
    except Exception as e:
        print("FAILED trtllm call:")
        traceback.print_exc()

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
