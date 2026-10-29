# okc: problem=muse_prefill_layer
"""Test trtllm_bf16_routed_moe and dequantize kernel in prefill."""

import torch
import triton
import triton.language as tl
import flashinfer.fused_moe as ffm

@triton.jit
def _dequant_nvfp4_kernel(
    packed_ptr,
    sf_ptr,
    gs_ptr,
    out_ptr,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_en = tl.program_id(0)
    pid_k = tl.program_id(1)

    pid_e = pid_en // (N // BLOCK_N)
    pid_n = pid_en % (N // BLOCK_N)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    offs_packed = pid_e * (N * (K // 2)) + offs_n[:, None] * (K // 2) + (offs_k[None, :] // 2)
    bytes_val = tl.load(packed_ptr + offs_packed)

    is_hi = (offs_k[None, :] % 2) == 1
    codes = tl.where(is_hi, (bytes_val >> 4) & 0x0F, bytes_val & 0x0F)

    sign = (codes & 0x08) != 0
    mag_code = codes & 0x07

    mag = tl.where(mag_code < 5, mag_code.to(tl.float32) * 0.5,
          tl.where(mag_code == 5, 3.0,
          tl.where(mag_code == 6, 4.0, 6.0)))

    val = tl.where(sign, -mag, mag)

    offs_sf = pid_e * (N * (K // 16)) + offs_n[:, None] * (K // 16) + (offs_k[None, :] // 16)
    sf_val = tl.load(sf_ptr + offs_sf).to(tl.float32)
    gs = tl.load(gs_ptr + pid_e)

    res = val * sf_val * gs
    offs_out = pid_e * (N * K) + offs_n[:, None] * K + offs_k[None, :]
    tl.store(out_ptr + offs_out, res.to(out_ptr.dtype.element_ty))

def prepare(weights):
    device = weights["qkv_gate"].device
    # Preallocate dequantized buffers
    w13_bf16 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.empty((32, 4096, 4096), dtype=torch.bfloat16, device=device)
    return {
        "w13": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "w13_bf16": w13_bf16,
        "w2_bf16": w2_bf16,
    }

def kernel_fn(data, state):
    tokens = data["config"]["tokens"]
    device = data["stream"].device

    # 1. Dequantize w13: [32, 8192, 4096]
    grid_w13 = (32 * (8192 // 128), 4096 // 128)
    _dequant_nvfp4_kernel[grid_w13](
        state["w13"], state["w13_sf"], state["w13_gs"], state["w13_bf16"],
        E=32, N=8192, K=4096, BLOCK_N=128, BLOCK_K=128,
    )

    # 2. Dequantize w2: [32, 4096, 4096]
    grid_w2 = (32 * (4096 // 128), 4096 // 128)
    _dequant_nvfp4_kernel[grid_w2](
        state["w2"], state["w2_sf"], state["w2_gs"], state["w2_bf16"],
        E=32, N=4096, K=4096, BLOCK_N=128, BLOCK_K=128,
    )

    # 3. Test trtllm_bf16_routed_moe
    try:
        topk_ids = torch.zeros((tokens, 8), dtype=torch.int32, device=device)
        topk_weights = torch.ones((tokens, 8), dtype=torch.float32, device=device)
        hidden_states = torch.randn((tokens, 4096), dtype=torch.bfloat16, device=device)
        
        # trtllm expects topk pair: (topk_ids, topk_weights)
        out = ffm.trtllm_bf16_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            hidden_states=hidden_states,
            gemm1_weights=state["w13_bf16"],
            gemm2_weights=state["w2_bf16"],
            num_experts=256,
            top_k=8,
            n_group=None,
            topk_group=None,
            intermediate_size=4096,
            local_expert_offset=0,
            local_num_experts=32,
            use_shuffled_weight=False,
            do_finalize=True,
        )
        print("trtllm_bf16_routed_moe SUCCESS! out shape:", out.shape if hasattr(out, 'shape') else len(out))
    except Exception as e:
        print("trtllm_bf16_routed_moe error:", e)

    return torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=device), data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=device)
