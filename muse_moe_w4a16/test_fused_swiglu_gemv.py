# okc: problem=muse_moe_w4a16
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from nvfp4 import dequantize, quantize

@triton.jit
def _fused_nvfp4_swiglu_gemv_kernel(
    x_ptr,           # [K]
    w13_ptr,         # [2 * I, K // 2]
    w13_sf_ptr,      # [2 * I, K // 16]
    w13_gs_ptr,      # []
    out_ptr,         # [I]
    I: tl.constexpr, # 4096
    K: tl.constexpr, # 4096
    BLOCK_I: tl.constexpr, # e.g. 32
    BLOCK_K: tl.constexpr, # e.g. 128
):
    pid_i = tl.program_id(0)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I

    # Rows for gate and up
    gate_rows = offs_i
    up_rows = offs_i + I

    acc_gate = tl.zeros((BLOCK_I,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_I,), dtype=tl.float32)

    gs = tl.load(w13_gs_ptr)

    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load x [BLOCK_K]
        x_val = tl.load(x_ptr + offs_k).to(tl.float32)

        # Offsets in packed weight: [BLOCK_I, BLOCK_K // 2]
        offs_k_byte = offs_k // 2
        is_hi = (offs_k % 2) == 1

        # Load gate bytes
        gate_bytes = tl.load(w13_ptr + gate_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        gate_codes = tl.where(is_hi[None, :], (gate_bytes >> 4) & 0x0F, gate_bytes & 0x0F)
        gate_sign = (gate_codes & 0x08) != 0
        gate_mag_code = gate_codes & 0x07
        gate_mag = tl.where(gate_mag_code < 5, gate_mag_code.to(tl.float32) * 0.5,
                   tl.where(gate_mag_code == 5, 3.0,
                   tl.where(gate_mag_code == 6, 4.0, 6.0)))
        gate_val = tl.where(gate_sign, -gate_mag, gate_mag)

        # Load gate scale factors [BLOCK_I, BLOCK_K // 16]
        offs_k_sf = offs_k // 16
        gate_sf = tl.load(w13_sf_ptr + gate_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        gate_dequant = gate_val * gate_sf

        # Load up bytes
        up_bytes = tl.load(w13_ptr + up_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        up_codes = tl.where(is_hi[None, :], (up_bytes >> 4) & 0x0F, up_bytes & 0x0F)
        up_sign = (up_codes & 0x08) != 0
        up_mag_code = up_codes & 0x07
        up_mag = tl.where(up_mag_code < 5, up_mag_code.to(tl.float32) * 0.5,
                 tl.where(up_mag_code == 5, 3.0,
                 tl.where(up_mag_code == 6, 4.0, 6.0)))
        up_val = tl.where(up_sign, -up_mag, up_mag)

        # Load up scale factors
        up_sf = tl.load(w13_sf_ptr + up_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        up_dequant = up_val * up_sf

        acc_gate += tl.sum(gate_dequant * x_val[None, :], axis=1)
        acc_up += tl.sum(up_dequant * x_val[None, :], axis=1)

    gate_final = acc_gate * gs
    up_final = acc_up * gs

    # SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    silu_gate = gate_final / (1.0 + tl.exp(-gate_final))
    swiglu = silu_gate * up_final

    tl.store(out_ptr + offs_i, swiglu.to(out_ptr.dtype.element_ty), mask=mask_i)

def prepare(weights):
    return {}

def kernel_fn(data, state):
    device = data["hidden_states"].device
    I, K = 4096, 4096
    
    # Generate test vector and NVFP4 weights
    torch.manual_seed(42)
    x = torch.randn((K,), dtype=torch.bfloat16, device=device)
    w_gate = torch.randn((I, K), device=device)
    w_up = torch.randn((I, K), device=device)
    w_full = torch.cat([w_gate, w_up], dim=0)
    
    packed, sf, gs = quantize(w_full)
    
    # Ground truth reference
    w_dequant = dequantize(packed, sf, gs).to(torch.bfloat16)
    ref_gate = F.linear(x.float(), w_dequant[:I].float())
    ref_up = F.linear(x.float(), w_dequant[I:].float())
    ref_swiglu = (F.silu(ref_gate) * ref_up).to(torch.bfloat16)
    
    # Triton fused GEMV
    triton_out = torch.empty((I,), dtype=torch.bfloat16, device=device)
    grid = (I // 32,)
    _fused_nvfp4_swiglu_gemv_kernel[grid](
        x,
        packed,
        sf,
        gs,
        triton_out,
        I=I,
        K=K,
        BLOCK_I=32,
        BLOCK_K=128,
        num_warps=4,
    )
    
    rel_err = (ref_swiglu.float() - triton_out.float()).norm() / ref_swiglu.float().norm()
    print(f"Fused SwiGLU GEMV relative error: {rel_err.item()}")
    assert rel_err.item() < 0.005, f"High error: {rel_err.item()}"
    print("FUSED SWIGLU GEMV VERIFIED SUCCESS!")
    
    return data["hidden_states"]
