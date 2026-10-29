# okc: problem=muse_moe_w4a16
import torch
import triton
import triton.language as tl
from nvfp4 import dequantize, quantize

@triton.jit
def _dequant_nvfp4_kernel(
    packed_ptr,
    sf_ptr,
    gs_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Each byte holds 2 values along K
    offs_packed = offs_n[:, None] * (K // 2) + (offs_k[None, :] // 2)
    bytes_val = tl.load(packed_ptr + offs_packed)

    is_hi = (offs_k[None, :] % 2) == 1
    codes = tl.where(is_hi, (bytes_val >> 4) & 0x0F, bytes_val & 0x0F)

    sign = (codes & 0x08) != 0
    mag_code = codes & 0x07

    # LUT: (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    # Piecewise or direct LUT
    # mag_code: 0->0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
    mag = tl.where(mag_code < 5, mag_code.to(tl.float32) * 0.5,
          tl.where(mag_code == 5, 3.0,
          tl.where(mag_code == 6, 4.0, 6.0)))

    val = tl.where(sign, -mag, mag)

    # Scale factors: 1 scale per 16 elements along K
    offs_sf = offs_n[:, None] * (K // 16) + (offs_k[None, :] // 16)
    # sf is fp8 e4m3fn, cast to float32
    sf_val = tl.load(sf_ptr + offs_sf).to(tl.float32)
    gs = tl.load(gs_ptr)

    res = val * sf_val * gs
    offs_out = offs_n[:, None] * K + offs_k[None, :]
    tl.store(out_ptr + offs_out, res.to(out_ptr.dtype.element_ty))

def prepare(weights):
    return {}

def kernel_fn(data, state):
    device = data["hidden_states"].device
    # Test on a small tensor: N=128, K=256
    N, K = 128, 256
    x = torch.randn(N, K, device=device)
    packed, sf, gs = quantize(x)
    
    # Ground truth dequantize
    ref_dequant = dequantize(packed, sf, gs)
    
    # Triton dequantize
    triton_out = torch.empty((N, K), dtype=torch.float32, device=device)
    grid = (N // 64, K // 64)
    _dequant_nvfp4_kernel[grid](
        packed,
        sf,
        gs,
        triton_out,
        N=N,
        K=K,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    
    diff = (ref_dequant - triton_out).abs().max().item()
    print(f"NVFP4 Triton dequant max absolute difference: {diff}")
    assert diff == 0.0, f"Dequant mismatch: {diff}"
    print("MATCH 100% EXACT!")
    
    return data["hidden_states"]
