# okc: problem=muse_moe_w4a16
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from nvfp4 import dequantize

POST_NORM_EPS = 1e-8
TOP_K = 8

@triton.jit
def _fused_nvfp4_swiglu_gemv_kernel(
    x_ptr,           # [K]
    topk_ids_ptr,    # [TOP_K]
    w13_ptr,         # [32, 2 * I, K // 2]
    w13_sf_ptr,      # [32, 2 * I, K // 16]
    w13_gs_ptr,      # [32]
    out_ptr,         # [TOP_K, I]
    local_ids_ptr,   # [TOP_K]
    local_start: int,
    local_count: int,
    I: tl.constexpr, # 4096
    K: tl.constexpr, # 4096
    BLOCK_I: tl.constexpr, # 32
    BLOCK_K: tl.constexpr, # 128
):
    pid_k = tl.program_id(0) # which route in [0, TOP_K)
    pid_i = tl.program_id(1) # row block in [0, I // BLOCK_I)

    exp_id = tl.load(topk_ids_ptr + pid_k)
    if exp_id < local_start or exp_id >= (local_start + local_count):
        if pid_i == 0:
            tl.store(local_ids_ptr + pid_k, -1)
        return

    local_idx = exp_id - local_start
    if pid_i == 0:
        tl.store(local_ids_ptr + pid_k, local_idx)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I

    gate_rows = offs_i
    up_rows = offs_i + I

    acc_gate = tl.zeros((BLOCK_I,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_I,), dtype=tl.float32)

    gs = tl.load(w13_gs_ptr + local_idx)

    stride_w_exp = 2 * I * (K // 2)
    stride_sf_exp = 2 * I * (K // 16)
    w_exp_ptr = w13_ptr + local_idx * stride_w_exp
    sf_exp_ptr = w13_sf_ptr + local_idx * stride_sf_exp

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_val = tl.load(x_ptr + offs_k).to(tl.float32)

        offs_k_byte = offs_k // 2
        is_hi = (offs_k % 2) == 1

        # Gate
        gate_bytes = tl.load(w_exp_ptr + gate_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        gate_codes = tl.where(is_hi[None, :], (gate_bytes >> 4) & 0x0F, gate_bytes & 0x0F)
        gate_sign = (gate_codes & 0x08) != 0
        gate_mag_code = gate_codes & 0x07
        gate_mag = tl.where(gate_mag_code < 5, gate_mag_code.to(tl.float32) * 0.5,
                   tl.where(gate_mag_code == 5, 3.0,
                   tl.where(gate_mag_code == 6, 4.0, 6.0)))
        gate_val = tl.where(gate_sign, -gate_mag, gate_mag)

        offs_k_sf = offs_k // 16
        gate_sf = tl.load(sf_exp_ptr + gate_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        gate_dequant = gate_val * gate_sf

        # Up
        up_bytes = tl.load(w_exp_ptr + up_rows[:, None] * (K // 2) + offs_k_byte[None, :])
        up_codes = tl.where(is_hi[None, :], (up_bytes >> 4) & 0x0F, up_bytes & 0x0F)
        up_sign = (up_codes & 0x08) != 0
        up_mag_code = up_codes & 0x07
        up_mag = tl.where(up_mag_code < 5, up_mag_code.to(tl.float32) * 0.5,
                 tl.where(up_mag_code == 5, 3.0,
                 tl.where(up_mag_code == 6, 4.0, 6.0)))
        up_val = tl.where(up_sign, -up_mag, up_mag)

        up_sf = tl.load(sf_exp_ptr + up_rows[:, None] * (K // 16) + offs_k_sf[None, :]).to(tl.float32)
        up_dequant = up_val * up_sf

        acc_gate += tl.sum(gate_dequant * x_val[None, :], axis=1)
        acc_up += tl.sum(up_dequant * x_val[None, :], axis=1)

    gate_final = acc_gate * gs
    up_final = acc_up * gs

    silu_gate = gate_final / (1.0 + tl.exp(-gate_final))
    swiglu = silu_gate * up_final

    out_row_ptr = out_ptr + pid_k * I
    tl.store(out_row_ptr + offs_i, swiglu.to(out_row_ptr.dtype.element_ty), mask=mask_i)


@triton.jit
def _fused_nvfp4_down_gemv_kernel(
    y_inter_ptr,     # [TOP_K, I]
    topk_ids_ptr,    # [TOP_K]
    w2_ptr,          # [32, H, I // 2]
    w2_sf_ptr,       # [32, H, I // 16]
    w2_gs_ptr,       # [32]
    out_ptr,         # [TOP_K, H]
    local_start: int,
    local_count: int,
    H: tl.constexpr, # 4096
    I: tl.constexpr, # 4096
    BLOCK_H: tl.constexpr, # 32
    BLOCK_I: tl.constexpr, # 128
):
    pid_k = tl.program_id(0) # which route in [0, TOP_K)
    pid_h = tl.program_id(1) # row block in [0, H // BLOCK_H)

    exp_id = tl.load(topk_ids_ptr + pid_k)
    if exp_id < local_start or exp_id >= (local_start + local_count):
        # Clear output row
        out_row_ptr = out_ptr + pid_k * H
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        tl.store(out_row_ptr + offs_h, 0.0, mask=offs_h < H)
        return

    local_idx = exp_id - local_start
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    gs = tl.load(w2_gs_ptr + local_idx)

    stride_w2_exp = H * (I // 2)
    stride_sf2_exp = H * (I // 16)
    w2_exp_ptr = w2_ptr + local_idx * stride_w2_exp
    sf2_exp_ptr = w2_sf_ptr + local_idx * stride_sf2_exp
    y_row_ptr = y_inter_ptr + pid_k * I

    for i_start in range(0, I, BLOCK_I):
        offs_i = i_start + tl.arange(0, BLOCK_I)
        y_val = tl.load(y_row_ptr + offs_i).to(tl.float32)

        offs_i_byte = offs_i // 2
        is_hi = (offs_i % 2) == 1

        w2_bytes = tl.load(w2_exp_ptr + offs_h[:, None] * (I // 2) + offs_i_byte[None, :])
        w2_codes = tl.where(is_hi[None, :], (w2_bytes >> 4) & 0x0F, w2_bytes & 0x0F)
        w2_sign = (w2_codes & 0x08) != 0
        w2_mag_code = w2_codes & 0x07
        w2_mag = tl.where(w2_mag_code < 5, w2_mag_code.to(tl.float32) * 0.5,
                 tl.where(w2_mag_code == 5, 3.0,
                 tl.where(w2_mag_code == 6, 4.0, 6.0)))
        w2_val = tl.where(w2_sign, -w2_mag, w2_mag)

        offs_i_sf = offs_i // 16
        w2_sf = tl.load(sf2_exp_ptr + offs_h[:, None] * (I // 16) + offs_i_sf[None, :]).to(tl.float32)
        w2_dequant = w2_val * w2_sf

        acc += tl.sum(w2_dequant * y_val[None, :], axis=1)

    res = acc * gs
    out_row_ptr = out_ptr + pid_k * H
    tl.store(out_row_ptr + offs_h, res.to(out_row_ptr.dtype.element_ty), mask=mask_h)


def prepare(weights):
    device = weights["w13"].device
    return {
        "w13": weights["w13"].contiguous(),
        "w13_sf": weights["w13_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].contiguous(),
        "w2": weights["w2"].contiguous(),
        "w2_sf": weights["w2_sf"].contiguous(),
        "w2_gs": weights["w2_gs"].contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "swiglu_buf": torch.empty((TOP_K, 4096), dtype=torch.bfloat16, device=device),
        "y_buf": torch.empty((1, TOP_K, 4096), dtype=torch.bfloat16, device=device),
        "local_ids_buf": torch.empty((1, TOP_K), dtype=torch.int32, device=device),
        "out_buf": torch.empty((1, 4096), dtype=torch.bfloat16, device=device),
    }


def kernel_fn(data, state):
    x = data["hidden_states"]
    num_tokens, H = x.shape
    topk_ids = data["topk_ids"]
    topk_weights = data["topk_weights"]
    cfg = data["config"]
    local_start = cfg["local_start"]
    local_count = cfg["local_count"]

    if num_tokens == 1:
        x_tok = x[0]
        y_buf = state["y_buf"]
        swiglu_buf = state["swiglu_buf"]
        local_ids_buf = state["local_ids_buf"]

        # 1. Fused SwiGLU GEMV in Triton
        _fused_nvfp4_swiglu_gemv_kernel[(TOP_K, 4096 // 32)](
            x_tok,
            topk_ids[0],
            state["w13"],
            state["w13_sf"],
            state["w13_gs"],
            swiglu_buf,
            local_ids_buf[0],
            local_start=local_start,
            local_count=local_count,
            I=4096,
            K=4096,
            BLOCK_I=32,
            BLOCK_K=128,
            num_warps=4,
        )

        # 2. Fused Down GEMV in Triton
        _fused_nvfp4_down_gemv_kernel[(TOP_K, 4096 // 32)](
            swiglu_buf,
            topk_ids[0],
            state["w2"],
            state["w2_sf"],
            state["w2_gs"],
            y_buf[0],
            local_start=local_start,
            local_count=local_count,
            H=4096,
            I=4096,
            BLOCK_H=32,
            BLOCK_I=128,
            num_warps=4,
        )

        # 3. Post expert norm and weighted sum
        out = state["out_buf"][:num_tokens]
        _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
            y_buf,
            state["norm_w"],
            topk_weights,
            out,
            local_ids_buf,
            TOPK=TOP_K,
            H=4096,
            eps=POST_NORM_EPS,
            APPLY_NORM=True,
            HAS_WEIGHT=True,
            WEIGHT_FIRST=True,
            HAS_IDS=True,
            BLOCK_H=4096,
            num_warps=8,
            num_stages=2,
        )
        return out

    return x
