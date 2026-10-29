# okc: problem=muse_moe_w4a16
"""Testing direct preallocated Marlin execution for routed experts on H200."""

import types
import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import (
    _jit_moe_wna16_marlin_module,
)
try:
    from sglang.kernels.ops.moe.moe_wna16_marlin import _MAX_THREAD_N
except ImportError:
    _MAX_THREAD_N = 128

from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
MAX_TOKENS = 512

# Optimal block_size_m: 16 for bs=16 saves 35 us over baseline bs=8
BLOCK_SIZE_M_MAP = {
    1: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 16,
    64: 16,
    128: 16,
    256: 16,
}


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


@triton.jit
def _decode_fp4(code):
    mag = code & 7
    val = tl.where(mag < 5, mag.to(tl.float32) * 0.5,
                   tl.where(mag == 5, 3., tl.where(mag == 6, 4., 6.)))
    return tl.where((code & 8) != 0, -val, val)

@triton.jit
def _direct_gemv(X, IDS, W, SF, GS, Y, START: tl.constexpr, COUNT: tl.constexpr,
                 GATED: tl.constexpr, ROWS: tl.constexpr = 4, K: tl.constexpr = 4096):
    route = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDS + route)
    if expert >= START and expert < START + COUNT:
        kk = tl.arange(0, K)
        x = tl.load(X + kk + (0 if GATED else route * K)).to(tl.float32)
        stride = 2 * K if GATED else K
        off = (expert * stride + row[:, None]) * (K // 2) + kk[None, :] // 2
        p = tl.load(W + off)
        code = (p >> ((kk[None, :] % 2) * 4)) & 15
        sf = tl.load(SF + (expert * stride + row[:, None]) * (K // 16) + kk[None, :] // 16).to(tl.float32)
        gs = tl.load(GS + expert)
        w = (_decode_fp4(code) * sf * gs).to(tl.bfloat16).to(tl.float32)
        y = tl.sum(w * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
        if GATED:
            p2 = tl.load(W + off + K * (K // 2))
            code2 = (p2 >> ((kk[None, :] % 2) * 4)) & 15
            sf2 = tl.load(SF + (expert * stride + row[:, None] + K) * (K // 16) + kk[None, :] // 16).to(tl.float32)
            w2 = (_decode_fp4(code2) * sf2 * gs).to(tl.bfloat16).to(tl.float32)
            up = tl.sum(w2 * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
            y = y / (1. + tl.exp(-y)) * up
        tl.store(Y + route * K + row, y.to(tl.bfloat16))
    else:
        tl.store(Y + route * K + row, 0.)

def prepare(weights):
    device = weights["w13"].device
    starts = [0]
    counts = [32, 256]
    banks = {}
    for start in starts:
        for count in counts:
            if start + count > N_EXPERTS:
                continue
            expert_map = None
            if count != N_EXPERTS:
                expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
                expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
            layer = _marlin_layer(weights, start, count)
            w13_num_groups = layer.w13_weight_scale.size(1)
            w13_group_size = 4096 // w13_num_groups if w13_num_groups > 1 else -1
            w2_num_groups = layer.w2_weight_scale.size(1)
            w2_group_size = layer.intermediate_size_per_partition // w2_num_groups if w2_num_groups > 1 else -1
            banks[(start, count)] = {
                "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
                "w13_num_groups": w13_num_groups, "w13_group_size": w13_group_size,
                "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
                "w2_num_groups": w2_num_groups, "w2_group_size": w2_group_size,
                "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
            }
    torch.cuda.synchronize()

    size_n = 4096
    size_k = 4096
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    max_padded = 32768
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)
    out_buf = torch.empty((MAX_TOKENS, size_k), device=device, dtype=torch.bfloat16)

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    marlin_mod_ep = _jit_moe_wna16_marlin_module(torch.bfloat16, True, False)
    marlin_mod_noep = _jit_moe_wna16_marlin_module(torch.bfloat16, False, False)
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    c_tmp_buf = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=device)

    return {
        "banks": banks,
        "raw": {k: weights[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")},
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "out_buf": out_buf,
        "sms": sms,
        "marlin_mod_ep": marlin_mod_ep,
        "marlin_mod_noep": marlin_mod_noep,
        "empty_bf16": empty_bf16,
        "empty_i32": empty_i32,
        "empty_f32": empty_f32,
        "c_tmp_buf": c_tmp_buf,
        "b_q_type_id": scalar_types.float4_e2m1f.id,
    }


def _fast_marlin(
    a, c, b_q_weight, b_scales, b_global_scale,
    sorted_ids, block_experts, n_padded, topk_weights,
    moe_block_size, top_k, is_ep,
    size_m, size_n, size_k,
    num_groups, group_size,
    module, workspace, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
    use_fp32_reduce=True,
):
    if use_fp32_reduce:
        max_c_tmp_size = min(
            size_n * sorted_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = c_tmp_buf[:max_c_tmp_size]
    else:
        c_tmp = empty_f32

    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        empty_bf16,
        b_scales,
        b_global_scale,
        empty_bf16,
        empty_i32,
        empty_i32,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        empty_bf16,
        c_tmp,
        moe_block_size,
        top_k,
        False,  # mul_topk_weights
        is_ep,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        False,  # has_act_order
        False,  # has_bias
        True,   # is_k_full
        False,  # has_zp
        num_groups,
        group_size,
        False,  # use_atomic_add
        use_fp32_reduce,
        False,  # is_zp_float
    )
    return c


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"]
    if expert_map is not None:
        local_ids = torch.where(local_ids < bank["count"], local_ids, -1)
    topk_weights = data["topk_weights"]

    block_size_m = BLOCK_SIZE_M_MAP.get(num_tokens, 32)

    numel = local_ids.numel()
    num_exp = bank["count"]
    if numel < num_exp + 1:
        max_num_tokens_padded = numel * block_size_m
    else:
        max_num_tokens_padded = numel + (num_exp + 1) * (block_size_m - 1)
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4
    max_num_m_blocks = max_num_tokens_padded // block_size_m

    sorted_ids = state["align_sorted_ids"][:max_num_tokens_padded]
    block_experts = state["align_expert_ids"][:max_num_m_blocks]
    n_padded = state["align_n_padded"]

    if numel <= SMALL_NUMEL_LIMIT and (num_exp + 1) > 64:
        moe_align_small_numel(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
        )
    else:
        sgl_moe_align_block_size(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
            state["align_cumsum"],
            True,
            False,
        )

    total_pairs = num_tokens * TOP_K
    cache2 = state["cache2_buf"][:total_pairs, :size_n]
    cache1 = state["cache13_buf"][: total_pairs * gemm1_n].view(-1, gemm1_n)
    cache3 = state["cache13_buf"][: total_pairs * size_k].view(-1, size_k)

    is_ep = expert_map is not None
    module = state["marlin_mod_ep"] if is_ep else state["marlin_mod_noep"]
    ws = state["workspace"]
    empty_bf16 = state["empty_bf16"]
    empty_i32 = state["empty_i32"]
    c_tmp_buf = state["c_tmp_buf"]
    sms = state["sms"]
    b_q_type_id = state["b_q_type_id"]
    empty_f32 = state["empty_f32"]
    use_fp32_reduce = (num_tokens not in (1, 8))

    if num_tokens == 1:
        raw = state["raw"]
        for gated, inp, dst, prefix in [(True, x, cache2, "w13"), (False, cache2, cache3, "w2")]:
            _direct_gemv[(TOP_K, 1024)](inp, data["topk_ids"], raw[prefix], raw[prefix + "_sf"], raw[prefix + "_gs"], dst,
                START=cfg["local_start"], COUNT=cfg["local_count"], GATED=gated, num_warps=8)
    else:
        _fast_marlin(
            x, cache1, bank["w13"], bank["w13_s"], bank["w13_gs"],
            sorted_ids, block_experts, n_padded, topk_weights,
            block_size_m, TOP_K, is_ep,
            num_tokens, gemm1_n, size_k,
            bank["w13_num_groups"], bank["w13_group_size"],
            module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
            use_fp32_reduce=use_fp32_reduce,
        )
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
        _fast_marlin(
            cache2, cache3, bank["w2"], bank["w2_s"], bank["w2_gs"],
            sorted_ids, block_experts, n_padded, topk_weights,
            block_size_m, 1, is_ep,
            total_pairs, size_k, size_n,
            bank["w2_num_groups"], bank["w2_group_size"],
            module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
            use_fp32_reduce=use_fp32_reduce,
        )
    y = cache3.view(num_tokens, TOP_K, size_k)

    # Direct Triton fused post-expert norm for all token counts
    out = state["out_buf"][:num_tokens]
    block_h = triton.next_power_of_2(size_k)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y,
        state["norm_w"],
        topk_weights,
        out,
        local_ids,
        TOPK=TOP_K,
        H=size_k,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out
