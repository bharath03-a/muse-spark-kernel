import torch
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import SMALL_NUMEL_LIMIT, moe_align_small_numel

topk_ids = torch.randint(0, 32, (1, 8), dtype=torch.int32, device="cuda")
block_size = 8
num_experts = 256

max_num_tokens_padded = (topk_ids.numel() * block_size + 3) // 4 * 4
sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
expert_ids = torch.empty((max_num_tokens_padded // block_size,), dtype=torch.int32, device="cuda")
num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")

print("Calling moe_align_small_numel directly...")
moe_align_small_numel(
    topk_ids,
    num_experts + 1,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
)
print("moe_align_small_numel success! n_padded:", num_tokens_post_pad.item())

# Test larger batch
topk_ids_large = torch.randint(0, 32, (32, 8), dtype=torch.int32, device="cuda")
max_num_tokens_padded_l = topk_ids_large.numel() + (num_experts + 1) * (16 - 1)
max_num_tokens_padded_l = (max_num_tokens_padded_l + 3) // 4 * 4
sorted_ids_l = torch.empty((max_num_tokens_padded_l,), dtype=torch.int32, device="cuda")
expert_ids_l = torch.empty((max_num_tokens_padded_l // 16,), dtype=torch.int32, device="cuda")
num_tokens_post_pad_l = torch.empty((1,), dtype=torch.int32, device="cuda")
cumsum_l = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")

sgl_moe_align_block_size(
    topk_ids_large,
    num_experts + 1,
    16,
    sorted_ids_l,
    expert_ids_l,
    num_tokens_post_pad_l,
    cumsum_l,
    True,
    False,
)
print("sgl_moe_align_block_size success! n_padded:", num_tokens_post_pad_l.item())
