import sys
from pathlib import Path
import torch
import triton

# Add muse_moe_w4a16 to sys.path
sys.path.insert(0, "/root/proj/../muse_moe_w4a16")
# But wait, in Modal REMOTE_DIR is /root/proj (which is muse_bf16_proj).
# Let us copy reference, checks, nvfp4 or test directly.
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import SMALL_NUMEL_LIMIT, moe_align_small_numel
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import _post_expert_norm_weighted_sum_kernel
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

print("All imports succeeded!")
