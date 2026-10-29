from sglang.kernels.ops.moe.moe_wna16_marlin import _jit_moe_wna16_marlin_module, _MAX_THREAD_N
import torch

mod = _jit_moe_wna16_marlin_module(torch.bfloat16, True, False)
print("mod:", mod)
print("_MAX_THREAD_N:", _MAX_THREAD_N)
