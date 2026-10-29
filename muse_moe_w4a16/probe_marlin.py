# okc: problem=muse_moe_w4a16
import inspect
import torch
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
import sgl_kernel

def prepare(weights):
    return {}

def kernel_fn(data, state):
    print("moe_wna16_marlin_gemm signature:", inspect.signature(moe_wna16_marlin_gemm))
    try:
        print("source:\n", inspect.getsource(moe_wna16_marlin_gemm))
    except Exception as e:
        print("error getting source:", e)

    print("sgl_kernel marlin ops:", [x for x in dir(torch.ops.sgl_kernel) if 'marlin' in x.lower()])
    return data["hidden_states"]
