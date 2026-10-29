# okc: problem=muse_moe_w4a16
import inspect
from sglang.kernels.ops.moe.moe_wna16_marlin import _jit_moe_wna16_marlin_module

def prepare(weights):
    print("=== _jit_moe_wna16_marlin_module ===")
    print("file:", inspect.getfile(_jit_moe_wna16_marlin_module))
    print("source:\n", inspect.getsource(_jit_moe_wna16_marlin_module))
    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
