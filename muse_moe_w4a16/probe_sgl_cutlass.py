# okc: problem=muse_moe_w4a16
import torch
import inspect
import sgl_kernel

def prepare(weights):
    print("=== sgl_kernel.cutlass_moe ===")
    try:
        print("cutlass_moe type:", type(sgl_kernel.cutlass_moe))
        print("cutlass_moe dir:", dir(sgl_kernel.cutlass_moe))
        if callable(sgl_kernel.cutlass_moe):
            print("cutlass_moe sig:", inspect.signature(sgl_kernel.cutlass_moe))
    except Exception as e:
        print("error cutlass_moe:", e)

    print("=== sgl_kernel.moe ===")
    try:
        print("moe type:", type(sgl_kernel.moe))
        print("moe dir:", dir(sgl_kernel.moe))
        if callable(sgl_kernel.moe):
            print("moe sig:", inspect.signature(sgl_kernel.moe))
    except Exception as e:
        print("error moe:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
