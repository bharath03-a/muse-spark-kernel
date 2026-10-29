# okc: problem=muse_moe_w4a16
import torch
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CutlassW4A16Runner Source ===")
    try:
        print(inspect.getsource(fi_moe.CutlassW4A16Runner))
    except Exception as e:
        print("error CutlassW4A16Runner:", e)

    print("\n=== CutlassW4A16Config Source ===")
    try:
        print(inspect.getsource(fi_moe.CutlassW4A16Config))
    except Exception as e:
        print("error CutlassW4A16Config:", e)

    print("\n=== gen_cutlass_fused_moe_sm90_module ===")
    try:
        mod = fi_moe.gen_cutlass_fused_moe_sm90_module()
        print("sm90 module:", dir(mod))
        for attr in dir(mod):
            if not attr.startswith('__'):
                val = getattr(mod, attr)
                print(f"  {attr}: {val}")
    except Exception as e:
        print("error sm90 module:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
