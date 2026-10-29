# okc: problem=muse_moe_w4a16
import torch
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CutlassW4A16Runner ===")
    src = inspect.getsource(fi_moe.CutlassW4A16Runner)
    # print first 8000 chars
    print(src[:8000])
    print("=== CutlassW4A16Config ===")
    src_cfg = inspect.getsource(fi_moe.CutlassW4A16Config)
    print(src_cfg[:4000])
    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
