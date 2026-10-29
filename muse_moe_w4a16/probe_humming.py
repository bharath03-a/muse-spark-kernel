# okc: problem=muse_moe_w4a16
import torch
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CutlassHummingRunner ===")
    try:
        src = inspect.getsource(fi_moe.CutlassHummingRunner)
        print(src[:6000])
    except Exception as e:
        print("error HummingRunner:", e)

    print("=== CutlassHummingConfig ===")
    try:
        src_cfg = inspect.getsource(fi_moe.CutlassHummingConfig)
        print(src_cfg[:3000])
    except Exception as e:
        print("error HummingConfig:", e)

    print("=== CuTileNvfp4Runner ===")
    try:
        src_cutile = inspect.getsource(fi_moe.CuTileNvfp4Runner)
        print(src_cutile[:4000])
    except Exception as e:
        print("error CuTileNvfp4Runner:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
