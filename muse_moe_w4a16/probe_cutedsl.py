# okc: problem=muse_moe_w4a16
import torch
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CuteDslRunner ===")
    try:
        src = inspect.getsource(fi_moe.CuteDslRunner)
        print(src[:6000])
    except Exception as e:
        print("error CuteDslRunner:", e)

    print("=== CuteDslConfig ===")
    try:
        src_cfg = inspect.getsource(fi_moe.CuteDslConfig)
        print(src_cfg[:3000])
    except Exception as e:
        print("error CuteDslConfig:", e)

    print("=== CuteDsl functions ===")
    for fn in ['cute_dsl_fused_moe', 'cute_dsl_fused_moe_nvfp4', 'b12x_fused_moe']:
        if hasattr(fi_moe, fn):
            f = getattr(fi_moe, fn)
            print(f"{fn} sig:", inspect.signature(f))

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
