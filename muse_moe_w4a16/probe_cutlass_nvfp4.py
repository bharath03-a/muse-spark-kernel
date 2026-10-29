# okc: problem=muse_moe_w4a16
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CutlassNvfp4Config and CutlassNvfp4Runner ===")
    try:
        print("CutlassNvfp4Config:")
        print(inspect.getsource(fi_moe.CutlassNvfp4Config))
    except Exception as e:
        print("CutlassNvfp4Config error:", e)

    try:
        print("CutlassNvfp4Runner:")
        print(inspect.getsource(fi_moe.CutlassNvfp4Runner))
    except Exception as e:
        print("CutlassNvfp4Runner error:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
