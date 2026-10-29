# okc: problem=muse_moe_w4a16
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== CutlassW4A16Config and CutlassW4A16Runner ===")
    try:
        print("CutlassW4A16Config:")
        print(inspect.getsource(fi_moe.CutlassW4A16Config))
    except Exception as e:
        print("CutlassW4A16Config error:", e)

    try:
        print("CutlassW4A16Runner:")
        print(inspect.getsource(fi_moe.CutlassW4A16Runner))
    except Exception as e:
        print("CutlassW4A16Runner error:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
