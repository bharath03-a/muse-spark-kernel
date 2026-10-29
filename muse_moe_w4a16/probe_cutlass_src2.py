# okc: problem=muse_moe_w4a16
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== cutlass_fused_moe source lines 60-140 ===")
    try:
        lines = inspect.getsourcelines(fi_moe.cutlass_fused_moe)[0]
        print("".join(lines[60:140]))
    except Exception as e:
        print("error:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
