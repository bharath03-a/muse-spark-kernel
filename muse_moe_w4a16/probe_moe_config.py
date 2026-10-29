# okc: problem=muse_moe_w4a16
import inspect
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    return {}

def kernel_fn(data, state):
    print("MoEConfig source:")
    try:
        print(inspect.getsource(fi_moe.MoEConfig))
    except Exception as e:
        print("error:", e)
    print("TrtllmFp4RoutedRunner source:")
    try:
        print(inspect.getsource(fi_moe.TrtllmFp4RoutedRunner))
    except Exception as e:
        print("error:", e)
    return data["hidden_states"]
