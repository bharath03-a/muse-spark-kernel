# okc: problem=muse_moe_w4a16
import inspect
import torch
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    return {}

def kernel_fn(data, state):
    print("=== TrtllmFp4RoutedRunner Source ===")
    try:
        print(inspect.getsource(fi_moe.TrtllmFp4RoutedRunner))
    except Exception as e:
        print("error getting TrtllmFp4RoutedRunner source:", e)

    print("=== B12xNvfp4Runner Source ===")
    try:
        print(inspect.getsource(fi_moe.B12xNvfp4Runner))
    except Exception as e:
        print("error getting B12xNvfp4Runner source:", e)

    return data["hidden_states"]
