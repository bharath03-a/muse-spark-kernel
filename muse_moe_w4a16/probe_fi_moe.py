# okc: problem=muse_moe_w4a16
import inspect
import torch

def prepare(weights):
    return {}

def kernel_fn(data, state):
    import flashinfer.fused_moe as fi_moe
    for name in ('CutlassNvfp4Runner', 'TrtllmFp4RoutedRunner', 'cutlass_fused_moe', 'trtllm_fp4_block_scale_routed_moe', 'prims_ts_fp4_block_scale_routed_moe', 'B12xNvfp4Runner'):
        if hasattr(fi_moe, name):
            obj = getattr(fi_moe, name)
            print(f"=== {name} ===")
            if inspect.isclass(obj):
                print("init:", inspect.signature(obj.__init__))
                for m in ('run', 'forward', 'execute'):
                    if hasattr(obj, m):
                        print(f"{m}:", inspect.signature(getattr(obj, m)))
            elif inspect.isfunction(obj):
                print("sig:", inspect.signature(obj))
    if hasattr(flashinfer, 'grouped_mm_fp4'):
        print("grouped_mm_fp4:", inspect.signature(flashinfer.grouped_mm_fp4))
    return data["hidden_states"]
