# okc: problem=muse_prefill_layer
import inspect
import torch
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    return {}

def kernel_fn(data, state):
    print("CutlassW4A16Runner:")
    try:
        print("source:", inspect.getsource(fi_moe.CutlassW4A16Runner))
    except Exception as e:
        print("error:", e)

    print("prims_ts_fp4_block_scale_routed_moe:")
    try:
        print("source:", inspect.getsource(fi_moe.prims_ts_fp4_block_scale_routed_moe))
    except Exception as e:
        print("error:", e)

    tokens = data["config"]["tokens"]
    return torch.zeros((tokens, 8192), dtype=torch.bfloat16, device=data["stream"].device), data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=data["stream"].device)
