# okc: problem=muse_moe_w4a16
import inspect
import torch
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    return {}

def kernel_fn(data, state):
    fn = getattr(fi_moe, "trtllm_fp4_block_scale_routed_moe", None)
    if fn:
        print("=== trtllm_fp4_block_scale_routed_moe ===")
        print("doc:", fn.__doc__)
        try:
            print("source:", inspect.getsource(fn))
        except Exception as e:
            print("source error:", e)
    return data["hidden_states"]
