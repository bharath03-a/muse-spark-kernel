# okc: problem=muse_prefill_layer
import os
import torch

def prepare(weights):
    import sys
    print("=== READING moe_wna16_marlin.cuh lines 340-550 ===", flush=True)
    candidates = [
        "/sgl-workspace/sglang-muse-spark/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh",
        "/sgl-workspace/sglang/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                lines = f.readlines()
                print("".join(lines[340:550]), flush=True)
            break
    sys.stdout.flush()
    return {}

def kernel_fn(data, state):
    stream = data["stream"]
    hidden = data["attn_input"]
    topk_ids = data["topk_ids"] if "topk_ids" in data else torch.zeros((stream.shape[0], 8), dtype=torch.int32, device=stream.device)
    return hidden, stream, topk_ids
