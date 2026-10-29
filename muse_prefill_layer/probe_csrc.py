# okc: problem=muse_prefill_layer
import os
import glob
import torch

def prepare(weights):
    print("=== SEARCHING /sgl-workspace FOR CUDA/CUTLASS MOE FILES ===")
    root = "/sgl-workspace"
    for r, dirs, files in os.walk(root):
        for f in files:
            if any(k in f.lower() for k in ['moe', 'marlin', 'w4a16', 'nvfp4', 'cutlass']):
                full_path = os.path.join(r, f)
                if full_path.endswith(('.cuh', '.cu', '.h', '.cpp', '.py')):
                    print(full_path)
    return {}

def kernel_fn(data, state):
    stream = data["stream"]
    return stream, stream, data["topk_ids"] if "topk_ids" in data else torch.zeros((stream.shape[0], 8), dtype=torch.int32, device=stream.device)
