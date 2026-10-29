# okc: problem=muse_attention_decode
"""Inspect data['scheduler_metadata'] for scored cases."""

import torch

def kernel_fn(data):
    cfg = data["config"]
    sched = data.get("scheduler_metadata")
    if sched is not None:
        print(f"B={cfg['batch']}, L={cfg['q_len']}, layer={cfg['window']}, sched={sched.shape}, num_splits={sched[0].item() if sched.numel() > 0 else 'empty'}")
        if sched.numel() > 10:
            print("  first 10 elements:", sched[:10].tolist())
    else:
        print(f"B={cfg['batch']}, L={cfg['q_len']}, layer={cfg['window']}, sched=None")
        
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    return torch.zeros((cfg["batch"], cfg["q_len"], hq * d), dtype=torch.bfloat16, device="cuda")
