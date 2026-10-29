# okc: problem=muse_attention_decode
import torch
from sgl_kernel.flash_attn import get_scheduler_metadata

def kernel_fn(data):
    cfg = data["config"]
    sched = data.get("scheduler_metadata")
    if sched is not None:
        print(f"Case batch={cfg['batch']}, layer={cfg['window']}, sched shape={sched.shape}, dtype={sched.dtype}, sample={sched[:10].tolist()}")
    else:
        print(f"Case batch={cfg['batch']}, layer={cfg['window']}, sched=None")
    return data["qkv"][:, :, :16*64]
