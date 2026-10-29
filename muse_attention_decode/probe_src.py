# okc: problem=muse_attention_decode
import inspect
from sgl_kernel.flash_attn import get_scheduler_metadata

def kernel_fn(data):
    try:
        src = inspect.getsource(get_scheduler_metadata)
        print("get_scheduler_metadata source:\n", src)
    except Exception as e:
        print("error getting source:", e)
    return data["qkv"][:, :, :16*64]
