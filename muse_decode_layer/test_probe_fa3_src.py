# okc: problem=muse_decode_layer
import inspect
import baseline
from sgl_kernel.flash_attn import flash_attn_with_kvcache

def prepare(weights):
    state = baseline.prepare(weights)
    print("=== FA3 SOURCE BODY ===")
    try:
        src = inspect.getsource(flash_attn_with_kvcache)
        # Find def flash_attn_with_kvcache body after docstring
        doc_end = src.find('"""\n', src.find('"""') + 3)
        print("BODY:\n", src[doc_end + 4:doc_end + 3500])
    except Exception as e:
        print("FA3 source error:", e)
    return state

def kernel_fn(data, state):
    return baseline.kernel_fn(data, state)
