# okc: problem=muse_attention_decode
import inspect
import flashinfer
import torch

print("=== FLASHINFER DECODE INSPECTION ===")
print("FlashInfer version:", getattr(flashinfer, "__version__", "unknown"))

wrapper_cls = getattr(flashinfer, "CUDAGraphBatchDecodeWithPagedKVCacheWrapper", None)
print("CUDAGraphBatchDecodeWithPagedKVCacheWrapper exists:", wrapper_cls is not None)

if wrapper_cls:
    print("Init doc/sig:", inspect.signature(wrapper_cls.__init__))
    print("Methods:", [m for m in dir(wrapper_cls) if not m.startswith("_")])
    for method_name in ["plan", "begin_forward", "end_forward", "forward", "run"]:
        if hasattr(wrapper_cls, method_name):
            print(f"Method {method_name} sig:", inspect.signature(getattr(wrapper_cls, method_name)))

base_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
print("BatchDecodeWithPagedKVCacheWrapper exists:", base_cls is not None)
if base_cls:
    print("Base Init doc/sig:", inspect.signature(base_cls.__init__))
    print("Base Methods:", [m for m in dir(base_cls) if not m.startswith("_")])
    for method_name in ["plan", "begin_forward", "end_forward", "forward", "run"]:
        if hasattr(base_cls, method_name):
            print(f"Base Method {method_name} sig:", inspect.signature(getattr(base_cls, method_name)))

# Fallback dummy kernel_fn so OKC check doesn't crash before printing
from submission import kernel_fn as orig_kernel_fn

def kernel_fn(data):
    return orig_kernel_fn(data)

