import torch
import flashinfer
import sgl_kernel
import sglang

print("FlashInfer version:", flashinfer.__version__)
print("FlashInfer dir:", [x for x in dir(flashinfer) if not x.startswith("_")])

# Check BatchDecodeWithPagedKVCacheWrapper or similar
try:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    print("Found BatchDecodeWithPagedKVCacheWrapper")
except Exception as e:
    print("BatchDecodeWithPagedKVCacheWrapper error:", e)

try:
    import flashinfer.page
    print("Found flashinfer.page")
except Exception as e:
    print("flashinfer.page error:", e)

# Check FA3 functions
from sgl_kernel.flash_attn import flash_attn_with_kvcache
print("flash_attn_with_kvcache:", flash_attn_with_kvcache)
