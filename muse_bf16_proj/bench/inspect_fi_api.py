import inspect
import flashinfer

print("fused_qk_rmsnorm_rope doc/signature:")
try:
    print(inspect.signature(flashinfer.fused_qk_rmsnorm_rope))
except Exception as e:
    print(e)

print("BatchDecodeWithPagedKVCacheWrapper methods:")
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper
print([m for m in dir(wrapper) if not m.startswith("_")])

print("CUDAGraphBatchDecodeWithPagedKVCacheWrapper methods:")
wrapper_cg = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper
print([m for m in dir(wrapper_cg) if not m.startswith("_")])
