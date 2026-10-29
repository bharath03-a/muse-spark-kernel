import inspect
import flashinfer

print("CUDAGraphBatchDecodeWithPagedKVCacheWrapper:")
cg_wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper
print("plan signature:", inspect.signature(cg_wrapper.plan))
print("run signature:", inspect.signature(cg_wrapper.run))
print("begin_forward signature:", inspect.signature(cg_wrapper.begin_forward))
print("doc:", cg_wrapper.__doc__)
