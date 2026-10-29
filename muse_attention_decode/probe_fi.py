# okc: problem=muse_attention_decode
import torch
import inspect
import flashinfer

def kernel_fn(data):
    try:
        wrapper_cls = flashinfer.BatchDecodeWithPagedKVCacheWrapper
        print("wrapper_cls:", wrapper_cls)
        print("wrapper init sig:", inspect.signature(wrapper_cls.__init__))
        print("wrapper plan sig:", inspect.signature(wrapper_cls.plan))
        print("wrapper forward sig:", inspect.signature(wrapper_cls.forward))
        if hasattr(flashinfer, 'CUDAGraphBatchDecodeWithPagedKVCacheWrapper'):
            cg_cls = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper
            print("cg_cls init sig:", inspect.signature(cg_cls.__init__))
            print("cg_cls plan sig:", inspect.signature(cg_cls.plan))
            print("cg_cls forward sig:", inspect.signature(cg_cls.forward))
    except Exception as e:
        print("probe error:", e)

    return data["qkv"][:, :, :16*64]
