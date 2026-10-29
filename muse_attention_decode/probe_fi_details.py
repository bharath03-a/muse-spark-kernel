# okc: problem=muse_attention_decode
import inspect
import torch
import flashinfer

def kernel_fn(data):
    cfg = data["config"]
    device = data["qkv"].device
    print("=== PROBING FLASHINFER DETAILS ===")
    
    # Check BatchDecodeWithPagedKVCacheWrapper
    try:
        ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", use_tensor_cores=True
        )
        print("wrapper init success!")
        print("plan doc/sig:", inspect.signature(wrapper.plan))
        print("forward doc/sig:", inspect.signature(wrapper.forward))
    except Exception as e:
        print("wrapper init error:", e)

    # Check CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    try:
        cg_wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", use_tensor_cores=True
        )
        print("cg_wrapper init success!")
        print("cg_wrapper plan sig:", inspect.signature(cg_wrapper.plan))
        print("cg_wrapper forward sig:", inspect.signature(cg_wrapper.forward))
    except Exception as e:
        print("cg_wrapper init error:", e)

    # Check TRT-LLM / XQA decode
    try:
        print("flashinfer.decode.trtllm_batch_decode_with_kv_cache:", 
              hasattr(flashinfer.decode, 'trtllm_batch_decode_with_kv_cache'))
        if hasattr(flashinfer.decode, 'trtllm_batch_decode_with_kv_cache'):
            print("trtllm sig:", inspect.signature(flashinfer.decode.trtllm_batch_decode_with_kv_cache))
    except Exception as e:
        print("trtllm error:", e)

    # Check sgl_kernel.attention
    try:
        import sgl_kernel
        if hasattr(sgl_kernel, 'attention'):
            print("sgl_kernel.attention:", [x for x in dir(sgl_kernel.attention) if not x.startswith('_')])
    except Exception as e:
        print("sgl_kernel.attention error:", e)

    # Check sglang attention ops
    try:
        from sglang.kernels.ops import attention
        print("sglang.kernels.ops.attention:", [x for x in dir(attention) if not x.startswith('_')])
    except Exception as e:
        print("sglang attention ops error:", e)

    return data["qkv"][:, :, :16*64]
