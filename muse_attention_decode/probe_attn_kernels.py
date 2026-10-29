# okc: problem=muse_attention_decode
import sys
import torch

def kernel_fn(data):
    print("=== PROBING ATTENTION KERNELS ===")
    
    # 1. Check sgl_kernel
    try:
        import sgl_kernel
        print("sgl_kernel attributes:", [x for x in dir(sgl_kernel) if not x.startswith('_')])
        if hasattr(sgl_kernel, 'ops'):
            print("sgl_kernel.ops:", [x for x in dir(sgl_kernel.ops) if not x.startswith('_')])
    except Exception as e:
        print("sgl_kernel error:", e)

    # 2. Check sglang.kernels
    try:
        import sglang.kernels
        print("sglang.kernels:", [x for x in dir(sglang.kernels) if not x.startswith('_')])
        import sglang.kernels.ops
        print("sglang.kernels.ops:", [x for x in dir(sglang.kernels.ops) if not x.startswith('_')])
    except Exception as e:
        print("sglang.kernels error:", e)

    # 3. Check flashinfer
    try:
        import flashinfer
        print("flashinfer version:", getattr(flashinfer, '__version__', 'unknown'))
        print("flashinfer attributes:", [x for x in dir(flashinfer) if 'decode' in x.lower() or 'paged' in x.lower() or 'attn' in x.lower()])
        if hasattr(flashinfer, 'decode'):
            print("flashinfer.decode:", [x for x in dir(flashinfer.decode) if not x.startswith('_')])
    except Exception as e:
        print("flashinfer error:", e)

    # 4. Check vllm
    try:
        import vllm
        print("vllm version:", getattr(vllm, '__version__', 'unknown'))
        from vllm import _custom_ops as ops
        print("vllm ops:", [x for x in dir(ops) if 'attn' in x.lower() or 'paged' in x.lower()])
    except Exception as e:
        print("vllm error/not installed:", e)

    # 5. Check torch SDPA or cutlass
    print("torch version:", torch.__version__)
    print("CUDA arch:", torch.cuda.get_device_capability())

    return data["qkv"][:, :, :16*64]
