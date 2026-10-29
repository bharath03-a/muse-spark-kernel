# okc: problem=muse_attention_decode
import torch
import sys

def kernel_fn(data):
    try:
        import flashinfer
        print(f"FlashInfer version: {flashinfer.__version__}")
        print(f"FlashInfer dir: {[x for x in dir(flashinfer) if 'decode' in x.lower() or 'attention' in x.lower() or 'batch' in x.lower()]}")
        if hasattr(flashinfer, 'decode'):
            print(f"flashinfer.decode dir: {dir(flashinfer.decode)}")
    except Exception as e:
        print(f"FlashInfer error: {e}")

    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except Exception as e:
        print(f"vLLM error: {e}")

    return data["qkv"][:, :, :16*64]
