# okc: problem=muse_moe_w4a16
import torch
import inspect
import sys

def prepare(weights):
    print("=== PyTorch & CUDA ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

    print("\n=== FlashInfer MoE Modules ===")
    try:
        import flashinfer.fused_moe as fi_moe
        print("flashinfer.fused_moe dir:", [x for x in dir(fi_moe) if not x.startswith('_')])
        for fn_name in ['get_trtllm_moe_sm100_module', 'get_trtllm_moe_sm90_module', 'get_cutlass_moe_module']:
            if hasattr(fi_moe, fn_name):
                fn = getattr(fi_moe, fn_name)
                print(f"{fn_name}:", fn)
                try:
                    mod = fn()
                    print(f"  instantiated {fn_name}:", dir(mod))
                except Exception as e:
                    print(f"  error instantiating {fn_name}:", e)
    except Exception as e:
        print("error importing flashinfer.fused_moe:", e)

    print("\n=== SGLang MoE Kernels ===")
    try:
        import sglang.kernels.ops.moe as sgl_moe
        print("sglang moe dir:", dir(sgl_moe))
    except Exception as e:
        print("error importing sglang moe:", e)

    try:
        import sgl_kernel
        print("sgl_kernel dir:", [x for x in dir(sgl_kernel) if 'moe' in x.lower() or 'marlin' in x.lower() or 'gemm' in x.lower()])
    except Exception as e:
        print("error inspecting sgl_kernel:", e)

    try:
        import vllm
        print("vllm available! vllm version:", vllm.__version__)
    except Exception as e:
        print("vllm not available:", e)

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
