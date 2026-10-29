# okc: problem=muse_prefill_layer
import torch
import types

def prepare(weights):
    device = weights["qkv_gate"].device
    return {
        "expert_out_buf": torch.zeros((16384, 8192), dtype=torch.bfloat16, device=device)
    }

def kernel_fn(data, state):
    try:
        import sgl_kernel
        print("sgl_kernel ops:", [x for x in dir(torch.ops.sgl_kernel) if any(k in x.lower() for k in ('moe', 'gemm', 'fp4', 'group', 'grouped'))])
    except Exception as e:
        print("sgl_kernel error:", e)

    try:
        import flashinfer
        print("flashinfer ops:", [x for x in dir(flashinfer) if any(k in x.lower() for k in ('moe', 'gemm', 'fp4', 'group', 'grouped'))])
        for mod in ('gemm', 'moe', 'fused_moe'):
            if hasattr(flashinfer, mod):
                print(f"flashinfer.{mod}:", dir(getattr(flashinfer, mod)))
    except Exception as e:
        print("flashinfer error:", e)

    try:
        import sglang.srt.layers.moe.fused_moe_triton as fmt
        print("fused_moe_triton dir:", [x for x in dir(fmt) if 'moe' in x.lower() or 'gemm' in x.lower()])
    except Exception as e:
        print("fused_moe_triton error:", e)

    try:
        import grouped_gemm
        print("grouped_gemm found!")
    except Exception as e:
        print("grouped_gemm error:", e)

    try:
        import cutlass
        print("cutlass found!")
    except Exception as e:
        print("cutlass error:", e)

    tokens = data["config"]["tokens"]
    return state["expert_out_buf"][:tokens], data["stream"], torch.zeros((tokens, 8), dtype=torch.int32, device=data["stream"].device)
