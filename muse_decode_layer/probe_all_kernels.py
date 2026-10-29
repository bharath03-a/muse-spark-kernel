# okc: problem=muse_decode_layer
"""Probe FlashInfer W4A16 and decode attention runners."""

import inspect
import torch
import baseline

def prepare(weights):
    state = baseline.prepare(weights)
    print("=== PROBING FLASHINFER RUNNERS ===")
    
    # 1. FlashInfer MoE W4A16
    try:
        from flashinfer.fused_moe import CutlassW4A16Runner, B12xW4A16Runner, CutlassW4A16Config, B12xW4A16Config
        print("CutlassW4A16Runner:", CutlassW4A16Runner)
        print("CutlassW4A16Runner init:", inspect.signature(CutlassW4A16Runner.__init__))
        print("CutlassW4A16Runner forward:", inspect.signature(CutlassW4A16Runner.forward))
        if hasattr(CutlassW4A16Runner, "supported_quant_variants"):
            print("CutlassW4A16Runner quant variants:", CutlassW4A16Runner.supported_quant_variants)
        if hasattr(CutlassW4A16Runner, "_required_weight_keys"):
            print("CutlassW4A16Runner required keys:", CutlassW4A16Runner._required_weight_keys)

        print("B12xW4A16Runner:", B12xW4A16Runner)
        print("B12xW4A16Runner init:", inspect.signature(B12xW4A16Runner.__init__))
        print("B12xW4A16Runner forward:", inspect.signature(B12xW4A16Runner.forward))
        if hasattr(B12xW4A16Runner, "supported_quant_variants"):
            print("B12xW4A16Runner quant variants:", B12xW4A16Runner.supported_quant_variants)
    except Exception as e:
        print("FlashInfer MoE error:", e)

    # 2. FlashInfer Attention Decode
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache, xqa_batch_decode_with_kv_cache, cudnn_batch_decode_with_kv_cache
        print("trtllm_batch_decode_with_kv_cache:", inspect.signature(trtllm_batch_decode_with_kv_cache))
        print("xqa_batch_decode_with_kv_cache:", inspect.signature(xqa_batch_decode_with_kv_cache))
        print("cudnn_batch_decode_with_kv_cache:", inspect.signature(cudnn_batch_decode_with_kv_cache))
    except Exception as e:
        print("FlashInfer Decode error:", e)

    return state

def kernel_fn(data, state):
    return baseline.kernel_fn(data, state)
