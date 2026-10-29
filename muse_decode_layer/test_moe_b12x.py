# okc: problem=muse_decode_layer
"""Probe MoEConfig, B12xRunner pack_inputs, and trtllm_batch_decode_with_kv_cache."""

import inspect
import torch
import baseline

def prepare(weights):
    state = baseline.prepare(weights)
    device = weights["qkv_gate"].device

    print("=== FMC CONFIG MEMBERS ===")
    try:
        import flashinfer.fused_moe.config as fmc
        for name in dir(fmc):
            if not name.startswith("_"):
                obj = getattr(fmc, name)
                if inspect.isclass(obj):
                    try:
                        print(f"CLASS {name}: {inspect.signature(obj.__init__)}")
                    except Exception:
                        print(f"CLASS {name}")
    except Exception as e:
        print("fmc error:", e)

    print("=== B12x PACK_INPUTS & BUILD ===")
    try:
        from flashinfer.fused_moe.runners import B12xW4A16Runner
        print("pack_inputs sig:", inspect.signature(B12xW4A16Runner.pack_inputs))
        print("pack_inputs source:")
        print(inspect.getsource(B12xW4A16Runner.pack_inputs))
        print("build source:")
        print(inspect.getsource(B12xW4A16Runner._build))
    except Exception as e:
        print("B12x pack error:", e)

    print("=== TRTLLM DECODE TEST ===")
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        print("trtllm doc/sig:", inspect.signature(trtllm_batch_decode_with_kv_cache))
    except Exception as e:
        print("trtllm error:", e)

    return state

def kernel_fn(data, state):
    return baseline.kernel_fn(data, state)
