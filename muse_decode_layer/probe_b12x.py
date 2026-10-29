# okc: problem=muse_decode_layer
"""Probe source code of B12xW4A16Runner and XQA decode."""

import inspect
import torch
import baseline

def prepare(weights):
    state = baseline.prepare(weights)
    print("=== B12xW4A16Runner SOURCE ===")
    try:
        from flashinfer.fused_moe.runners import B12xW4A16Runner
        print(inspect.getsource(B12xW4A16Runner))
        # Also print base class
        for base in B12xW4A16Runner.__mro__[1:3]:
            print(f"=== BASE {base.__name__} ===")
            print(inspect.getsource(base))
    except Exception as e:
        print("B12x error:", e)

    print("=== XQA SOURCE ===")
    try:
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        print(inspect.getsource(xqa_batch_decode_with_kv_cache))
    except Exception as e:
        print("XQA error:", e)

    return state

def kernel_fn(data, state):
    return baseline.kernel_fn(data, state)
