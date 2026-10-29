# okc: problem=muse_prefill_layer
"""Probe SM90 MoE backends and CUTLASS / CuTe options on H200."""
import sys
import inspect
import torch

def prepare(weights):
    print("=== PROBING SM90 MOE BACKENDS & CUTLASS RUNNERS ===", flush=True)

    # 1. Inspect test_modelopt_nvfp4_moe_dispatch.py
    for path in [
        "/sgl-workspace/sglang/test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_dispatch.py",
        "/sgl-workspace/sglang/test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py",
        "/sgl-workspace/sglang/test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py",
    ]:
        try:
            with open(path) as f:
                content = f.read()
            print(f"\n--- {path} --- ({len(content)} chars)", flush=True)
            print(content[:2000], flush=True)
        except Exception as e:
            print(f"Failed to read {path}: {e}", flush=True)

    # 2. Check FlashInfer runners on SM90
    import flashinfer.fused_moe as fm
    for name in ['CutlassHummingConfig', 'CutlassNvfp4Config', 'TrtllmBf16Config', 'CutlassBf16Config']:
        if hasattr(fm, name):
            cfg = getattr(fm, name)
            if hasattr(cfg, 'supported'):
                try:
                    print(f"{name}.supported(90): {cfg.supported(90)}", flush=True)
                except Exception as e:
                    print(f"{name}.supported(90) error: {e}", flush=True)

    # 3. Check what runners MoeRunnerBackend supports
    try:
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        print("\nMoeRunnerBackend enum values:", [b.value for b in MoeRunnerBackend], flush=True)
    except Exception as e:
        print("MoeRunnerBackend error:", e, flush=True)

    # 4. Check modelopt_nvfp4 quantization method
    try:
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
        cfg = ModelOptFp4Config()
        print("\nModelOptFp4Config quant_method:", cfg.get_quant_method(None, None), flush=True)
    except Exception as e:
        print("ModelOptFp4Config error:", e, flush=True)

    sys.stdout.flush()
    return {}

def kernel_fn(data, state):
    stream = data["stream"]
    hidden = data["attn_input"]
    T = stream.shape[0]
    topk_ids = torch.arange(8, dtype=torch.int32, device=stream.device).unsqueeze(0).expand(T, 8).contiguous()
    return hidden, stream, topk_ids
