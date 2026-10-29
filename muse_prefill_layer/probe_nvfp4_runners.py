# okc: problem=muse_prefill_layer
import inspect
import torch

def prepare(weights):
    print("=== PROBING FLASHINFER NVFP4 RUNNERS ===")
    try:
        import flashinfer.fused_moe as fm
        for name in ['CutlassNvfp4Config', 'CutlassNvfp4Runner', 'cute_dsl_fused_moe_nvfp4', 'CuteDslMoEWrapper', 'CutlassHummingConfig', 'CutlassHummingRunner']:
            if hasattr(fm, name):
                obj = getattr(fm, name)
                print(f"\n--- {name} ---")
                print("type:", type(obj))
                if inspect.isclass(obj):
                    print("class doc:", inspect.getdoc(obj)[:300] if inspect.getdoc(obj) else "None")
                    print("methods:", [m for m in dir(obj) if not m.startswith('__')])
                    for m in ['supported', 'prepare_weights', 'run', 'forward']:
                        if hasattr(obj, m):
                            print(f"  {m} sig:", inspect.signature(getattr(obj, m)))
                elif callable(obj):
                    print("callable sig:", inspect.signature(obj))
                    print("doc:", inspect.getdoc(obj)[:300] if inspect.getdoc(obj) else "None")
    except Exception as e:
        print("Error inspecting fused_moe:", e)

    return {}

def kernel_fn(data, state):
    return data["stream"], data["stream"], data["stream"]
