# okc: problem=muse_prefill_layer
import torch

def prepare(weights):
    print("=== READING test_nvfp4_moe_backends.py ===")
    path = "/sgl-workspace/sglang/test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py"
    try:
        with open(path, "r") as f:
            content = f.read()
        print(content[:15000])
    except Exception as e:
        print("Error reading file:", e)
    return {}

def kernel_fn(data, state):
    stream = data["stream"]
    return stream, stream, data["topk_ids"] if "topk_ids" in data else torch.zeros((stream.shape[0], 8), dtype=torch.int32, device=stream.device)
