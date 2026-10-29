# okc: problem=muse_attention_decode
import inspect
import flashinfer
import torch

cls = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper
print("=== CUDAGRAPH INIT ===")
print(inspect.getsource(cls.__init__))
print("=== CUDAGRAPH BEGIN_FORWARD ===")
print(inspect.getsource(cls.begin_forward))
if hasattr(cls, "plan"):
    print("=== CUDAGRAPH PLAN ===")
    print(inspect.getsource(cls.plan))
if hasattr(cls, "forward"):
    print("=== CUDAGRAPH FORWARD ===")
    print(inspect.getsource(cls.forward))

def kernel_fn(data):
    cfg = data["config"]
    shape = (cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * cfg["head_dim"])
    return torch.zeros(shape, dtype=torch.bfloat16, device=data["qkv"].device)
