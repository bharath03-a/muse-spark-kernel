import torch
import flashinfer
from attn_reference import generate_input, check_implementation, ref_kernel

print("Testing FlashInfer BatchDecodeWithPagedKVCacheWrapper with FP8...")
device = "cuda"
data = generate_input(batch=16, ctx=80000, heads="tp8", layer="full", seed=42, page=1, device=device)
cfg = data["config"]

# Check wrapper capabilities
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
    "NHD",
    use_tensor_cores=True
)

k_cache = data["k_cache"]
v_cache = data["v_cache"]
print("k_cache shape:", k_cache.shape, "dtype:", k_cache.dtype)
print("page_table shape:", data["page_table"].shape)
print("seq_lens:", data["seq_lens"][:5])

# Let's see what wrapper.plan accepts
try:
    import inspect
    print("wrapper.plan signature:", inspect.signature(wrapper.plan))
    print("wrapper.run signature:", inspect.signature(wrapper.run))
except Exception as e:
    print("inspect error:", e)
