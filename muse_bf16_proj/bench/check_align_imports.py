import inspect
import sglang.srt.layers.moe.fused_moe_triton as fmt

print("sgl_moe_align_block_size:", getattr(fmt, "sgl_moe_align_block_size", None))
print("moe_align_small_numel:", getattr(fmt, "moe_align_small_numel", None))
print("SMALL_NUMEL_LIMIT:", getattr(fmt, "SMALL_NUMEL_LIMIT", None))
print("_CUDA_SMALL_BATCH_MAX_BUCKETS:", getattr(fmt, "_CUDA_SMALL_BATCH_MAX_BUCKETS", None))
