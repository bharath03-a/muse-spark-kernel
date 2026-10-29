import inspect
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

print("=== SGLang _bf16_gemm_dispatch_impl ===")
print(inspect.getsource(_bf16_gemm_dispatch_impl))
