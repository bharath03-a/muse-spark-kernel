import sglang.srt.layers.quantization.unquant as uq

print("_enable_bf16_splitk_gemm:", uq._enable_bf16_splitk_gemm)
print("_use_hopper_bf16_gemv:", uq._use_hopper_bf16_gemv)
print("_use_cutedsl_bf16_gemm:", uq._use_cutedsl_bf16_gemm)
if uq._use_hopper_bf16_gemv:
    print("hopper gemv fn:", uq._hopper_bf16_gemv)
if uq._use_cutedsl_bf16_gemm:
    print("cutedsl gemm fn:", uq._cutedsl_bf16_gemm)
