import inspect
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

print("=== FusedQKNorm Source ===")
print(inspect.getsource(FusedQKNorm))

print("=== rms_norm_gate Source ===")
print(inspect.getsource(rms_norm_gate))
