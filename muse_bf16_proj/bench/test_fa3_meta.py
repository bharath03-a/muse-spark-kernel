import inspect
from sgl_kernel.flash_attn import get_scheduler_metadata

print("get_scheduler_metadata signature:")
try:
    print(inspect.signature(get_scheduler_metadata))
except Exception as e:
    print(e)
