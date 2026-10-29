"""Call before importing torch/SGLang on BOTH CPU and H200 Modal workers."""
import os


def configure_h200_cache(root='/cache'):
    values = dict(TORCH_EXTENSIONS_DIR=root + '/torch_ext',
                  TRITON_CACHE_DIR=root + '/triton',
                  SGLANG_JIT_CACHE_DIR=root + '/sglang_jit',
                  TVM_FFI_CACHE_DIR=root + '/tvm-ffi',
                  TORCH_CUDA_ARCH_LIST='9.0a', MAX_JOBS='2')
    os.environ.update(values)
    return values
