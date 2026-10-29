# okc: problem=muse_moe_w4a16
"""Testing direct preallocated Marlin execution for routed experts on H200."""

import types
from pathlib import Path
import tempfile
MARLIN_CUDA_SOURCE = '/*\n * Modified by Neural Magic\n * Copyright (C) Marlin.2024 Elias Frantar\n *\n * Licensed under the Apache License, Version 2.0 (the "License");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *         http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an "AS IS" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\n/*\n * Adapted from https://github.com/IST-DASLab/marlin\n */\n\n#pragma once\n\n#include <sgl_kernel/tensor.h>\n\n#include <sgl_kernel/scalar_type.hpp>\n\n#include <gemm/marlin_moe/kernel.h>\n#include <gemm/marlin_moe/marlin_template.h>\n\nnamespace sglang {\n\nnamespace device::marlin_moe {\n\n__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};\n\nusing MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800\n\ntemplate <int moe_block_size>\n__global__ void permute_cols_kernel(\n    int4 const* __restrict__ a_int4_ptr,\n    int const* __restrict__ perm_int_ptr,\n    int4* __restrict__ out_int4_ptr,\n    const int32_t* __restrict__ sorted_token_ids_ptr,\n    const int32_t* __restrict__ expert_ids_ptr,\n    const int32_t* __restrict__ num_tokens_past_padded_ptr,\n    int size_m,\n    int size_k,\n    int top_k) {};\n\n#else\n\n// For a given "a" of size [M,K] performs a permutation of the K columns based\n// on the given "perm" indices.\ntemplate <int moe_block_size>\n__global__ void permute_cols_kernel(\n    int4 const* __restrict__ a_int4_ptr,\n    int const* __restrict__ perm_int_ptr,\n    int4* __restrict__ out_int4_ptr,\n    const int32_t* __restrict__ sorted_token_ids_ptr,\n    const int32_t* __restrict__ expert_ids_ptr,\n    const int32_t* __restrict__ num_tokens_past_padded_ptr,\n    int size_m,\n    int size_k,\n    int top_k) {\n  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];\n  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);\n  int32_t block_sorted_ids[moe_block_size];\n  int block_num_valid_tokens = 0;\n  int64_t old_expert_id = 0;\n  int64_t expert_id = 0;\n  int row_stride = size_k * sizeof(half) / 16;\n\n  auto read_moe_block_data = [&](int block_id) {\n    block_num_valid_tokens = moe_block_size;\n    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);\n    for (int i = 0; i < moe_block_size / 4; i++) {\n      tmp_block_sorted_ids[i] = ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];\n    }\n    for (int i = 0; i < moe_block_size; i++) {\n      if (block_sorted_ids[i] >= size_m * top_k) {\n        block_num_valid_tokens = i;\n        break;\n      };\n    }\n  };\n\n  auto permute_row = [&](int row) {\n    int iters = size_k / default_threads;\n    int rest = size_k % default_threads;\n\n    int in_offset = (row / top_k) * row_stride;\n    int out_offset = row * row_stride;\n\n    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + in_offset);\n    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);\n\n    int base_k = 0;\n\n    for (int i = 0; i < iters; i++) {\n      auto cur_k = base_k + threadIdx.x;\n      int src_pos = perm_int_ptr[cur_k];\n\n      out_half[cur_k] = a_row_half[src_pos];\n\n      base_k += default_threads;\n    }\n\n    if (rest) {\n      if (threadIdx.x < rest) {\n        auto cur_k = base_k + threadIdx.x;\n        int src_pos = perm_int_ptr[cur_k];\n\n        out_half[cur_k] = a_row_half[src_pos];\n      }\n    }\n  };\n\n  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {\n    old_expert_id = expert_id;\n    int tmp_expert_id = expert_ids_ptr[index];\n    if (tmp_expert_id == -1) continue;\n    expert_id = tmp_expert_id;\n    perm_int_ptr += (expert_id - old_expert_id) * size_k;\n    read_moe_block_data(index);\n\n    for (int i = 0; i < block_num_valid_tokens; i++)\n      permute_row(block_sorted_ids[i]);\n  }\n}\n\ntypedef struct {\n  int thread_k;\n  int thread_n;\n  int num_threads;\n} thread_config_t;\n\nthread_config_t small_batch_thread_configs[] = {\n    // Ordered by priority\n\n    // thread_k, thread_n, num_threads\n    {128, 128, 256},\n    {64, 128, 128}};\n\nthread_config_t large_batch_thread_configs[] = {\n    // Ordered by priority\n\n    // thread_k, thread_n, num_threads\n    {64, 256, 256},\n    {64, 128, 128}};\n\ntypedef struct {\n  int blocks_per_sm;\n  thread_config_t tb_cfg;\n} exec_config_t;\n\nconstexpr int kSharedMemoryValidityMargin = 512;\nconstexpr int kSharedMemoryLaunchReserve = 1024;\n\nint get_scales_cache_size(\n    thread_config_t const& th_config,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    int num_bits,\n    int group_size,\n    bool has_act_order,\n    bool is_k_full) {\n  bool cache_scales_chunk = has_act_order && !is_k_full;\n\n  int tb_n = th_config.thread_n;\n  int tb_k = th_config.thread_k;\n\n  // Get max scale groups per thread-block\n  int tb_groups;\n  if (group_size == -1) {\n    tb_groups = 1;\n  } else if (group_size == 0) {\n    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size\n  } else {\n    tb_groups = div_ceil(tb_k, group_size);\n  }\n\n  if (cache_scales_chunk) {\n    int load_groups = tb_groups * pipe_stages * 2;  // Chunk size is 2x pipeline over dim K\n    load_groups = max(load_groups, 32);             // We load at least 32 scale groups\n    return load_groups * tb_n * 2;\n  } else {\n    int tb_scales = tb_groups * tb_n * 2;\n\n    return tb_scales * pipe_stages;\n  }\n}\n\nint get_kernel_cache_size(\n    thread_config_t const& th_config,\n    bool m_block_size_8,\n    int thread_m_blocks,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    int num_bits,\n    int group_size,\n    bool has_act_order,\n    bool is_k_full,\n    int has_zp,\n    int is_zp_float) {\n  int pack_factor = 32 / num_bits;\n\n  // Get B size\n  int tb_k = th_config.thread_k;\n  int tb_n = th_config.thread_n;\n  int tb_m = thread_m_blocks * 16;\n\n  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights\n  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)\n  int sh_block_meta_size = tb_m * 4;\n  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;\n  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;\n  int sh_red_size = tb_m * (tb_n + 8) * 2;\n  int sh_bias_size = tb_n * 2;\n  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;\n  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);\n\n  int sh_s_size =\n      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);\n  int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;\n  int sh_zp_size = 0;\n  if (has_zp) {\n    if (is_zp_float)\n      sh_zp_size = sh_s_size;\n    else if (num_bits == 4)\n      sh_zp_size = sh_s_size / 4;\n    else if (num_bits == 8)\n      sh_zp_size = sh_s_size / 2;\n  }\n\n  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size;\n\n  return total_size;\n}\n\nbool is_valid_config(\n    thread_config_t const& th_config,\n    bool m_block_size_8,\n    int thread_m_blocks,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    int num_bits,\n    int group_size,\n    bool has_act_order,\n    bool is_k_full,\n    int has_zp,\n    int is_zp_float,\n    int max_shared_mem) {\n  // Sanity\n  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {\n    return false;\n  }\n\n  // Verify K/N are divisible by thread K/N\n  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {\n    return false;\n  }\n\n  // Verify min for thread K/N\n  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {\n    return false;\n  }\n\n  // num_threads must be at least 128 (= 4 warps)\n  if (th_config.num_threads < 128) {\n    return false;\n  }\n\n  // Check that pipeline fits into cache\n  int cache_size = get_kernel_cache_size(\n      th_config,\n      m_block_size_8,\n      thread_m_blocks,\n      prob_m,\n      prob_n,\n      prob_k,\n      num_bits,\n      group_size,\n      has_act_order,\n      is_k_full,\n      has_zp,\n      is_zp_float);\n  return cache_size + kSharedMemoryValidityMargin <= max_shared_mem;\n}\n\n#define _GET_IF(                                                                                                       \\\n    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \\\n  else if (                                                                                                            \\\n      q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                  \\\n      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&        \\\n      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                      \\\n    constexpr auto S_TYPE = W_TYPE == host::kFE2M1f                                                                    \\\n                                ? (GROUP_BLOCKS == 1 ? host::kFE4M3fn : host::kFE8M0fnu)                               \\\n                                : (std::is_same<scalar_t, half>::value ? host::kFloat16 : host::kBFloat16);            \\\n    kernel = Marlin<                                                                                                   \\\n        scalar_t,                                                                                                      \\\n        W_TYPE.id(),                                                                                                   \\\n        S_TYPE.id(),                                                                                                   \\\n        NUM_THREADS,                                                                                                   \\\n        THREAD_M_BLOCKS,                                                                                               \\\n        THREAD_N_BLOCKS,                                                                                               \\\n        THREAD_K_BLOCKS,                                                                                               \\\n        M_BLOCK_SIZE_8,                                                                                                \\\n        pipe_stages,                                                                                                   \\\n        GROUP_BLOCKS,                                                                                                  \\\n        IS_ZP_FLOAT,                                                                                                   \\\n        kIsEP,                                                                                                         \\\n        kHasBias>;                                                                                                     \\\n  }\n\n// COMMON: cases for (group_blocks in [-1, 2, 4, 8] and is_zp_float == false)\n//         this is the most common cases\n// BIGGROUP: cases for big group size (group_blocks in [-1, 8])\n// FZP: cases for float-zero-point (is_zp_float = true)\n// ACT: cases for act order case (group_blocks == 0)\n// NVFP4: cases for nvfp4(e2m1) (group_blocks == 1)\n// MXFP4: cases for mxfp4(e2m1) (group_blocks == 2)\n#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)\n\n#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \\\n                                                                        \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \\\n                                                                        \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)\n\n#define COMMON_GET_IF(W_TYPE)            \\\n  COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)\n\n#define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)\n\n#define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)   \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)\n\n#define BIGGROUP_GET_IF(W_TYPE)            \\\n  BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)\n\n#define NVFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)\n\n#define NVFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)\n\n#define NVFP4_GET_IF(W_TYPE)            \\\n  NVFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  NVFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  NVFP4_GET_IF_M234(W_TYPE, 8, 4, 128)\n\n#define MXFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)\n\n#define MXFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)\n\n#define MXFP4_GET_IF(W_TYPE)            \\\n  MXFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  MXFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  MXFP4_GET_IF_M1(W_TYPE, 4, 8, 128)    \\\n  MXFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  MXFP4_GET_IF_M234(W_TYPE, 8, 4, 128)\n\n// We currently have 4-bit models only with group_blocks == 4\n#define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)\n\n#define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)\n\n#define FZP_GET_IF(W_TYPE)            \\\n  FZP_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  FZP_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  FZP_GET_IF_M234(W_TYPE, 8, 4, 128)\n\n// We currently have 4-bit models only with group_blocks == 4\n#define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)        \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)\n\n#define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \\\n  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \\\n  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)\n\n#define ACT_GET_IF(W_TYPE)            \\\n  ACT_GET_IF_M1(W_TYPE, 8, 8, 256)    \\\n  ACT_GET_IF_M1(W_TYPE, 8, 4, 128)    \\\n  ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \\\n  ACT_GET_IF_M234(W_TYPE, 8, 4, 128)\n\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nMarlinFuncPtr get_marlin_kernel(\n    const host::ScalarType q_type,\n    int thread_m_blocks,\n    int thread_n_blocks,\n    int thread_k_blocks,\n    bool m_block_size_8,\n    bool has_act_order,\n    bool has_zp,\n    int group_blocks,\n    int num_threads,\n    bool is_zp_float) {\n  int num_bits = q_type.size_bits();\n  auto kernel = MarlinDefault;\n  if (false) {\n  }\n\n  COMMON_GET_IF(host::kU4)\n  COMMON_GET_IF(host::kU4B8)\n  COMMON_GET_IF(host::kU8B128)\n\n  NVFP4_GET_IF(host::kFE2M1f)\n\n  BIGGROUP_GET_IF(host::kFE4M3fn)\n\n  ACT_GET_IF(host::kU4B8)\n  ACT_GET_IF(host::kU8B128)\n  if (std::is_same<scalar_t, nv_bfloat16>::value) {\n    if (false) {\n    }\n    MXFP4_GET_IF(host::kFE2M1f)\n  }\n\n  return kernel;\n}\n\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nexec_config_t determine_exec_config(\n    const host::ScalarType& q_type,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    int top_k,\n    int thread_m_blocks,\n    bool m_block_size_8,\n    int num_bits,\n    int group_size,\n    bool has_act_order,\n    bool is_k_full,\n    bool has_zp,\n    bool is_zp_float,\n    int max_shared_mem,\n    int sms) {\n#if SGL_CUDA_ARCH == 900\n  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16> && !kIsEP && !kHasBias) {\n    // H200 TP8 DeepSeek-V4.1 gate/up: six sparse expert rows need a narrower\n    // N tile and fewer persistent blocks than the occupancy-only heuristic.\n    if (q_type == host::kFE2M1f && group_size == 32 && prob_m == 1 && prob_n == 640 && prob_k == 5120 && top_k == 6 &&\n        thread_m_blocks == 1 && m_block_size_8 && !has_act_order && !has_zp) {\n      return exec_config_t{2, thread_config_t{128, 64, 128}};\n    }\n  }\n#endif\n  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};\n  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;\n  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)\n                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);\n\n  int count = 0;\n  constexpr int device_max_reg_size = 255 * 1024;\n  for (int i = 0; i < thread_configs_size; i++) {\n    thread_config_t th_config = thread_configs[i];\n\n    if (!is_valid_config(\n            th_config,\n            m_block_size_8,\n            thread_m_blocks,\n            prob_m,\n            prob_n,\n            prob_k,\n            num_bits,\n            group_size,\n            has_act_order,\n            is_k_full,\n            has_zp,\n            is_zp_float,\n            max_shared_mem)) {\n      continue;\n    }\n\n    int cache_size = get_kernel_cache_size(\n        th_config,\n        m_block_size_8,\n        thread_m_blocks,\n        prob_m,\n        prob_n,\n        prob_k,\n        num_bits,\n        group_size,\n        has_act_order,\n        is_k_full,\n        has_zp,\n        is_zp_float);\n\n    int group_blocks = 0;\n    if (!has_act_order) {\n      group_blocks = group_size == -1 ? -1 : (group_size / 16);\n    }\n\n    auto kernel = get_marlin_kernel<scalar_t, kIsEP, kHasBias>(\n        q_type,\n        thread_m_blocks,\n        th_config.thread_n / 16,\n        th_config.thread_k / 16,\n        m_block_size_8,\n        has_act_order,\n        has_zp,\n        group_blocks,\n        th_config.num_threads,\n        is_zp_float);\n\n    if (kernel == MarlinDefault) continue;\n\n    cudaFuncAttributes attr;\n    cudaFuncGetAttributes(&attr, kernel);\n    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;\n    int allow_count =\n        min(device_max_reg_size / reg_size,\n            max_shared_mem / (cache_size + kSharedMemoryValidityMargin + kSharedMemoryLaunchReserve));\n    allow_count = max(min(allow_count, thread_m_blocks == 1 ? 4 : 2), 1);\n\n    if (thread_m_blocks > 1) {\n      int problem_blocks = prob_n / th_config.thread_n * prob_m * top_k * 4;\n      if (problem_blocks < sms * allow_count) {\n        allow_count = max(problem_blocks / sms, 1);\n      }\n    }\n\n    if (allow_count > count) {\n      count = allow_count;\n      exec_cfg = {count, th_config};\n    }\n  }\n\n  return exec_cfg;\n}\n\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nvoid marlin_mm(\n    const void* A,\n    const void* B,\n    void* C,\n    void* C_tmp,\n    void* b_bias,\n    void* s,\n    void* s2,\n    void* zp,\n    void* g_idx,\n    void* perm,\n    void* a_tmp,\n    void* sorted_token_ids,\n    void* expert_ids,\n    void* num_tokens_past_padded,\n    void* topk_weights,\n    int moe_block_size,\n    int top_k,\n    bool mul_topk_weights,\n    bool is_ep,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    void* workspace,\n    host::ScalarType const& q_type,\n    bool has_bias,\n    bool has_act_order,\n    bool is_k_full,\n    bool has_zp,\n    int num_groups,\n    int group_size,\n    int dev,\n    cudaStream_t stream,\n    int thread_k,\n    int thread_n,\n    int sms,\n    bool use_atomic_add,\n    bool use_fp32_reduce,\n    bool is_zp_float) {\n  int thread_m_blocks = div_ceil(moe_block_size, 16);\n  bool m_block_size_8 = moe_block_size == 8;\n\n  if (has_zp) {\n    host::RuntimeCheck(\n        q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());\n  } else {\n    host::RuntimeCheck(\n        q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,\n        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "\n        "has_zp = False. Got = ",\n        q_type.str());\n  }\n\n  host::RuntimeCheck(\n      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");\n\n  int group_blocks = 0;\n  if (has_act_order) {\n    if (is_k_full) {\n      host::RuntimeCheck(group_size != -1);\n      group_blocks = group_size / 16;\n      host::RuntimeCheck(\n          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);\n    } else {\n      host::RuntimeCheck(group_size == 0);\n      group_blocks = 0;\n    }\n  } else {\n    if (group_size == -1) {\n      group_blocks = -1;\n    } else {\n      group_blocks = group_size / 16;\n      host::RuntimeCheck(\n          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);\n    }\n  }\n\n  int num_bits = q_type.size_bits();\n  const int4* A_ptr = (const int4*)A;\n  const int4* B_ptr = (const int4*)B;\n  int4* C_ptr = (int4*)C;\n  int4* C_tmp_ptr = (int4*)C_tmp;\n  const int4* bias_ptr = (const int4*)b_bias;\n  const int4* s_ptr = (const int4*)s;\n  const uint16_t* s2_ptr = (const uint16_t*)s2;\n  const int4* zp_ptr = (const int4*)zp;\n  const int* g_idx_ptr = (const int*)g_idx;\n  const int* perm_ptr = (const int*)perm;\n  int4* a_tmp_ptr = (int4*)a_tmp;\n  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;\n  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;\n  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;\n  const float* topk_weights_ptr = (const float*)topk_weights;\n  int* locks = (int*)workspace;\n\n  if (has_act_order) {\n    // Permute A columns\n    auto perm_kernel = permute_cols_kernel<8>;\n    if (moe_block_size == 8) {\n    } else if (moe_block_size == 16)\n      perm_kernel = permute_cols_kernel<16>;\n    else if (moe_block_size == 32)\n      perm_kernel = permute_cols_kernel<32>;\n    else if (moe_block_size == 48)\n      perm_kernel = permute_cols_kernel<48>;\n    else if (moe_block_size == 64)\n      perm_kernel = permute_cols_kernel<64>;\n    else\n      host::Panic("unsupported moe_block_size ", moe_block_size);\n\n    // clang-format off\n    perm_kernel<<<sms, default_threads, 0, stream>>>(\n        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,\n        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);\n    // clang-format on\n    A_ptr = a_tmp_ptr;\n    prob_m = prob_m * top_k;\n    top_k = 1;\n\n    // If we have a full K, then we can run the non-act-order version of Marlin\n    // (since the weight rows are reordered by increasing group ids, and by\n    // having a full K, we have full original groups)\n    if (is_k_full) has_act_order = false;\n  }\n\n  int max_shared_mem = 0;\n  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  host::RuntimeCheck(max_shared_mem > 0);\n\n  // Set thread config\n  exec_config_t exec_cfg;\n  thread_config_t thread_tfg;\n  if (thread_k != -1 && thread_n != -1) {\n    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};\n    exec_cfg = exec_config_t{1, thread_tfg};\n    host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);\n    host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);\n  } else {\n    // Auto config\n    exec_cfg = determine_exec_config<scalar_t, kIsEP, kHasBias>(\n        q_type,\n        prob_m,\n        prob_n,\n        prob_k,\n        top_k,\n        thread_m_blocks,\n        m_block_size_8,\n        num_bits,\n        group_size,\n        has_act_order,\n        is_k_full,\n        has_zp,\n        is_zp_float,\n        max_shared_mem,\n        sms);\n    thread_tfg = exec_cfg.tb_cfg;\n  }\n\n  int num_threads = thread_tfg.num_threads;\n  thread_k = thread_tfg.thread_k;\n  thread_n = thread_tfg.thread_n;\n  int blocks = sms * exec_cfg.blocks_per_sm;\n  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - kSharedMemoryLaunchReserve;\n\n  int thread_k_blocks = thread_k / 16;\n  int thread_n_blocks = thread_n / 16;\n\n  host::RuntimeCheck(\n      is_valid_config(\n          thread_tfg,\n          m_block_size_8,\n          thread_m_blocks,\n          prob_m,\n          prob_n,\n          prob_k,\n          num_bits,\n          group_size,\n          has_act_order,\n          is_k_full,\n          has_zp,\n          is_zp_float,\n          max_shared_mem),\n      "Invalid thread config: thread_m_blocks = ",\n      thread_m_blocks,\n      ", thread_k = ",\n      thread_tfg.thread_k,\n      ", thread_n = ",\n      thread_tfg.thread_n,\n      ", num_threads = ",\n      thread_tfg.num_threads,\n      " for MKN = [",\n      prob_m,\n      ", ",\n      prob_k,\n      ", ",\n      prob_n,\n      "] and num_bits = ",\n      num_bits,\n      ", group_size = ",\n      group_size,\n      ", has_act_order = ",\n      has_act_order,\n      ", is_k_full = ",\n      is_k_full,\n      ", has_zp = ",\n      has_zp,\n      ", is_zp_float = ",\n      is_zp_float,\n      ", max_shared_mem = ",\n      max_shared_mem);\n\n  auto kernel = get_marlin_kernel<scalar_t, kIsEP, kHasBias>(\n      q_type,\n      thread_m_blocks,\n      thread_n_blocks,\n      thread_k_blocks,\n      m_block_size_8,\n      has_act_order,\n      has_zp,\n      group_blocks,\n      num_threads,\n      is_zp_float);\n\n  if (kernel == MarlinDefault) {\n    host::Panic(\n        "Unsupported shapes: MNK = [",\n        prob_m,\n        ", ",\n        prob_n,\n        ", ",\n        prob_k,\n        "]",\n        ", has_act_order = ",\n        has_act_order,\n        ", num_groups = ",\n        num_groups,\n        ", group_size = ",\n        group_size,\n        ", thread_m_blocks = ",\n        thread_m_blocks,\n        ", thread_n_blocks = ",\n        thread_n_blocks,\n        ", thread_k_blocks = ",\n        thread_k_blocks,\n        ", num_bits = ",\n        num_bits);\n  }\n\n  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));\n  // clang-format off\n  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(\n      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,\n      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,\n      topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,\n      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem);\n  // clang-format on\n}\n\n#endif\n\n}  // namespace device::marlin_moe\n\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nvoid moe_wna16_marlin_gemm_tuned(\n    tvm::ffi::TensorView a,\n    tvm::ffi::TensorView c,\n    tvm::ffi::TensorView b_q_weight,\n    tvm::ffi::TensorView b_bias,\n    tvm::ffi::TensorView b_scales,\n    tvm::ffi::TensorView global_scale,\n    tvm::ffi::TensorView b_zeros,\n    tvm::ffi::TensorView g_idx,\n    tvm::ffi::TensorView perm,\n    tvm::ffi::TensorView workspace,\n    tvm::ffi::TensorView sorted_token_ids,\n    tvm::ffi::TensorView expert_ids,\n    tvm::ffi::TensorView num_tokens_post_padded,\n    tvm::ffi::TensorView topk_weights,\n    tvm::ffi::TensorView a_tmp,\n    tvm::ffi::TensorView c_tmp,\n    int64_t moe_block_size,\n    int64_t top_k,\n    bool mul_topk_weights,\n    bool is_ep,\n    int64_t b_q_type_id,\n    int64_t size_m,\n    int64_t size_n,\n    int64_t size_k,\n    bool has_act_order,\n    bool has_bias,\n    bool is_k_full,\n    bool has_zp,\n    int64_t num_groups,\n    int64_t group_size,\n    bool use_atomic_add,\n    bool use_fp32_reduce,\n    bool is_zp_float, int64_t tile_k, int64_t tile_n) {\n  using namespace host;\n\n  RuntimeCheck(is_ep == kIsEP, "is_ep does not match the compiled Marlin MoE specialization");\n  RuntimeCheck(has_bias == kHasBias, "has_bias does not match the compiled Marlin MoE specialization");\n\n  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);\n  int pack_factor = 32 / b_q_type.size_bits();\n\n  if (moe_block_size != 8) {\n    RuntimeCheck(moe_block_size % 16 == 0, "unsupported moe_block_size=", moe_block_size);\n    RuntimeCheck(moe_block_size >= 16 && moe_block_size <= 64, "unsupported moe_block_size=", moe_block_size);\n  }\n\n  // Verify A\n  RuntimeCheck(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0), ", size_m = ", size_m);\n  RuntimeCheck(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1), ", size_k = ", size_k);\n\n  // Verify B\n  RuntimeCheck(\n      size_k % device::marlin::tile_size == 0,\n      "size_k = ",\n      size_k,\n      " is not divisible by tile_size = ",\n      device::marlin::tile_size);\n  RuntimeCheck(\n      (size_k / device::marlin::tile_size) == b_q_weight.size(1),\n      "Shape mismatch: b_q_weight.size(1) = ",\n      b_q_weight.size(1),\n      ", size_k = ",\n      size_k,\n      ", tile_size = ",\n      device::marlin::tile_size);\n  RuntimeCheck(\n      b_q_weight.size(2) % device::marlin::tile_size == 0,\n      "b_q_weight.size(2) = ",\n      b_q_weight.size(2),\n      " is not divisible by tile_size = ",\n      device::marlin::tile_size);\n  int64_t actual_size_n = (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor;\n  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);\n\n  // Verify device and strides\n  auto device = SymbolicDevice{};\n  device.set_options<kDLCUDA>();\n  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);\n\n  device.verify(b_q_weight.device());\n  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");\n\n  device.verify(b_scales.device());\n  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");\n\n  // thread_k, thread_n, sms\n  int thread_k = tile_k;\n  int thread_n = tile_n;\n  int sms = -1;\n  DLDevice dl_device = device.unwrap();\n  int dev = dl_device.device_id;\n  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);\n  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));\n\n  // Verify c (allocation done in Python)\n  device.verify(c.device());\n  RuntimeCheck(c.is_contiguous(), "c is not contiguous");\n  RuntimeCheck(\n      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);\n  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);\n\n  // Alloc c_tmp: SKIP, done in Python\n\n  // Detect groupsize: b_scales rank and dims\n  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");\n  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);\n  RuntimeCheck(\n      b_scales.size(1) == num_groups, "b_scales dim 1 = ", b_scales.size(1), " is not num_groups = ", num_groups);\n\n  // Validate g_idx, perm (Optional unwrap done in Python; empty tensors when absent)\n  if (g_idx.size(g_idx.dim() - 1) > 0 && perm.size(perm.dim() - 1) > 0) {\n    device.verify(g_idx.device());\n    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");\n    device.verify(perm.device());\n    RuntimeCheck(perm.is_contiguous(), "perm is not contiguous");\n\n    int64_t g_idx_last = g_idx.size(g_idx.dim() - 1);\n    int64_t perm_last = perm.size(perm.dim() - 1);\n    RuntimeCheck(\n        (g_idx_last == 0 && perm_last == 0) || (g_idx_last == size_k && perm_last == size_k),\n        "Unexpected g_idx.size(-1) = ",\n        g_idx_last,\n        " and perm.size(-1) = ",\n        perm_last,\n        ", where size_k = ",\n        size_k);\n  }\n  // has_act_order derivation: SKIP (passed as param)\n\n  // Verify group_size consistency\n  if (has_act_order) {\n    // SKIP: a_tmp allocation done in Python\n    if (is_k_full) {\n      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");\n      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);\n    }\n  } else {\n    if (num_groups > 1) {\n      RuntimeCheck(\n          size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);\n    }\n  }\n\n  // Verify global_scale (Optional unwrap done in Python)\n  int64_t global_scale_size = global_scale.size(0);\n  if (global_scale_size > 0) {\n    RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scale can only be used for nvfp4 format.");\n  } else {\n    RuntimeCheck(\n        !(b_q_type == kFE2M1f && group_size == 16), "the global_scale parameter must be passed for nvfp4 format.");\n  }\n\n  // Verify b_bias (Optional unwrap done in Python)\n  if (has_bias) {\n    device.verify(b_bias.device());\n    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");\n    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(0) != size_n");\n    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");\n  }\n\n  // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)\n\n  // Verify b_q_type vs has_zp\n  if (has_zp) {\n    device.verify(b_zeros.device());\n    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");\n    RuntimeCheck(\n        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());\n  } else {\n    RuntimeCheck(\n        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,\n        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "\n        "float4_e2m1f when "\n        "has_zp = False. Got = ",\n        b_q_type.str());\n  }\n\n  if (has_zp && is_zp_float) {\n    RuntimeCheck(\n        std::is_same<scalar_t, fp16_t>::value,\n        "Computation type must be float16 (half) when using float zero "\n        "points.");\n  }\n\n  if (b_q_type == kFE2M1f) {\n    RuntimeCheck(\n        group_size == 16 || group_size == 32,\n        "float4_e2m1f only supports group_size == 16 (NVFP4) or group_size == 32 (MXFP4). Got group_size = ",\n        group_size);\n    RuntimeCheck(\n        group_size != 32 || std::is_same<scalar_t, nv_bfloat16>::value,\n        "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 activations.");\n  }\n\n  // Verify b_zeros\n  if (has_zp) {\n    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");\n    if (is_zp_float) {\n      RuntimeCheck(b_zeros.size(2) == size_n, "b_zeros dim 2 = ", b_zeros.size(2), " is not size_n = ", size_n);\n      RuntimeCheck(\n          num_groups == b_zeros.size(1), "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);\n      RuntimeCheck(num_groups != -1, "num_groups must be != -1");\n    } else {\n      RuntimeCheck(\n          b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);\n      RuntimeCheck(\n          b_zeros.size(2) == size_n / pack_factor,\n          "b_zeros dim 2 = ",\n          b_zeros.size(2),\n          " is not size_n / pack_factor = ",\n          size_n / pack_factor);\n    }\n  }\n\n  // Verify workspace size\n  RuntimeCheck(\n      size_n % device::marlin::min_thread_n == 0,\n      "size_n = ",\n      size_n,\n      ", is not divisible by min_thread_n = ",\n      device::marlin::min_thread_n);\n\n  int64_t max_n_tiles = size_n / device::marlin::min_thread_n;\n  int64_t min_workspace_size =\n      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);\n  RuntimeCheck(\n      workspace.size(0) >= min_workspace_size,\n      "workspace.numel = ",\n      workspace.size(0),\n      " is below min_workspace_size = ",\n      min_workspace_size);\n\n  // Early return for zero-size M (moved after all validation)\n  if (size_m == 0) return;\n\n  device::marlin_moe::marlin_mm<scalar_t, kIsEP, kHasBias>(\n      a.data_ptr(),\n      b_q_weight.data_ptr(),\n      c.data_ptr(),\n      c_tmp.data_ptr(),\n      b_bias.data_ptr(),\n      b_scales.data_ptr(),\n      global_scale.data_ptr(),\n      b_zeros.data_ptr(),\n      g_idx.data_ptr(),\n      perm.data_ptr(),\n      a_tmp.data_ptr(),\n      sorted_token_ids.data_ptr(),\n      expert_ids.data_ptr(),\n      num_tokens_post_padded.data_ptr(),\n      topk_weights.data_ptr(),\n      static_cast<int>(moe_block_size),\n      static_cast<int>(top_k),\n      mul_topk_weights,\n      is_ep,\n      static_cast<int>(size_m),\n      static_cast<int>(size_n),\n      static_cast<int>(size_k),\n      workspace.data_ptr(),\n      b_q_type,\n      has_bias,\n      has_act_order,\n      is_k_full,\n      has_zp,\n      static_cast<int>(num_groups),\n      static_cast<int>(group_size),\n      dev,\n      stream,\n      thread_k,\n      thread_n,\n      sms,\n      use_atomic_add,\n      use_fp32_reduce,\n      is_zp_float);\n}\n\n}  // namespace sglang\n'
from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.kernels.jit import utils as jit_utils
TILE_K = -1
TILE_N = -1

def _tuned_module():
    args = make_cpp_args(torch.bfloat16, True, False)
    source_dir = Path(tempfile.gettempdir()) / "muse_marlin_tiles_v1"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / "marlin_tuned.cuh"
    source_path.write_text(MARLIN_CUDA_SOURCE)
    return load_jit("muse_moe_tiles_v1", *args,
        cuda_files=[str(source_path)],
        extra_include_paths=[str(Path(jit_utils.__file__).parents[1] / "csrc")],
        cuda_wrappers=[("moe_wna16_marlin_gemm", f"moe_wna16_marlin_gemm_tuned<{args}>")])


import torch
import triton
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import (
    _jit_moe_wna16_marlin_module,
)
try:
    from sglang.kernels.ops.moe.moe_wna16_marlin import _MAX_THREAD_N
except ImportError:
    _MAX_THREAD_N = 128

from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
MAX_TOKENS = 512

# Optimal block_size_m: 16 for bs=16 saves 35 us over baseline bs=8
BLOCK_SIZE_M_MAP = {
    1: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 16,
    64: 16,
    128: 16,
    256: 16,
}


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    starts = [0]
    counts = [32]
    banks = {}
    for start in starts:
        for count in counts:
            if start + count > N_EXPERTS:
                continue
            expert_map = None
            if count != N_EXPERTS:
                expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
                expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
            layer = _marlin_layer(weights, start, count)
            w13_num_groups = layer.w13_weight_scale.size(1)
            w13_group_size = 4096 // w13_num_groups if w13_num_groups > 1 else -1
            w2_num_groups = layer.w2_weight_scale.size(1)
            w2_group_size = layer.intermediate_size_per_partition // w2_num_groups if w2_num_groups > 1 else -1
            banks[(start, count)] = {
                "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
                "w13_num_groups": w13_num_groups, "w13_group_size": w13_group_size,
                "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
                "w2_num_groups": w2_num_groups, "w2_group_size": w2_group_size,
                "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
            }
    torch.cuda.synchronize()

    size_n = 4096
    size_k = 4096
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    max_padded = 32768
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)
    out_buf = torch.empty((MAX_TOKENS, size_k), device=device, dtype=torch.bfloat16)

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    marlin_mod_ep = _tuned_module()
    marlin_mod_noep = marlin_mod_ep
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    c_tmp_buf = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=device)

    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "out_buf": out_buf,
        "sms": sms,
        "marlin_mod_ep": marlin_mod_ep,
        "marlin_mod_noep": marlin_mod_noep,
        "empty_bf16": empty_bf16,
        "empty_i32": empty_i32,
        "empty_f32": empty_f32,
        "c_tmp_buf": c_tmp_buf,
        "b_q_type_id": scalar_types.float4_e2m1f.id,
    }


def _fast_marlin(
    a, c, b_q_weight, b_scales, b_global_scale,
    sorted_ids, block_experts, n_padded, topk_weights,
    moe_block_size, top_k, is_ep,
    size_m, size_n, size_k,
    num_groups, group_size,
    module, workspace, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
    use_fp32_reduce=True,
):
    if use_fp32_reduce:
        max_c_tmp_size = min(
            size_n * sorted_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = c_tmp_buf[:max_c_tmp_size]
    else:
        c_tmp = empty_f32

    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        empty_bf16,
        b_scales,
        b_global_scale,
        empty_bf16,
        empty_i32,
        empty_i32,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        empty_bf16,
        c_tmp,
        moe_block_size,
        top_k,
        False,  # mul_topk_weights
        is_ep,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        False,  # has_act_order
        False,  # has_bias
        True,   # is_k_full
        False,  # has_zp
        num_groups,
        group_size,
        False,  # use_atomic_add
        use_fp32_reduce,
        False,  # is_zp_float
        TILE_K, TILE_N,
    )
    return c


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"]
    if expert_map is not None:
        local_ids = torch.where(local_ids < bank["count"], local_ids, -1)
    topk_weights = data["topk_weights"]

    block_size_m = BLOCK_SIZE_M_MAP.get(num_tokens, 32)

    numel = local_ids.numel()
    num_exp = bank["count"]
    if numel < num_exp + 1:
        max_num_tokens_padded = numel * block_size_m
    else:
        max_num_tokens_padded = numel + (num_exp + 1) * (block_size_m - 1)
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4
    max_num_m_blocks = max_num_tokens_padded // block_size_m

    sorted_ids = state["align_sorted_ids"][:max_num_tokens_padded]
    block_experts = state["align_expert_ids"][:max_num_m_blocks]
    n_padded = state["align_n_padded"]

    if numel <= SMALL_NUMEL_LIMIT and (num_exp + 1) > 64:
        moe_align_small_numel(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
        )
    else:
        sgl_moe_align_block_size(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
            state["align_cumsum"],
            True,
            False,
        )

    total_pairs = num_tokens * TOP_K
    cache2 = state["cache2_buf"][:total_pairs, :size_n]
    cache1 = state["cache13_buf"][: total_pairs * gemm1_n].view(-1, gemm1_n)
    cache3 = state["cache13_buf"][: total_pairs * size_k].view(-1, size_k)

    is_ep = expert_map is not None
    module = state["marlin_mod_ep"] if is_ep else state["marlin_mod_noep"]
    ws = state["workspace"]
    empty_bf16 = state["empty_bf16"]
    empty_i32 = state["empty_i32"]
    c_tmp_buf = state["c_tmp_buf"]
    sms = state["sms"]
    b_q_type_id = state["b_q_type_id"]
    empty_f32 = state["empty_f32"]
    use_fp32_reduce = (num_tokens not in (1, 8))

    _fast_marlin(
        x, cache1, bank["w13"], bank["w13_s"], bank["w13_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, TOP_K, is_ep,
        num_tokens, gemm1_n, size_k,
        bank["w13_num_groups"], bank["w13_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    _fast_marlin(
        cache2, cache3, bank["w2"], bank["w2_s"], bank["w2_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, 1, is_ep,
        total_pairs, size_k, size_n,
        bank["w2_num_groups"], bank["w2_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    y = cache3.view(num_tokens, TOP_K, size_k)

    # Direct Triton fused post-expert norm for all token counts
    out = state["out_buf"][:num_tokens]
    block_h = triton.next_power_of_2(size_k)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y,
        state["norm_w"],
        topk_weights,
        out,
        local_ids,
        TOPK=TOP_K,
        H=size_k,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out
