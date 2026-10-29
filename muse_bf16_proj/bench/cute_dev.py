"""CUTLASS 3.x sm90 warp-specialized TMA + wgmma GEMMs for m = 128/256 (out = x @ W^T, bf16).

Each config is its own translation unit so ninja builds them in parallel. Build: --compile (CPU ok).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time

# name -> (tile m, tile n, tile k, cluster m, cluster n, kind, stages); kind 0 = cooperative + stream-K
# scheduler, 1 = pingpong + persistent, 2 = cooperative + persistent. stages 0 = auto carveout.
CONFIGS = {
    "c128x128": (128, 128, 64, 1, 1, 0, 0),
    "c128x64": (128, 64, 128, 1, 1, 0, 0),
    "c256x128": (256, 128, 64, 1, 1, 0, 0),
    "c128x128c2": (128, 128, 64, 2, 1, 0, 0),
}

HEADER = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;

template <int TM, int TN, int TK, int CM, int CN, int KIND, int STAGES>
struct Cfg {
  using E = cutlass::bfloat16_t;
  using LA = cutlass::layout::RowMajor;
  using LB = cutlass::layout::ColumnMajor;
  using LD = cutlass::layout::RowMajor;
  using Tile = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using Cl = Shape<Int<CM>, Int<CN>, _1>;
  using KS = std::conditional_t<KIND == 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using ES = std::conditional_t<KIND == 1, cutlass::epilogue::TmaWarpSpecialized,
                                cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cl,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LD, 8, E, LD, 8, ES>::CollectiveOp;
  using Stages = std::conditional_t<STAGES == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::collective::StageCount<(STAGES > 0 ? STAGES : 1)>>;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, E, LA, 8, E, LB, 8, float, Tile, Cl, Stages,
      KS>::CollectiveOp;
  using Sched = std::conditional_t<KIND == 0, cutlass::gemm::StreamKScheduler, cutlass::gemm::PersistentScheduler>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

static int sm_count() {
  static int n = 0;
  if (!n) n = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return n;
}

// mode: 0 heuristic, 1 data-parallel, 2 split-K, 3 stream-K (stream-K scheduler only).
template <class C>
int64_t run_cfg(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, int64_t splits,
                int64_t mode, int64_t swizzle, int64_t raster, int64_t sms) {
  using Gemm = typename C::Gemm;
  using K = typename Gemm::GemmKernel;
  int M = x.size(0), Kd = x.size(1), N = w.size(0);
  auto sa = cutlass::make_cute_packed_stride(typename K::StrideA{}, cute::make_shape(M, Kd, 1));
  auto sb = cutlass::make_cute_packed_stride(typename K::StrideB{}, cute::make_shape(N, Kd, 1));
  auto sd = cutlass::make_cute_packed_stride(typename K::StrideD{}, cute::make_shape(M, N, 1));
  using E = cutlass::bfloat16_t;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, Kd, 1},
      {reinterpret_cast<E const*>(x.data_ptr()), sa, reinterpret_cast<E const*>(w.data_ptr()), sb},
      {{1.f, 0.f}, nullptr, sd, reinterpret_cast<E*>(out.data_ptr()), sd}};
  args.hw_info.device_id = 0;
  args.hw_info.sm_count = sms > 0 ? int(sms) : sm_count();
  using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
  args.scheduler.max_swizzle_size = int(swizzle);
  args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
  if constexpr (std::is_same_v<typename C::Sched, cutlass::gemm::StreamKScheduler>) {
    using DM = cutlass::gemm::kernel::detail::DecompositionMode;
    args.scheduler.splits = int(splits);
    args.scheduler.decomposition_mode =
        mode == 1 ? DM::DataParallel : mode == 2 ? DM::SplitK : mode == 3 ? DM::StreamK : DM::Heuristic;
    args.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
  }
  size_t need = Gemm::get_workspace_size(args);
  if (ws.numel() < 0) return int64_t(need);
  TORCH_CHECK(size_t(ws.numel()) >= need, "cutlass workspace too small: ", need);
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cutlass: cannot implement");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "cutlass: init failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "cutlass: launch failed");
  return int64_t(need);
}

#define DECL(name) int64_t name(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, \
                                int64_t, int64_t, int64_t);
"""

FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
         "-DNDEBUG", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "--use_fast_math"]


def cutlass_dir() -> str:
    spec = importlib.util.find_spec("flashinfer")
    for loc in spec.submodule_search_locations if spec else ():
        d = os.path.join(loc, "data", "cutlass")
        if os.path.isdir(os.path.join(d, "include", "cute")):
            return d
    raise RuntimeError("cutlass headers not found")


def build(configs: dict = CONFIGS, verbose: bool = False):
    from torch.utils.cpp_extension import load

    root = cutlass_dir()
    tag = hashlib.sha1((HEADER + repr(sorted(configs.items())) + " ".join(FLAGS)).encode()).hexdigest()[:10]
    base = os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
    src_dir = os.path.join(base, f"cutsrc_{tag}")
    os.makedirs(src_dir, exist_ok=True)
    sources = []
    for name, p in configs.items():
        path = os.path.join(src_dir, f"{name}.cu")
        body = HEADER + f"int64_t {name}(torch::Tensor x, torch::Tensor w, torch::Tensor o, torch::Tensor ws, " \
            "int64_t s, int64_t m, int64_t z, int64_t r, int64_t n) {\n" \
            f"  return run_cfg<Cfg<{', '.join(map(str, p))}>>(x, w, o, ws, s, m, z, r, n);\n}}\n"
        _write_if_changed(path, body)
        sources.append(path)
    decls = "".join(f"DECL({n})\n" for n in configs)
    binds = "".join(f'  m.def("{n}", &{n});\n' for n in configs)
    main = os.path.join(src_dir, "bind.cu")
    _write_if_changed(main, HEADER + decls + "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n" + binds + "}\n")
    sources.append(main)
    t0 = time.time()
    os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 8))
    mod = load(name=f"muse_cutlass_{tag}", sources=sources, extra_cuda_cflags=FLAGS, extra_cflags=["-O3"],
               extra_include_paths=[os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")],
               extra_ldflags=["-lcuda"], verbose=verbose)
    print(f"cutlass build/load {time.time() - t0:.1f}s ({len(configs)} configs)", flush=True)
    return mod


def _write_if_changed(path: str, text: str) -> None:
    if os.path.exists(path) and open(path).read() == text:
        return
    with open(path, "w") as f:
        f.write(text)


def launch(mod, cfg: str, x, w, ws, splits=1, mode=0, swizzle=1, raster=0, sms=0):
    import torch

    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    getattr(mod, cfg)(x, w, out, ws, splits, mode, swizzle, raster, sms)
    return out


if __name__ == "__main__":
    if "--compile" in sys.argv:
        build(verbose="-v" in sys.argv)
