"""Build a disconnected NCW2 n64 tail experiment from the promoted grid-P source.

Upstream Apache-2.0 source copyright 2026 Lee Penkman. Full license is in
../OPENKERNELS_LICENSE.txt. This does not alter a production dispatch.
"""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def build():
    tree = ast.parse((ROOT / 'candidate_prefill_grid_constant.py').read_text())
    source = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'W4G_CUDA' for t in n.targets))
    gen = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_w4g_wgm')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[gen], type_ignores=[])), '<generator>', 'exec'), scope)
    source = source.replace('@@WGM@@', scope['_w4g_wgm']())
    replacements = {
        'static constexpr int MINB = NCW == 1 ? 4 : 1;':
        'static constexpr int MINB = (NT == 64 && NCW == 2) ? 2 : (NCW == 1 ? 4 : 1);',
        '(NT == 32 ? (rem > 0 && rem <= 32 ? 1 : 0) : full + (!p.fullonly && rem > 32 ? 1 : 0))':
        '(NT == 64 ? (rem >= 33 && rem <= 64 ? 1 : 0) : NT == 32 ? (rem > 0 && rem <= 32 ? 1 : 0) : full + (!p.fullonly && rem > 32 ? 1 : 0))',
        'const int t0 = NT == 32 ? cnt / p.tsz : tt;':
        'const int t0 = (NT == 32 || NT == 64) ? cnt / p.tsz : tt;',
        'if constexpr (NCW == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\\n" ::: "memory");':
        'if constexpr (NCW == 2 && NT == 64) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\\n" ::: "memory");\n    else if constexpr (NCW == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\\n" ::: "memory");',
        'if constexpr (NCW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\\n" ::: "memory");':
        'if constexpr (NCW == 2 && NT == 64) asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\\n" ::: "memory");\n    else if constexpr (NCW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\\n" ::: "memory");',
    }
    for old, new in replacements.items():
        assert old in source, old
        source = source.replace(old, new)
    source = source.replace('(!p.fullonly && rem > 32 ? 1 : 0)', '((p.fullonly == 0 && rem > 32) || (p.fullonly == 2 && rem > 64) ? 1 : 0)')
    source += r'''
// Research only: emits just final 33..64-token tails relative to 128-token tiles.
// All other rows are intentionally untouched. Full K only; no partial BF16 sum.
void w4tail64(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g,
              torch::Tensor out, torch::Tensor ctr, torch::Tensor prof, int64_t fuse) {
  TORCH_CHECK(A.size(1) == 4096 && offs.numel() == 32, "tail64 input shape");
  w4g_init();
  cudaFuncSetAttribute(w4g_k<64,2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem_need<64,2>());
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(),
      g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(),
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr,
      (int)W.size(1), (int)out.size(1), (int)out.size(0), 1, 128, 0, 0, (int)fuse};
  w4g_k<64,2><<<2*g_nsm, Cfg<64,2>::NTHR, smem_need<64,2>(),
                 at::cuda::getCurrentCUDAStream()>>>(p, tmap(A.data_ptr(), A.size(0), 64));
}
int64_t tail64_smem() { return smem_need<64,2>(); }
int64_t tail64_occupancy() {
  int n = 0;
  cudaFuncSetAttribute(w4g_k<64,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_need<64,2>());
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,w4g_k<64,2>,384,smem_need<64,2>()) == cudaSuccess, "occupancy query");
  return n;
}
// Mode0 sequence: big kernel excludes only 33..64 tails, then old n32, then n64.
void w4filtered(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g,
                torch::Tensor out, torch::Tensor ctr, torch::Tensor prof, int64_t fuse) {
  w4g_init();
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(),
      g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), nullptr,
      (int)W.size(1), (int)out.size(1), (int)out.size(0), 1, 128, 0, 2, (int)fuse};
  auto st = at::cuda::getCurrentCUDAStream();
  KBIG<<<g_nsm, Cfg<128,4>::NTHR, smem_need<128,4>(),st>>>(p,tmap(A.data_ptr(),A.size(0),128));
  p.ctr += 4; p.fullonly = 0;
  KSMALL<<<4*g_nsm,Cfg<32,1>::NTHR,smem_need<32,1>(),st>>>(p,tmap(A.data_ptr(),A.size(0),32));
  w4tail64(A,offs,W,g,out,ctr.slice(0,8,10),prof,fuse);
}

'''
    return source

if __name__ == '__main__':
    dest = ROOT / 'research/kernel_sources/w4tail64.cu'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(build())
    print(dest)
