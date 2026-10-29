"""Create isolated BF16-PV attention variant, retaining stable online softmax."""
from pathlib import Path
ROOT=Path(__file__).parent
source=(ROOT/'candidate_openkernels_attention.py').read_text()
original=source
# Clone only the register-A PV and denominator MMA helpers; FP8 QK is untouched.
helpers=[]
for name,nextname in [('wg_rsz','wg_rs8'),('wg_rs8','wg_rs'),('wg_rs','lds32a')]:
 start=source.index('__device__ __forceinline__ void '+name+'(')
 finish=source.index('__device__ __forceinline__',start+len('__device__ __forceinline__'))
 fragment=source[start:finish]
 fragment=fragment.replace(name+'(',name+'_bf16(').replace('.f32.f16.f16 ','.f32.bf16.bf16 ')
 # The emulator expects half. This variant is Hopper-only; do not claim emulation support.
 import re
 fragment=re.sub(r'#elif WGEMU\n.*?(?=#endif)', '',fragment,flags=re.S)
 helpers.append(fragment)
helpers.append('''
__device__ __forceinline__ uint32_t pk_bf16(float x, float y) {
  __nv_bfloat162 b = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&b);
}
__device__ __forceinline__ uint32_t e2b(uint32_t w) {
  uint32_t h = e2h(w);
  float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
  return pk_bf16(f.x, f.y);
}
''')
start=source.index('template <int NL, int X>\n__global__ void __launch_bounds__(128 * (2 * NL + 1), 1) attnp(')
finish=source.index('template <int NL, int X>\nstatic void apf',start)
part=source[start:finish]
# Safe first probe: preserve running maxima, masked scores and exact rescaling;
# disable the packed-half fast-range check because probabilities are now BF16.
part=part.replace('constexpr bool FS = X & 4,','constexpr bool FS = false,')
part=part.replace('return AP_NOCVT ? w : (ACV ? e2h_alu(w, hi) : e2h(hi ? w >> 16 : w));','return AP_NOCVT ? w : e2b(hi ? w >> 16 : w);')
part=part.replace('const float vsc_ = f.vscale * (ACV ? 256.f : 1.f);','const float vsc_ = f.vscale;')
part=part.replace('0x3C003C00u','0x3F803F80u')
part=part.replace('pa[i] = pk(', 'pa[i] = pk_bf16(')
for name in ['wg_rsz','wg_rs8','wg_rs']:
 part=part.replace(name+'(',name+'_bf16(')
assert 'e2b(hi ? w >> 16 : w)' in part
source=source[:start]+''.join(helpers)+part+source[finish:]
source=source.replace('load_inline("attn_v60_%d"','load_inline("attn_v60_bf16pv_%d"')
(ROOT/'candidate_openkernels_bf16pv.py').write_text(source)
import ast
ast.parse(source)
print('Generated candidate_openkernels_bf16pv.py; safe online-softmax retained, Hopper-only BF16-PV')

# Matched control changes only the same fastpath toggle, keeping half P/V.
a=original.index('template <int NL, int X>\n__global__ void __launch_bounds__(128 * (2 * NL + 1), 1) attnp(')
b=original.index('template <int NL, int X>\nstatic void apf',a)
control=original[:a]+original[a:b].replace('constexpr bool FS = X & 4,','constexpr bool FS = false,')+original[b:]
control=control.replace('load_inline("attn_v60_%d"','load_inline("attn_v60_pvcontrol_%d"')
(ROOT/'candidate_openkernels_pvcontrol.py').write_text(control)
ast.parse(control)
print('Generated matched FP16-PV/FS-off control')
