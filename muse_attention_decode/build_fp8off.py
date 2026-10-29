"""v75 FP8-PV with an FA3-style probability offset (P scaled by 2^FOFF before E4M3 rounding).

Without the offset, P = 2^(s - running max) <= 1 and every probability below 2^-9 of the row max flushes to zero in
E4M3. The scored full layers use dist="coherent": the many low-attention keys share a value direction, so their summed
tail mass is exactly what gets lost. FA3 (the installed baseline) offsets by 8 for this reason. Here P = 2^(s - mx +
FOFF), lazy-max headroom shrinks to 8 - FOFF so P stays <= 2^8 (E4M3 max 448), and the overflow rescale restores max P
to 2^FOFF. Numerator (PV) and denominator (P x ones) use the same rounded P, so the offset cancels. attnp only.
"""
import ast
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
foff = int(sys.argv[1]) if len(sys.argv) > 1 else 4
src = root / 'candidate_openkernels_v75.py'
text = src.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CUDA')
c = node.value.value
a = c.index('__global__ void __launch_bounds__(128 * (2 * NL + 1), 1) attnp(A a, WF f) {')
b = c.index('__global__ void gtres_k(')
body = c[a:b]


def change(s, old, new, n=1):
    assert s.count(old) == n, (old[:90], s.count(old), n)
    return s.replace(old, new)


body = change(body, 'ex2p(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh])) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh])));',
              'ex2p(fmaf(s[4 * j + 2 * hh + e], a.qk, (FP8 ? FOFF : 0.f) - mx[hh])) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, (FP8 ? FOFF : 0.f) - mx[hh])));')
body = change(body, '''          if (pm > LZP) {
            const float inv = 1.f / pm;
            mx[hh] += __log2f(pm);''', '''          if (pm > LZP) {
            const float inv = (FP8 ? FOFFP : 1.f) / pm;  // max P back to 2^FOFF (FP8) or 1
            mx[hh] += __log2f(pm) - (FP8 ? FOFF : 0.f);''')
body = change(body, 'const bool up = bm * a.qk > mx[hh] + LZT;', 'const bool up = bm * a.qk > mx[hh] + (FP8 ? LZT - FOFF : LZT);')
body = change(body, ': ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf));',
              ': ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, (FP8 ? FOFF : 0.f) - nmf));', 2)
c = c[:a] + f'#define FOFF {foff}.f\n#define FOFFP {float(2 ** foff)}f\n' + body + c[b:]
text = text.replace(ast.get_source_segment(text, node.value), repr(c), 1)
text = text.replace('"attn_v75_%d"', f'"attn_v75off{foff}_%d"', 1)
assert f'attn_v75off{foff}_' in text
text = text.replace('# Isolated v75 audit/probe; best submission.py remains unchanged.',
                    f'# v75 FP8-PV + FA3-style P offset 2^{foff} (build_fp8off.py); source openkernels 47103fd, Apache-2.0.', 1)
out = root / f'candidate_v75_fp8off{foff}.py'
out.write_text(text)
compile(text, str(out), 'exec')
print(out)
