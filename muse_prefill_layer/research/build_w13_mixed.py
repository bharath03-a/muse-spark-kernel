"""Enable only measured standard/mixed W13 fusion modes, preserving all pins."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'candidate_prefill_w13_integrated.py').read_text()
old='FUSED_W13_MODES = (0, 1)'
assert s.count(old)==1
s=s.replace(old,'FUSED_W13_MODES = (0, 2)')
s=s.replace('# Mode2 uses two different big-tile schedules plus a remainder and remains native.', '# Mode2 passed skewed/uneven finite numerical and paired timing validation.')
p=r/'candidate_prefill_w13_mixed.py'
p.write_text(s)
compile(s,str(p),'exec')
print(p)
