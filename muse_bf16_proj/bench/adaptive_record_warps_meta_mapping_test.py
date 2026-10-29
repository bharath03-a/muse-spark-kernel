"""CPU-only proof of original tile-order metadata preload and safe tail reads."""
from pathlib import Path
import ast
from adaptive_record_warps_mapping_test import main as existing_mapping_proof


def main():
    existing_mapping_proof()
    for cw in (4,8):
        for k in (1024,4096,8192):
            ki,nr=k//(cw*64),2*cw
            assert ki & (ki-1)==0
            for cta in range(2):
                n0=cta*32
                staged=[[None]*nr for _ in range(ki)]
                seen=set()
                for step in range(ki):
                    for r in range(nr):
                        # Exact m2 preload logical coordinates. Metadata retains
                        # _ac_pack row16-major, K64-record-major ordering.
                        rid=(n0//16+r//cw)*(k//64)+(r%cw)*ki+step
                        assert rid not in seen
                        seen.add(rid)
                        raw=(rid%7)==0
                        value=(rid*128)|(int(raw)<<31)
                        staged[step][r]=value
                origin=(n0//16)*(k//64)
                assert seen==set(range(origin,origin+2*(k//64)))
                # Simulate byte-identical unswizzled [KI,NR] -> [KI,NR,1]
                # view and runtime .index(min(step,KI-1)).load(cl).
                physical=[v for row in staged for v in row]
                for depth in (2,3,4):
                    if ki<depth:continue
                    for step in range(ki+depth):
                        for r in range(nr):
                            slot=min(step,ki-1)*nr+r
                            assert 0<=slot<len(physical)
                            pred=step<ki
                            loaded=physical[slot] if pred else 0
                            rid=(n0//16+r//cw)*(k//64)+(r%cw)*ki+step
                            expect=((rid*128)|(int(rid%7==0)<<31)) if pred else 0
                            assert loaded==expect
                            assert ((loaded>>31)&1)==(int(rid%7==0) if pred else 0)
                            assert (loaded&0x7fffffff)*16==(rid*128*16 if pred else 0)
                assert len(physical)*4==2*(k//64)*4
            if k==8192:
                ring=3*(cw*4096+cw*32*128+cw*8)
                assert ring+2*(k//64)*4==({4:99424,8:197824}[cw])
    root=Path(__file__).resolve().parent
    for filename in ('adaptive_record_async_warps_meta.py','adaptive_warps_meta_compile_only.py','top50_candidate_cw8_meta_pooler.py'):
        ast.parse((root/filename).read_text())
    print('PASS: CW4/8 tile-order metadata coverage, flags/offsets, tail clamps, '
          'KI2..32 layouts, 1024B pooler metadata; candidate/probe/driver AST PASS')


if __name__=='__main__':main()
