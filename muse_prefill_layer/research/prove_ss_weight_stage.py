"""CPU proof: existing RS fragments cover the proposed SS SW128 stage bijectively.

This proves addressing only, not asynchronous barrier safety or performance.
Weights and scales must continue through the existing exact dq_half routine.
"""
for ncw in (1,2,4):
    seen={}
    for group in range(ncw):
        for warp in range(4):
            for lane in range(32):
                row0=group*64+warp*16+(lane>>2)
                col0=2*(lane&3)
                for ki in range(4):
                    for reg,(r,c) in enumerate(((row0,col0),(row0+8,col0),(row0,col0+8),(row0+8,col0+8))):
                        for half in (0,1):
                            col=ki*16+c+half
                            logical=r*128+2*col
                            shared=logical^(((logical>>7)&7)<<4)
                            assert 0<=shared<ncw*64*128
                            assert shared not in seen
                            seen[shared]=(r,col)
    expected={(r,c) for r in range(ncw*64) for c in range(64)}
    assert set(seen.values())==expected
    assert set(seen)==set(range(0,ncw*64*128,2))
    print('PASS',ncw,'warpgroups',len(seen),'BF16 weights; SW128 in bounds, bijective, preserves RS coordinates')
