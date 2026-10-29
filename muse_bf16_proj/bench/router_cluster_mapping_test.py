"""CPU-only proof: split-plane CTAs redistribute into contiguous token owners.

Mirrors BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0]) and
CGA src[[1,0,0],...] / dst[[0,0,1],...]. No CUDA imports or jobs.
"""
from collections import Counter


def prove(c):
    rdim,tdim=32,64
    chunk=tdim//c
    srcset=set()
    destinations=[{} for _ in range(c)]
    incoming=Counter();pairbytes=Counter()
    for source in range(c):
        for thread in range(128):
            warp,lane=divmod(thread,32)
            row=warp*8+lane//4
            for reg in range(16):
                token=(lane%4)*2+(reg%2)+(reg//2)*8
                key=(source,row,token)
                assert key not in srcset
                srcset.add(key)
                owner=token//chunk
                local_t=token%chunk
                offset=((source*rdim+row)*chunk+local_t)*4
                assert 0<=offset<8192 and offset%4==0
                assert offset not in destinations[owner]
                destinations[owner][offset]=key
                incoming[owner]+=4
                pairbytes[source,owner]+=4
    assert len(srcset)==c*rdim*tdim
    for destination in range(c):
        assert incoming[destination]==8192
        assert len(destinations[destination])==2048
        assert set(destinations[destination])==set(range(0,8192,4))
        for source in range(c):assert pairbytes[source,destination]==8192//c
        outputs={}
        reads=set()
        for thread in range(128):
            warp,lane=divmod(thread,32)
            row=warp*8+lane//4
            # Register repetitions first fill the local token tile, then source.
            token_regs=2*(chunk//8)
            for reg in range(16):
                source=reg//token_regs
                tr=reg%token_regs
                local_t=(lane%4)*2+tr%2+(tr//2)*8
                token=destination*chunk+local_t
                offset=((source*rdim+row)*chunk+local_t)*4
                assert destinations[destination][offset]==(source,row,token)
                assert offset not in reads
                reads.add(offset)
                outputs.setdefault((row,token),[]).append(source)
        assert len(reads)==2048
        assert len(outputs)==rdim*chunk
        for (row,token),sources in outputs.items():
            assert sorted(sources)==list(range(c))
            values=[(s+1)*1000+row*tdim+token for s in sources]
            assert sum(values)==sum((s+1)*1000+row*tdim+token for s in range(c))
    print(f'PASS C={c}: {len(srcset)} unique partials; {chunk} contiguous tokens/owner; '
          f'{8192//c} B/source->owner; 8192 B expected/CTA incl self; complete C-way local sums')


def prove_mainloop_addresses():
    # Entire logical router grid: each source CTA owns one contiguous K shard.
    for m,c in ((128,8),(256,4)):
        n,k,bk,depth=256,8192,128,3
        clusters=(n//32)*(m//64)
        assert clusters*c==128
        covered_outputs=set()
        for pid in range(clusters):
            n0=(pid%(n//32))*32
            t0=(pid//(n//32))*64
            for row in range(n0,n0+32):
                for token in range(t0,t0+64):
                    assert (token,row) not in covered_outputs
                    covered_outputs.add((token,row))
        assert len(covered_outputs)==m*n
        ki=k//(c*bk)
        shards=[]
        for split in range(c):
            columns=[split*(k//c)+step*bk+kk for step in range(ki) for kk in range(bk)]
            assert columns==list(range(split*(k//c),(split+1)*(k//c)))
            shards.extend(columns)
            for step in range(ki+depth):
                for start,rows in ((0,32),(32,32),(0,64)):
                    # Statically valid prologue masks and masked tail pointers.
                    offset=(start+rows-1)*k+split*(k//c)+step*bk+(bk-1)
                    assert offset>=0
                    if step<ki:assert offset<(start+rows)*k
        assert shards==list(range(k))
        per_cta_ring=depth*(32+32+64)*bk*2
        assert per_cta_ring==98304
        print(f'PASS mainloop M={m}/C={c}:128 physicalCTAs; complete disjoint outputtiles/Kshards; '
              f'KI={ki};98304B ring/CTA; static/tail masks tensorizable')


if __name__=='__main__':
    for c in (2,4,8):prove(c)
    prove_mainloop_addresses()
