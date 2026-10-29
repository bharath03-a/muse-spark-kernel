"""CPU-only CW4/8 physical record and ring address proof."""
from adaptive_record_mapping_test import main as vector_proof


def main():
    vector_proof()
    for cw in (4, 8):
        seen = set()
        n, k = 64, 1024
        ki = k // (cw * 64)
        for cta in range(n // 32):
            for it in range(ki):
                for record in range(2 * cw):
                    row16, batch = divmod(record, cw)
                    # issue_warps uses this exact rowtile/Krecord map.
                    rid = ((cta * 32 + row16 * 16) // 16) * (k // 64) + batch * ki + it
                    assert rid == (cta * 2 + row16) * (k // 64) + batch * ki + it
                    for word in range(512):
                        for half in range(2):
                            plane, rest = divmod(word, 128)
                            lane, subword = divmod(rest, 4)
                            row = cta * 32 + row16 * 16 + (subword & 1) * 8 + lane // 4
                            col = (plane // 2) * 32 + (plane & 1) * 16 + (subword // 2) * 8 + (lane & 3) * 2 + half
                            global_k = batch * (k // cw) + it * 64 + col
                            pos = (row, global_k)
                            assert pos not in seen
                            seen.add(pos)
        assert len(seen) == n * k
        assert seen == {(row, col) for row in range(n) for col in range(k)}
        for depth in (2, 3, 4):
            steps = 16
            queue, slots = list(range(depth)), list(range(depth))
            for it in range(steps):
                assert queue.pop(0) == slots[it % depth] == it
                slots[it % depth] = it + depth
                queue.append(it + depth)
            assert queue == list(range(steps, steps + depth))
        # Total useful copied payload/X is independentof CW partition count.
        assert ki * (2 * cw) * 1024 == 32 * k
        assert ki * cw * 32 * 64 == 32 * k
        shared = 3 * (cw * 4096 + cw * 32 * 128 + cw * 8)
        assert shared <= 220 * 1024
        print('PASS', 'CW', cw, 'weightcoverage', len(seen), 'shared_raw_bytes', shared)

if __name__ == '__main__':
    main()
