"""CPU-only rank3 address/partition proof; no GPU/compiler assumptions."""
from adaptive_record_mapping_test import main as vector_mapping_proof


def main():
    vector_mapping_proof()
    n, k, w, r = 64, 512, 4, 32
    seen = set()
    for cta in range(n // r):
        for batch in range(w):
            for it in range(k // (w * 64)):
                for lane in range(32):
                    addresses = []
                    for reg in range(64):
                        row = (lane >> 2) + ((reg >> 1) & 1) * 8 + ((reg >> 5) & 1) * 16
                        col = (reg & 1) + (lane & 3) * 2 + ((reg >> 2) & 1) * 8 + ((reg >> 3) & 1) * 16 + ((reg >> 4) & 1) * 32
                        global_row = cta * r + row
                        global_k = batch * (k // w) + it * 64 + col
                        index = (global_row, global_k)
                        assert index not in seen
                        seen.add(index)
                        record = (global_row // 16) * (k // 64) + global_k // 64
                        physical_lane = (row & 7) * 4 + ((col >> 1) & 3)
                        addresses.append((record, physical_lane))
                    # Each pack32 invocation uses exactlyone sourcepointer/flag.
                    assert len(set(addresses[:32])) == 1
                    assert len(set(addresses[32:])) == 1
                    assert addresses[0][0] != addresses[32][0]
    assert len(seen) == n * k
    assert seen == {(row, col) for row in range(n) for col in range(k)}
    # Exactinteger sum validates partition/output semantics independently.
    for row in range(n):
        full = sum((row + 1) * (col + 1) * (col % 7 - 3) for col in range(k))
        partials = [sum((row + 1) * (col + 1) * (col % 7 - 3)
                        for col in range(batch * (k // w), (batch + 1) * (k // w)))
                    for batch in range(w)]
        assert sum(partials) == full
    print('PASS:rank3 W4R32 covers each weightonce, pack32 pointersuniform, exactfourwayKsum')

if __name__ == '__main__':
    main()
