"""CPU check of zb_kernel's stream-K partition (mirrors the device integer math)."""
import sys


def segs(c, G, RG, U):
    T = RG * U
    s_c, e_c = c * T // G, (c + 1) * T // G
    rf, rl = s_c // U, (e_c - 1) // U
    out = []
    for kind in range(3):
        for i in range(3):
            r = rf + i
            if r <= rl:
                a, b = max(s_c, r * U), min(e_c, (r + 1) * U)
                ln = b - a
                cut_prev, cut_next = a > r * U, b < (r + 1) * U
                k = (2 if 2 * ln > U else 0) if cut_prev else (2 if 2 * ln >= U else 0) if cut_next else 1
                if k == kind:
                    out.append((r, a, b, k))
    assert rl - rf + 1 <= 3
    return out


def check(G, RG, U, W):
    cover = [0] * (RG * U)
    fin, prod = {}, {}
    for c in range(G):
        sg = segs(c, G, RG, U)
        assert len(sg) <= 3
        nprod = sum(1 for s in sg if s[3] == 0)
        nfin = sum(1 for s in sg if s[3] != 0)
        assert nprod <= 2 and nfin <= 2, (c, sg)
        assert all(sg[i][3] <= sg[i + 1][3] for i in range(len(sg) - 1))
        for r, a, b, k in sg:
            for w in range(W):
                st = a + w * (b - a) // W
                en = a + (w + 1) * (b - a) // W
                for u in range(st, en):
                    cover[u] += 1
            if k == 0:
                assert r not in prod
                prod[r] = c
            else:
                assert r not in fin
                fin[r] = (c, k)
    assert all(v == 1 for v in cover)
    assert sorted(fin) == list(range(RG))
    for r, (c, k) in fin.items():
        assert (k == 2) == (r in prod), r
    print(f"ok G={G} RG={RG} U={U} W={W} shared={len(prod)}")


if __name__ == "__main__":
    for G in (132, 120, 100, 73, 144):
        for W in (8, 16):
            check(G, 144, 128, W)
    sys.exit(0)
