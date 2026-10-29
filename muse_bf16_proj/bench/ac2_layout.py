"""Consumption-ordered payload placement for ac2 (lossless; kernel addressing unchanged)."""
import torch


def ac2_position(n, k, rw, split):
    """Physical rank of each record (tile order g*(K/64)+kb) in kernel consumption order."""
    nw, nr, kc = 4 * rw, 8 * rw, k // split
    ki = kc // 256
    g = torch.arange(n // 16).repeat_interleave(k // 64)
    kb = torch.arange(k // 64).repeat(n // 16)
    rowblock, lg = g // (2 * rw), g % (2 * rw)
    j, kbl = kb // (kc // 64), kb % (kc // 64)
    batch, step = kbl // ki, kbl % ki
    rgih, rowhalf = lg // rw, lg % rw
    r = rowhalf + rw * batch + nw * rgih
    return ((rowblock * split + j) * ki + step) * nr + r


def kernel_rids(n, k, rw, split):
    """rid sequence exactly as _ac2_issue computes it, in (cta, step, record) order."""
    nw, nr, kc = 4 * rw, 8 * rw, k // split
    ki = kc // 256
    out = []
    for pid_n in range(n // (32 * rw)):
        for pid_s in range(split):
            for step in range(ki):
                for r in range(nr):
                    rowhalf, batch, rgih = r % rw, (r // rw) % 4, r // nw
                    row = pid_n * 32 * rw + (rgih * rw + rowhalf) * 16
                    out.append((row // 16) * (k // 64) + pid_s * (kc // 64) + batch * ki + step)
    return torch.tensor(out)
