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


def _ac2_pack(w, position, base=None, force_raw=False, chunk_records=1024, order=None):
    import torch

    if w.dtype != torch.bfloat16 or w.ndim != 2:
        raise ValueError("expected 2D BF16 weights")
    n, k = w.shape
    if n <= 0 or k <= 0 or n % 16 or k % 64 or chunk_records <= 0:
        raise ValueError("weights require N%16=0,K%64=0 and positive chunk_records")
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if base is None:
        hist = torch.bincount(((bits >> 8) & 127).flatten().to(torch.int64), minlength=128)
        base = int(hist.unfold(0, 8, 1).sum(dim=1).argmax())
    if not isinstance(base, int) or not 0 <= base <= 120:
        raise ValueError("invalid exponent window base")
    tiles = bits.reshape(n // 16, 16, k // 64, 64).permute(0, 2, 1, 3).reshape(-1, 1024)
    upper = (tiles >> 8) & 127
    raw = ((upper < base) | (upper >= base + 8)).any(dim=1)
    original_exp = (tiles >> 7) & 255
    raw |= ((original_exp == 0) | (original_exp == 255)).any(dim=1)
    del original_exp
    if force_raw:
        raw = torch.ones_like(raw)
    del upper
    shift = 112 - 2 * base
    # Verify raw rebase without a data-dependent kernel_fn path. All operations
    # here are untimed prepare operations, including scalar synchronization.
    exponents = (tiles >> 7) & 255
    zero = (tiles & 32767) == 0
    invalid = (~zero) & ((exponents == 0) | (exponents == 255) |
                         (exponents + shift < 1) | (exponents + shift > 254))
    if bool((invalid & raw[:, None]).any()):
        raise ValueError("raw weights cannot be exactly rebased as normal BF16")
    del exponents, zero, invalid
    lengths = torch.where(raw, 512, 384).to(torch.int64)  # uint32 words
    position = position.to(device=w.device, dtype=torch.int64)
    phys = torch.empty_like(lengths)
    phys[position] = lengths
    starts = (phys.cumsum(dim=0) - phys)[position]
    ends = starts + lengths
    total = int(lengths.sum())
    if total // 4 > 0x7fffffff:
        raise ValueError("payload exceeds metadata offset range")
    meta = ((starts // 4) | (raw.to(torch.int64) << 31)).to(torch.int32).view(torch.uint32)
    payload = torch.empty(total, dtype=torch.int32, device=w.device)
    fragment_indices = _ac_order() if order is None else list(order)
    if sorted(fragment_indices) != list(range(1024)):
        raise ValueError("fragment order must permute1024record indices")
    order_tensor = torch.tensor(fragment_indices, dtype=torch.int64, device=w.device)
    indices = torch.arange(512, dtype=torch.int64, device=w.device)
    for lo in range(0, tiles.shape[0], chunk_records):
        hi = min(lo + chunk_records, tiles.shape[0])
        ordered = tiles[lo:hi].index_select(1, order_tensor)
        count = hi - lo
        nib = ((ordered >> 15) << 3) | ((((ordered >> 8) & 127) - base) & 7)
        nw = nib[:, :512].reshape(count, 32, 4, 4)
        nx = nib[:, 512:].reshape(count, 32, 4, 4)
        codes = (((nw & 8) << 4) | (nw & 7) | (nx & 8) |
                 (torch.roll(nx & 7, -1, dims=-1) << 4)).reshape(count, 512)
        compressed_bytes = torch.cat((ordered & 255, codes), dim=1).reshape(count, 384, 4)
        comp = (compressed_bytes[:, :, 0] | (compressed_bytes[:, :, 1] << 8) |
                (compressed_bytes[:, :, 2] << 16) | (compressed_bytes[:, :, 3] << 24))
        exp = (ordered >> 7) & 255
        rebased = torch.where((ordered & 32767) == 0, ordered,
                              (ordered & 0x807f) | ((exp + shift) << 7))
        raw_bits = rebased.reshape(count, 2, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(count, 512, 2)
        raw_words = raw_bits[:, :, 0] | (raw_bits[:, :, 1] << 16)
        # Select one512word buffer, predicate its final128words for compressed
        # records, and scatter compactly; no worst-case per-record reservation.
        comp_pad = torch.cat((comp, torch.zeros((count, 128), dtype=torch.int32, device=w.device)), dim=1)
        data = torch.where(raw[lo:hi, None], raw_words, comp_pad)
        valid = indices[None, :] < lengths[lo:hi, None]
        positions = starts[lo:hi, None] + indices[None, :]
        payload[positions[valid]] = data[valid]
    return {"payload": payload.view(torch.uint32), "metadata": meta,
            "base": base, "scale": 2.0 ** (2 * base - 112),
            "n": n, "k": k, "records": tiles.shape[0],
            "raw_records": int(raw.sum()), "raw_exponent_shift": shift,
            "bits_per_weight": 8 * (total * 4 + meta.numel() * 4) / w.numel()}
