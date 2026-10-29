"""Vectorized prepare-only adaptive record packing; torch imported lazily.

Returns uint32 payload/metadata on w.device, with CPU codec's exact byte ABI.
No network, subprocesses, GPU launches or timing at module import.
"""
from __future__ import annotations


def pack_weights(w, base=None, force_raw=False, chunk_records=1024, order=None):
    import torch
    try:
        from .adaptive_record_codec import ORDER
    except ImportError:
        from adaptive_record_codec import ORDER

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
    ends = lengths.cumsum(dim=0)
    starts = ends - lengths
    total = int(ends[-1])
    if total // 4 > 0x7fffffff:
        raise ValueError("payload exceeds metadata offset range")
    meta = ((starts // 4) | (raw.to(torch.int64) << 31)).to(torch.int32).view(torch.uint32)
    payload = torch.empty(total, dtype=torch.int32, device=w.device)
    fragment_indices = ORDER if order is None else list(order)
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


def cpu_selftest():
    """Byte-for-byte compare torch and stdlib packers, CPU only."""
    import torch
    try:
        from .adaptive_record_codec import pack_records, unpack_records
    except ImportError:
        from adaptive_record_codec import pack_records, unpack_records
    from array import array
    words = torch.arange(4096, dtype=torch.int32).reshape(32, 128)
    words = ((words & 1) << 15) | ((56 + words % 8) << 8) | (words % 256)
    words[0, 0] = 0x2000  # one raw record; safely rebases
    w = words.to(torch.int16).view(torch.bfloat16)
    tile_words = words.reshape(2, 16, 2, 64).permute(0, 2, 1, 3).reshape(-1).tolist()
    for force_raw in (False, True):
        for chunk in (1, 3, 1024):
            expected = pack_records(tile_words, base=56, rebase_raw=True, force_raw=force_raw)
            actual = pack_weights(w, base=56, force_raw=force_raw, chunk_records=chunk)
            payload_bytes = bytes(actual['payload'].view(torch.uint8).tolist())
            metadata = actual['metadata'].to(torch.int64).tolist()
            assert payload_bytes == expected['payload']
            assert metadata == expected['metadata']
            assert actual['bits_per_weight'] == expected['bits_per_weight']
            assert unpack_records(expected) == array('H', tile_words)
    custom_order = list(reversed(range(1024)))
    expected = pack_records(tile_words, base=56, rebase_raw=True, order=custom_order)
    actual = pack_weights(w, base=56, order=custom_order)
    assert bytes(actual['payload'].view(torch.uint8).tolist()) == expected['payload']
    assert unpack_records(expected) == array('H', tile_words)
    # Nontrivial rebasing and automatic window selection.
    w = (torch.randn((32, 128), generator=torch.Generator().manual_seed(5151)) / 90).bfloat16()
    actual = pack_weights(w, force_raw=True, chunk_records=1)
    tile_words = (w.view(torch.int16).to(torch.int32) & 65535).reshape(2, 16, 2, 64).permute(0, 2, 1, 3).reshape(-1).tolist()
    expected = pack_records(tile_words, base=actual['base'], rebase_raw=True, force_raw=True)
    assert bytes(actual['payload'].view(torch.uint8).tolist()) == expected['payload']
    bad = w.clone()
    bad[0, 0] = float('inf')
    try:
        pack_weights(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite rebase must reject")
    zeros = pack_weights(torch.zeros((16, 64), dtype=torch.bfloat16))
    assert zeros['raw_records'] == 1
    assert not bool(zeros['payload'].to(torch.int64).any())
    return {"passed": True, "device": "cpu", "tests": "mixed/raw/chunk/rebase/invalid"}


if __name__ == '__main__':
    print(cpu_selftest())
