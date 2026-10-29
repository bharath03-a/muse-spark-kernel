"""CPU-only exact prototype of cp14 16x64 fragment records with raw fallback.

No torch dependency. Metadata uint32: bit31 raw flag; low31 payload offset
in 16-byte units. Every record starts at a 512-byte boundary. Compressed
records are1536B; raw records2048B. Input/output are row-major BF16 bit words.
Runtime kernels should use predicated loads and arithmetic select, not branch
on this flag. This prototype measures representation only, not GPU speed.
"""
from __future__ import annotations
from array import array
from collections import Counter
import math
import random
import struct
import time

RAW_FLAG = 1 << 31
OFFSET_MASK = RAW_FLAG - 1
RECORD_WEIGHTS = 1024


def fragment_order():
    """cp14 arr(...): two planes,32lanes,16bytes per lane."""
    order = []
    for p in range(2):
        for lane in range(32):
            g, t = divmod(lane, 4)
            for c in range(2):
                for jl in range(2):
                    for h in range(2):
                        for eo in range(2):
                            row = h * 8 + g
                            k = ((((c * 4 + t) * 2 + jl) * 2 + p) * 2 + eo)
                            order.append(row * 64 + k)
    assert sorted(order) == list(range(RECORD_WEIGHTS))
    return order


ORDER = fragment_order()


def choose_base(words):
    hist = Counter((int(w) >> 8) & 127 for w in words)
    return max(range(121), key=lambda b: sum(hist[e] for e in range(b, b + 8)))


def rebase_word(word, exponent_shift):
    """Exact multiplication by2**shift for normal BF16; reject lossy cases.

    Signed zero survives. Reject subnormals, infinities/NaNs, and shifts into
    subnormal/overflow ranges so optimized runtime can fall back by layer.
    """
    exponent = (word >> 7) & 255
    if (word & 0x7fff) == 0:
        return word
    if exponent == 0 or exponent == 255 or not 1 <= exponent + exponent_shift <= 254:
        raise ValueError("raw BF16 cannot be exactly rebased as normal")
    return (word & 0x807f) | ((exponent + exponent_shift) << 7)


def pack_records(words, base=None, rebase_raw=False, force_raw=False, order=None):
    if len(words) % RECORD_WEIGHTS:
        raise ValueError("expected complete16x64records")
    if any(not 0 <= int(w) <= 65535 for w in words):
        raise ValueError("expected BF16 bitwords")
    if base is None:
        base = choose_base(words)
    if not 0 <= base <= 120:
        raise ValueError("invalid exponent window base")
    # cp14 decode emits exponent112..127 and rescales by2**(2base-112).
    # The raw path must emit that SAME rebase, never original-valued operands.
    raw_shift = 112 - 2 * base if rebase_raw else 0
    fragment_indices = ORDER if order is None else list(order)
    if sorted(fragment_indices) != list(range(RECORD_WEIGHTS)):
        raise ValueError("fragment order must permute1024record indices")
    payload, metadata = bytearray(), []
    raw_count = 0
    for start in range(0, len(words), RECORD_WEIGHTS):
        tile = words[start:start + RECORD_WEIGHTS]
        raw = force_raw or any(
            not base <= ((int(v) >> 8) & 127) < base + 8 or
            (rebase_raw and ((int(v) >> 7) & 255) in (0, 255))
            for v in tile)
        offset = len(payload) // 16
        if offset > OFFSET_MASK:
            raise ValueError("payload exceeds metadata offset range")
        metadata.append(offset | (RAW_FLAG if raw else 0))
        ordered = [int(tile[i]) for i in fragment_indices]
        if raw:
            raw_count += 1
            # Four coalesced512Bplanes;16B/lane is a0,a1,a2,a3.
            vals = [ordered[p * 512 + lane * 16 + j * 4 + q]
                    for j in range(4) for lane in range(32)
                    for p in range(2) for q in range(4)]
            if rebase_raw:
                vals = [rebase_word(v, raw_shift) for v in vals]
            payload.extend(struct.pack('<1024H', *vals))
        else:
            low = bytes(v & 255 for v in ordered)
            nib = [((v >> 15) << 3) | (((v >> 8) & 127) - base) for v in ordered]
            codes = bytearray(512)
            for lane in range(32):
                for group in range(4):
                    at = lane * 16 + group * 4
                    for i in range(4):
                        nw, nx = nib[at + i], nib[512 + at + i]
                        nx_next = nib[512 + at + (i + 1) % 4]
                        codes[at + i] = ((nw & 8) << 4) | (nw & 7) | (nx & 8) | ((nx_next & 7) << 4)
            payload.extend(low + codes)
    return {"payload": bytes(payload), "metadata": metadata, "base": base,
            "records": len(metadata), "raw_records": raw_count,
            "raw_exponent_shift": raw_shift,
            "order": fragment_indices,
            "bits_per_weight": 8 * (len(payload) + 4 * len(metadata)) / max(1, len(words))}


def unpack_records(packed):
    payload, base = packed["payload"], packed["base"]
    words = array('H')
    for meta in packed["metadata"]:
        pos = (meta & OFFSET_MASK) * 16
        if meta & RAW_FLAG:
            vals = struct.unpack_from('<1024H', payload, pos)
            shift = packed.get('raw_exponent_shift', 0)
            if shift:
                vals = [rebase_word(v, -shift) for v in vals]
            ordered = [0] * RECORD_WEIGHTS
            for lane in range(32):
                for j in range(4):
                    for p in range(2):
                        for q in range(4):
                            ordered[p * 512 + lane * 16 + j * 4 + q] = vals[j * 256 + lane * 8 + p * 4 + q]
        else:
            low = payload[pos:pos + 1024]
            codes = payload[pos + 1024:pos + 1536]
            ordered = [0] * RECORD_WEIGHTS
            for lane in range(32):
                for group in range(4):
                    at = lane * 16 + group * 4
                    for i in range(4):
                        c = codes[at + i]
                        hi0 = ((c & 7) + base) | (c & 128)
                        hi1 = (((codes[at + (i - 1) % 4] >> 4) & 7) + base) | ((c & 8) << 4)
                        ordered[at + i] = low[at + i] | (hi0 << 8)
                        ordered[512 + at + i] = low[512 + at + i] | (hi1 << 8)
        tile = [0] * RECORD_WEIGHTS
        for index, value in zip(packed.get('order', ORDER), ordered):
            tile[index] = value
        words.extend(tile)
    return words


def selftest():
    # Exhaustively covers every BF16 bit pattern, including signed zero/NaNs.
    for values, base in ((array('H', range(65536)), 56),
                         (array('H', (0x3800 + (i % 2048) for i in range(4096))), 56)):
        packed = pack_records(values, base)
        assert unpack_records(packed) == values
    # Compressed record with both signs, all valid codes and low bytes.
    values = array('H', (((i & 1) << 15) | ((56 + i % 8) << 8) | (i % 256) for i in range(4096)))
    packed = pack_records(values, 56)
    assert packed['raw_records'] == 0
    assert unpack_records(packed) == values
    normal_values = array('H', (0x3800 + (i % 2048) for i in range(4096)))
    rebased = pack_records(normal_values, 58, rebase_raw=True)
    assert unpack_records(rebased) == normal_values
    try:
        pack_records(array('H', [0x7f80] * 1024), 56, rebase_raw=True)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite optimized rebase must reject")
    values[1024] = 0x7f80
    packed = pack_records(values, 56)
    assert packed['raw_records'] == 1
    assert unpack_records(packed) == values


def synthetic_study(records=1024, seed=5151):
    rng = random.Random(seed)
    values = array('H')
    for _ in range(records * RECORD_WEIGHTS):
        f = struct.unpack('<I', struct.pack('<f', rng.gauss(0, 1) / math.sqrt(8192)))[0]
        values.append(((f + 0x7fff + ((f >> 16) & 1)) >> 16) & 65535)
    start = time.perf_counter()
    packed = pack_records(values, rebase_raw=True)
    assert unpack_records(packed) == values
    base = packed['base']
    escapes = sum(not base <= ((v >> 8) & 127) < base + 8 for v in values)
    return {"weights": len(values), "seed": seed, "base": base,
            "escapes": escapes, "escape_fraction": escapes / len(values),
            "raw_records": packed['raw_records'], "records": records,
            "raw_record_fraction": packed['raw_records'] / records,
            "bits_per_weight": packed['bits_per_weight'],
            "payload_and_metadata_bytes": len(packed['payload']) + 4 * records,
            "roundtrip": True, "cpu_pack_unpack_seconds": time.perf_counter() - start}


if __name__ == '__main__':
    import json
    selftest()
    print(json.dumps(synthetic_study(), indent=2))
