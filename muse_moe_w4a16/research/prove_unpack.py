"""CPU exhaustive check of v48's FP4-to-BF16 scaled integer representation.
No GPU, no SGLang imports. Validates arithmetic identity, not lane layout/performance.
"""
import struct

def bf16(bits):
    return struct.unpack('>f', struct.pack('>I', (bits & 65535) << 16))[0]

GRID = [0., .5, 1., 1.5, 2., 3., 4., 6.]
count = 0
for code in range(16):
    a_bits = ((code & 7) << 6) | ((code & 8) << 12)
    a = bf16(a_bits)
    for sf in range(0x08, 0x7f):  # normal finite positive E4M3, incl max448
        scale = 2. ** ((sf >> 3) - 7) * (1. + (sf & 7) / 8.)
        s = bf16(0x7700 + sf * 16)
        value = (-1 if code & 8 else 1) * GRID[code & 7]
        assert a * s == value * scale / 256., (code, sf, a * s, value * scale / 256.)
        count += 1
print(f'PASS {count} code/normal-scale combinations; global scale must restore x256')
# Negative scale is losslessly folded into the FP4 sign during prepare.
# Zero scale maps codes to0 and substitutes normal minimumscale; still exactly0.
# E4M3 subnormal scales are explicitly rejected by upstream preparation.
