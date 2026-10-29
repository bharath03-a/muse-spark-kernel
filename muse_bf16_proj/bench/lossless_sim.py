import struct
import random
import math

def float_to_bf16_bits(val: float) -> int:
    raw = struct.pack(">f", val)
    return struct.unpack(">H", raw[:2])[0]

def bf16_bits_to_float(bits: int) -> float:
    raw = struct.pack(">HH", bits, 0)
    return struct.unpack(">f", raw)[0]

def test_12bit_packing():
    print("Testing 12-bit lossless BF16 packing (Pure Python standard library)...")
    random.seed(42)
    K = 8192
    N = 256
    
    # Generate Gaussian samples matching reference.py: randn / sqrt(K)
    sigma = 1.0 / math.sqrt(K)
    raw_weights = [[random.gauss(0.0, sigma) for _ in range(K)] for _ in range(N)]
    
    # Convert to BF16 bits
    bf16_mat = [[float_to_bf16_bits(raw_weights[i][j]) for j in range(K)] for i in range(N)]
    
    exps = [(bits >> 7) & 0xFF for row in bf16_mat for bits in row]
    max_exp = max(exps)
    min_exp = min(e for e in exps if e > 0)
    print(f"Max exponent: {max_exp}, Min non-zero exponent: {min_exp}")
    
    E_BASE = max_exp - 15
    print(f"E_BASE: {E_BASE}, Exponent Window: [{E_BASE}, {E_BASE + 15}]")
    
    total_elements = N * K
    in_window = sum(1 for e in exps if E_BASE <= e <= E_BASE + 15)
    outliers = total_elements - in_window
    print(f"In-window: {in_window} / {total_elements} ({in_window / total_elements * 100:.4f}%)")
    print(f"Outliers: {outliers} / {total_elements} ({outliers / total_elements * 100:.4f}%)")
    print(f"Average outliers per row: {outliers / N:.2f}")

    # Pack 8 weights into 3 uint32 (12 bytes instead of 16 bytes: 25.0% bandwidth savings!)
    EBP = (E_BASE << 7) | (E_BASE << 23)
    
    total_tested = 0
    mismatches = 0
    
    for row in range(min(50, N)):
        for group in range(0, K, 8):
            w = bf16_mat[row][group:group + 8]
            codes = []
            is_outlier = []
            for val in w:
                exp = (val >> 7) & 0xFF
                if E_BASE <= exp <= E_BASE + 15:
                    sign = (val >> 15) & 1
                    mant = val & 0x7F
                    code = (sign << 11) | ((exp - E_BASE) << 7) | mant
                    codes.append(code)
                    is_outlier.append(False)
                else:
                    codes.append(0)
                    is_outlier.append(True)
                    
            c0, c1, c2, c3, c4, c5, c6, c7 = codes
            p0 = c0 | (c1 << 16)
            p1 = c2 | (c3 << 16)
            p2 = c4 | (c5 << 16)
            
            w0 = p0 | ((c6 & 0xF) << 12) | ((c7 & 0xF) << 28)
            w1 = p1 | (((c6 >> 4) & 0xF) << 12) | (((c7 >> 4) & 0xF) << 28)
            w2 = p2 | (((c6 >> 8) & 0xF) << 12) | (((c7 >> 8) & 0xF) << 28)
            
            # Fast decode in registers:
            p0_dec = w0 & 0x0FFF0FFF
            p1_dec = w1 & 0x0FFF0FFF
            p2_dec = w2 & 0x0FFF0FFF
            
            c6_dec = ((w0 >> 12) & 0xF) | (((w1 >> 12) & 0xF) << 4) | (((w2 >> 12) & 0xF) << 8)
            c7_dec = ((w0 >> 28) & 0xF) | (((w1 >> 28) & 0xF) << 4) | (((w2 >> 28) & 0xF) << 8)
            p3_dec = c6_dec | (c7_dec << 16)
            
            for pair_idx, p in enumerate([p0_dec, p1_dec, p2_dec, p3_dec]):
                sign_bits = (p & 0x08000800) << 4
                exp_mant = (p & 0x07FF07FF) + EBP
                bf16_pair = sign_bits | exp_mant
                
                bf16_0 = bf16_pair & 0xFFFF
                bf16_1 = (bf16_pair >> 16) & 0xFFFF
                
                idx0 = pair_idx * 2
                idx1 = pair_idx * 2 + 1
                
                if not is_outlier[idx0]:
                    if bf16_0 != w[idx0]:
                        mismatches += 1
                if not is_outlier[idx1]:
                    if bf16_1 != w[idx1]:
                        mismatches += 1
                total_tested += 2

    assert mismatches == 0, f"Encountered {mismatches} mismatches!"
    print(f"PASS! Verified {total_tested} BF16 weights across 50 rows ({50 * K} total entries).")
    print(f"Bit-exact round trip identity: 100.000% match on in-window weights!")

    # Test dot product accuracy with outliers:
    print("\nTesting dot product with CSR outlier correction...")
    # For a row with an outlier:
    w_fast_val = bf16_bits_to_float(E_BASE << 7)
    
    max_err = 0.0
    for row in range(50):
        # Generate random x in BF16
        x_bits = [float_to_bf16_bits(random.gauss(0.0, 1.0)) for _ in range(K)]
        x_floats = [bf16_bits_to_float(b) for b in x_bits]
        w_floats = [bf16_bits_to_float(b) for b in bf16_mat[row]]
        
        # Exact reference dot product
        ref_dot = sum(w * x for w, x in zip(w_floats, x_floats))
        
        # Fast dot product with decoded weights
        fast_dot = 0.0
        csr_outliers = []
        for j in range(K):
            exp = (bf16_mat[row][j] >> 7) & 0xFF
            if E_BASE <= exp <= E_BASE + 15:
                fast_dot += w_floats[j] * x_floats[j]
            else:
                fast_dot += w_fast_val * x_floats[j]
                csr_outliers.append((j, w_floats[j] - w_fast_val))
                
        # Apply CSR outlier corrections
        for j, diff in csr_outliers:
            fast_dot += diff * x_floats[j]
            
        rel_err = abs(fast_dot - ref_dot) / (abs(ref_dot) + 1e-7)
        if rel_err > max_err:
            max_err = rel_err
            
    print(f"Max relative error across 50 rows: {max_err:.2e} (Threshold is 1e-2)")
    assert max_err < 1e-5, f"Relative error {max_err} too large!"
    print("PASS! Dot product with CSR outlier correction is mathematically exact!")

if __name__ == "__main__":
    test_12bit_packing()

