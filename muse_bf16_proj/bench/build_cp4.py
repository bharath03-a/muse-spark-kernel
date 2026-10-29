"""Assemble Checkpoint 4 (cp4_gluon_wgmma_v3.py) combining all verified Round 3 winners."""
import re

with open("submission.py") as f:
    sub = f.read()

with open("bench/exp_cuda_wgmma_r3.py") as f:
    cuda_r3 = f.read()

# Extract C++ source from exp_cuda_wgmma_r3.py (between SRC = r\"\"\" and \"\"\")
m_src = re.search(r'SRC = r"""(.*?)"""', cuda_r3, re.DOTALL)
assert m_src, "Failed to extract C++ source from exp_cuda_wgmma_r3.py"
wgmma_cpp_body = m_src.group(1)

# Notice we also keep row_gemv from submission.py for m=1 on unpooler_slice and router m=1
m_row_gemv = re.search(r'(template <bool F32, int R, int MT, int PER>.*?void launch_gemv\(.*?\n\})', sub, re.DOTALL)
assert m_row_gemv, "Failed to extract row_gemv from submission.py"
row_gemv_code = m_row_gemv.group(1)

print("Extracted C++ components successfully!")
