# OWN rewrite notes
Learned (keel/LESSONS): per-route backend choice; 12-bit lossless weights streamed at small m; swap-AB mma with deep weight prefetch; row-group x split-K warps.
Implemented independently: (1) extended our 12-bit zip kernel to m17-32 (NT=4) with timed selection; (2) larger row tiles + cluster K-split; (3) register-direct kernel rd_kernel (own code, bench/own_ab.py + modal_own_v1.py harness).
Result: none beat the base end-to-end; the existing cwm/async paths already take pooler/unpooler m32. Remaining gap (10-20% at m16..256) not closed.
