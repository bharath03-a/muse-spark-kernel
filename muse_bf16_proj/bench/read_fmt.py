with open("/opt/sglang/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton.py") as f:
    lines = f.readlines()
for i in range(min(60, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
