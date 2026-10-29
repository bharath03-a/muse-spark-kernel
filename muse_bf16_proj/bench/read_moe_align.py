with open("/sgl-workspace/sglang-muse-spark/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py") as f:
    lines = f.readlines()
for i in range(min(50, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
