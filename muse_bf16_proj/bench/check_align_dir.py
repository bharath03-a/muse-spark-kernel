import sglang.srt.layers.moe.fused_moe_triton as fmt

print([k for k in dir(fmt) if "align" in k.lower() or "moe" in k.lower()])
