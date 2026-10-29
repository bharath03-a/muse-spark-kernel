# okc: problem=muse_prefill_layer
"""Probe BF16 MoE runners and NVFP4 dequant speed on H200."""
import sys
import inspect
import torch
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    device = weights["w13"].device
    print("=== PROBING BF16 MOE RUNNERS ON H200 ===", flush=True)
    
    # 1. Check CutlassBf16Runner
    try:
        runner_cls = getattr(fi_moe, "CutlassBf16Runner", None)
        cfg_cls = getattr(fi_moe, "CutlassBf16Config", None)
        print("CutlassBf16Runner:", runner_cls, flush=True)
        print("CutlassBf16Config:", cfg_cls, flush=True)
        if cfg_cls:
            print("CutlassBf16Config.supported(90):", cfg_cls.supported(90), flush=True)
            print("CutlassBf16Config.prepare_weights sig:", inspect.signature(cfg_cls.prepare_weights), flush=True)
    except Exception as e:
        print("CutlassBf16 error:", e, flush=True)

    # 2. Check TrtllmBf16RoutedRunner
    try:
        trt_runner = getattr(fi_moe, "TrtllmBf16RoutedRunner", None)
        trt_cfg = getattr(fi_moe, "TrtllmBf16Config", None)
        print("TrtllmBf16RoutedRunner:", trt_runner, flush=True)
        print("TrtllmBf16Config:", trt_cfg, flush=True)
        if trt_cfg:
            print("TrtllmBf16Config methods:", [m for m in dir(trt_cfg) if not m.startswith("_")], flush=True)
    except Exception as e:
        print("TrtllmBf16 error:", e, flush=True)

    # 3. Check cutlass_fused_moe
    try:
        cfm = getattr(fi_moe, "cutlass_fused_moe", None)
        print("cutlass_fused_moe callable:", cfm is not None, flush=True)
        if cfm:
            print("cutlass_fused_moe sig:", inspect.signature(cfm), flush=True)
    except Exception as e:
        print("cutlass_fused_moe error:", e, flush=True)

    # 4. Check flashinfer.ops.grouped_mm_bf16
    try:
        import flashinfer
        gmm = getattr(flashinfer, "grouped_mm_bf16", None) or getattr(flashinfer.ops, "grouped_mm_bf16", None)
        print("grouped_mm_bf16:", gmm, flush=True)
        if gmm:
            print("grouped_mm_bf16 sig:", inspect.signature(gmm), flush=True)
    except Exception as e:
        print("grouped_mm_bf16 error:", e, flush=True)

    # 5. Measure NVFP4 -> BF16 dequant speed
    # Shape of 32 experts: w13 is [32, 8192, 2048] uint8
    w13 = weights["w13"]
    w13_sf = weights["w13_sf"]
    w13_gs = weights["w13_gs"]
    print(f"w13 shape: {w13.shape}, dtype: {w13.dtype}", flush=True)
    print(f"w13_sf shape: {w13_sf.shape}, dtype: {w13_sf.dtype}", flush=True)
    print(f"w13_gs shape: {w13_gs.shape}, dtype: {w13_gs.dtype}", flush=True)
    
    # Simple PyTorch dequant time test
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    # Time 10 iterations of fast unpack & scale
    # w13: [32, 8192, 2048] uint8 -> 536 MB
    grid_lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=torch.bfloat16)
    
    start_ev.record()
    # Unpack low and high nibbles
    lo = (w13 & 0x0F).long()
    hi = ((w13 >> 4) & 0x0F).long()
    # lut
    v_lo = grid_lut[lo & 0x7]
    v_lo = torch.where((lo & 0x8) != 0, -v_lo, v_lo)
    end_ev.record()
    torch.cuda.synchronize()
    elapsed = start_ev.elapsed_time(end_ev)
    print(f"Pytorch unpack 536 MB took: {elapsed:.2f} ms", flush=True)

    sys.stdout.flush()
    return {}

def kernel_fn(data, state):
    stream = data["stream"]
    hidden = data["attn_input"]
    # Return valid distinct topk_ids to satisfy harness
    T = stream.shape[0]
    topk_ids = torch.arange(8, dtype=torch.int32, device=stream.device).unsqueeze(0).expand(T, 8).contiguous()
    return hidden, stream, topk_ids
