# v75 FP8 PV precision audit

Source: lee101/openkernels commit `47103fd9c1212868018eb615a4c6467e366e9ce0`, Apache-2.0. Candidate is `candidate_openkernels_v75.py`; the conservative FP16 PV control is `candidate_openkernels_v75_f16pv.py`. Their embedded CUDA is identical; Python dispatch selects explicitly compiled FP8/FP16 instances. Existing best submission is untouched.

## Baseline evidence

The task's SGLang baseline casts normalized Q to E4M3 before calling `sgl_kernel.flash_attn.flash_attn_with_kvcache`. K/V caches are already E4M3. Installed wrapper calls `torch.ops.sgl_kernel.fwd.default` in sglang-kernel0.4.9. CPU inspection located its actual binary at `/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so` (1,122,619,328bytes).

`modal_bf16pv_compile.py::inspect_fa3_binary` selects installed kernels by demangled QK head64/V64, FP8 element type, causal/varlen/non-TMA paged flags. Audit v5 found48 matching instantiations. Selected tile192x160x64,V64 has21 `QGMMA.*.F32.E4M3.E4M3` instructions, zero HGMMA,120 `F2FP.*.E4M3.F32` conversions. QK has output width160; PV has width64 and register P operands, for example:

```
QGMMA.64x64x32.F32.E4M3.E4M3 R104, R152, gdesc[UR12], R104
```

Thus the installed paged head64 baseline kernel family uses FP8 P/V tensor multiplication with FP32 accumulators. This is binary evidence from the installed image, not inference from a generic paper. The audit does not trace one exact runtime baseline launch; several scheduling specializations exist. Raw CPU audit: `/tmp/attention_installed_fa3_binary_v5.log`.

Primary source cross-check: [SGLang FA3 mainloop](https://github.com/sgl-project/sgl-flash-attn/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) selects PV operand type from the same Element as QK. [Its softmax](https://github.com/sgl-project/sgl-flash-attn/blob/main/hopper/softmax.h) uses offset8 for FP8, extending probabilities to[0,256]. These current sources help interpret the binary and are not asserted to be the installed version pin.

## Candidate semantics and scope

FP8-PV dispatch is selected only for full verification B1/16/32; standalone decode and S1K sliding paths retain their existing arithmetic. Q/K and cached V remain their original E4M3 inputs. P is E4M3, output BF16, accumulation FP32. Candidate passes the same rounded FP8 P to numerator and denominator tensor operations. It retains online running max and rescaling, all-key iteration, per-query causal/tail masks, current-token cache insertion, and final cache writes. No input-dependent CPU length reads or retained answers are introduced.

Candidate lazy-max scaling is not bitwise identical to FA3's max-offset policy. Therefore binary dtype evidence does not substitute for canonical numerical validation. The conservative version disables the FP8 flag while retaining S1K and other v75 scheduling improvements. Native dispatch flags are539230276 (B1) and537002308 (B16/32) for research versus2359364 and131396 for control.

## Validation

Broad cached H200 probe `ap-b02BaHv6batK19LtYGOlWR` passed171 canonical/finite checks, all24 ABBA comparisons, and4x200 fresh QKV/cache CUDA-graph replay cases. Full verify B16/B32 improved about6.8%; see `research_v75_fp8_results.json` for raw samples and CIs. Most broad correctness shapes use fallback dispatch, so they do not independently validate the new FP8 branch. The actual FP8 full B1/B16 stress cases passed canonical references every25replays and final.

A separate27-input targeted scored-branch validation is required before official checking: three seeds × B1/16/32 × flat/coherent/scaled-edge distributions, all full qlen8/page1/context80K. The runner asserts the native FP8 flag and reports per-case differences from FP16, finite output, canonical reference tolerance, and cache-write checks. Completed in app `ap-ApdExlLDgwo1XSrD7Oxhel`: all81 checks passed (27 inputs, each current/conservative/research). Every research call asserted var=-2 and the native FP8 flag; all27 F16/FP8 comparisons were nonzero, global relative L2 at most0.01981. In scaled-edgeB32 seed1471, research globalrelativeL2 was0.09135 against0.150, and worstrow0.1446 against the actual rowbound0.750 (5×globalbound); FP16 controlworstrow0.1385. The globalbound and rowbound are distinct. Cache entries pass canonical cache tolerance checks for every variant and input. See `research_v75_fp8_precision_results.json` for exact checks, differences, and dispatch; raw log `/tmp/attention_v75_fp8_targeted.log`. This supports an official check, rather than a claim of correctness on arbitrary untested distributions.
