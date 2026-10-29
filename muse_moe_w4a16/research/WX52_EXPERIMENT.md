# Strict WXG2 v52 experiment

The isolated candidate ports only CUDA_X from public commit47103fd's v52. All other CUDA bodies match strict grid baseline. Build: `research/build_latest_wx.py`; standalone: `research_moe_latest_wx_strict.py`. Preserve original scored default.

## Precision and scheduling

Scaled gate/up round to BF16 before FP32 SiLU/product and final BF16 activation. Completed W2 accumulation times that expert's global scale rounds to BF16 before effective normweight, sum of squares, normalization and route combination. Effective normweight keeps strict baseline BF16 preparation. No partial-K rounding introduced. Prefill specialist independently reviewed math and alternating-bank lifetime. Final WGMMA wait remains before epilogue.

RPW2/WXG2 uses two alternating A banks and four 64-K chunks per slot. CPU sm90a compile: REG168, STACK0, LOCAL0;73.7s. No GPU compilation.

## H200 result

One bounded run,20 cases with varied per-expert global scales(w13 .7..1.3,w2 1.4...6). All40 baseline/candidate reference comparisons finite and within tolerance. All20 graph replay stability checks pass. Each comparison uses24 balanced AB/BA samples. Speedup >1 favors WX.

|M|seed|routing|baseline us|WX us|paired speedup|95% CI|
|---:|---:|---|---:|---:|---:|---|
|32|7|normal|184.69|195.33|0.9475|0.9333–0.9617|
|32|19|normal|182.94|191.71|0.9558|0.9414–0.9701|
|32|41|skewed|147.39|100.16|1.4748|1.4474–1.4970|
|128|7|normal|261.58|273.97|0.9520|0.9377–0.9663|
|128|19|normal|230.74|248.80|0.9263|0.9124–0.9402|
|128|41|skewed|163.36|133.73|1.2217|1.2034–1.2400|
|256|7|normal|307.31|300.83|1.0220|1.0066–1.0373|
|256|19|normal|316.32|310.56|1.0259|1.0105–1.0413|
|256|41|skewed|290.85|172.30|1.6890|1.6637–1.7144|
|17|71|skewed|144.37|144.27|1.0030|0.9880–1.0199|
|33|71|skewed|195.60|98.93|1.9822|1.9525–2.0119|
|63|71|skewed|294.53|103.10|2.8480|2.8053–2.8907|
|65|71|skewed|161.63|103.42|1.5666|1.5431–1.5901|
|127|71|skewed|193.28|133.31|1.4497|1.4280–1.4715|
|257|71|skewed|334.77|203.63|1.6437|1.6191–1.6684|
|64|7|normal|212.40|230.14|0.9200|0.9062–0.9338|
|64|41|skewed|291.55|102.03|2.8576|2.7863–2.9005|
|96|113|normal|257.23|273.39|0.9399|0.9258–0.9540|
|192|113|skewed|254.02|168.19|1.5163|1.4936–1.5391|
|384|113|skewed|585.65|332.29|1.7633|1.7368–1.7897|

Broad M32..512 dispatch is rejected: normal M32/M128 lose5–7%, M64 loses8%, M96 loses6%. Normal M256 gains2–3%, while skewed cases gain22–186%; routing concentration materially changes crossover. Do not choose a normal/skewed flag or seed-based dispatch. Existing provisional research-only dispatch must not be copied to production.

Whole decoder SOFT changes attention-key distribution; agentic changes context lengths. Actual routed work comes from sigmoid router rows+bias on mean/low-rank residuals, with judge-chosen router seed targeting router_load and router_distinct. Standalone skewed synthetic data is not evidence that whole SOFT/agentic routes share the same histogram. Next bounded experiment should reuse actual whole frontend outputs/topk weights, plus fresh M256 normal seeds, before any dispatch promotion.

A future GPU-side route-dependent dispatch could use the existing routing histogram, but must include its cost, preserve full routing, avoid host item synchronization, and account for unequal CTA paths/barriers. Prefer proving a shape-only region first.

Run https://modal.com/apps/<workspace>/main/ap-ELPiJ6icjAIMpjDs3butTo ; local log `/tmp/moe_wx52_gpu.log`; raw samples/environment in `profiles/research_wx52_gpu.json`.

SOURCE_SHA256 baseline b8d304a841189a8e382f6a8af402b1bdbdbbcb43167c9a70578aa66172912a9f
SOURCE_SHA256 research_moe_latest_wx_strict fee01dd845016a961c9edad8166edb3ce381b7806680777540213da90c07343e
PERSISTED_REPORT /cache/moe_probe_reports/1793148947801125723.json
