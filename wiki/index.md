# Wiki index

Updated with every ingest. One line per page.

## Concepts

- [prefill-decode](concepts/prefill-decode.md) — the two-phase workload; compute-bound vs. bandwidth-bound
- [kv-cache](concepts/kv-cache.md) — cached K/V projections; decode's memory constraint
- [continuous-batching](concepts/continuous-batching.md) — iteration-level scheduling (Orca lineage)
- [paged-attention](concepts/paged-attention.md) — block tables, fragmentation math, prefix sharing, block-size tuning
- [prefix-caching](concepts/prefix-caching.md) — prefix reuse, hashing/eviction, chunked prefill, P99 impact
- [quantization](concepts/quantization.md) — lower precision as a bandwidth optimization
- [speculative-decoding](concepts/speculative-decoding.md) — draft-and-verify, accept/reject proof, acceptance→speedup math
- [disaggregated-serving](concepts/disaggregated-serving.md) — split P/D fleets, KV transfer cost, P:D sizing
- [attention-kernels](concepts/attention-kernels.md) — FlashAttention lineage and decode kernels
- [moe-inference](concepts/moe-inference.md) — serving mixture-of-experts models
- [inference-metrics](concepts/inference-metrics.md) — TTFT/TPS/ITL, percentiles, end-to-end vs. inference-only
- [model-parallelism](concepts/model-parallelism.md) — TP/PP/EP/context parallelism, sizing math
- [production-operations](concepts/production-operations.md) — cold starts, autoscaling, routing, cost/TCO
- [inference-engines](concepts/inference-engines.md) — vLLM vs SGLang vs TensorRT-LLM, the software stack
- [hardware-landscape](concepts/hardware-landscape.md) — GPU generations, interconnects, MIG, alt accelerators, local
- [vllm-internals](concepts/vllm-internals.md) — anatomy of a step: scheduler, block pool, worker, the eight knobs
- [fine-tuning-for-inference](concepts/fine-tuning-for-inference.md) — LoRA/QLoRA, distillation, multi-LoRA serving
- [multimodal-inference](concepts/multimodal-inference.md) — token rates by modality, voice budgets, video KV, embodied loops
- [training-vs-inference](concepts/training-vs-inference.md) — teacher forcing, 3×/6× rule, regime asymmetry
- [attention-variants](concepts/attention-variants.md) — MHA/MQA/GQA/MLA + sliding window, linear attention, SSM/Mamba

## Entities

- [vllm](entities/vllm.md) — leading open engine; PagedAttention origin
- [sglang](entities/sglang.md) — open engine; RadixAttention
- [tensorrt-llm](entities/tensorrt-llm.md) — NVIDIA's engine
- [nvidia-dynamo](entities/nvidia-dynamo.md) — disaggregated serving orchestration
- [kvscope](entities/kvscope.md) — the series' companion profiler (ours)

## Claims

- [decode-bandwidth-ceilings](claims/decode-bandwidth-ceilings.md) — batch-1 tok/s ceilings from weights÷bandwidth
- [hardware-specs](claims/hardware-specs.md) — GPU spec numbers and sizing rules of thumb
- [technique-performance](claims/technique-performance.md) — measured/claimed gains per optimization technique
- [inference-economics](claims/inference-economics.md) — $/M formula, cost spectrum, case studies, API-vs-self-host
- [kv-compression](claims/kv-compression.md) — head/token compression ratios and retrieval-quality tradeoffs
- [optimization-ladder](claims/optimization-ladder.md) — measured 55× stack-up on one H100 (Week 30 benchmark target)

## Sources ingested

- [2026-08-22 Kiely, Inference Engineering](../sources/2026-08-22-kiely-inference-engineering.md) — book, Ch 0–7 complete (pending: appendices only)
- [2026-08-22 Vizuara, Workshop Guide](../sources/2026-08-22-vizuara-workshop-guide.md) — book, **complete** (Ch 0–26 + conclusion)
