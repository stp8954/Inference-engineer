# Inference Engineering — Philip Kiely

- **Type:** Book (Baseten Books, 2026). ISBN 979-8-9943597-2-3. Content cutoff: January 2026.
- **Raw source:** `Inference Engineering-1.pdf` in Drive folder `inference-engineer-inbox` (file id `1tQ2ulGJbAD-Vfg2oRED9sHCWJl1pKUU7`)
- **Ingested:** 2026-08-22
- **Coverage:** COMPLETE for main text (Ch 0–7, pp. 1–208): Ch 0–3.2 via Drive extraction, Ch 3.2–4 via part-1 PDF, Ch 5–7 via part-2 PDF (all 2026-08-22). **Pending:** Appendices A–B only (glossary + recommended reading — low priority; App B's reading list is worth mining for the series' "recommended reading" page someday).

## Summary

A practitioner's map of production inference from Baseten's lead DevRel. Frames inference as three layers: runtime (single-instance model performance), infrastructure (autoscaling → multi-region capacity → unified global compute pool), and tooling (developer experience between "black box API" and raw compute). Chapter 2 is the strongest ingested material: LLM mechanics (tokenization → prefill/decode → logits → sampling), architecture parsing (config.json, e.g. Qwen3MoeForCausalLM), MoE routing granularity, and a worked arithmetic-intensity example proving decode is memory-bound. Chapter 1 contributes the product-side framing (shared vs. dedicated inference, online vs. offline, TTFT/TPS/ITL definitions, latency percentiles). Useful counterpoint source for the series: it's breadth-first where the series is depth-first.

## Wiki pages touched

- concepts/prefill-decode.md (roofline framing, worked example)
- concepts/kv-cache.md (prefill builds it, VRAM headroom rule)
- concepts/attention-kernels.md (FlashAttention/PagedAttention framing, attention variants)
- concepts/moe-inference.md (routing granularity, batch-vs-local sparsity)
- concepts/inference-metrics.md (new page)
- claims/decode-bandwidth-ceilings.md (H100 ops:byte)
- claims/hardware-specs.md (new page)

## Wiki pages touched (part-2 ingest, Ch 5–7)

- concepts/quantization.md, speculative-decoding.md, prefix-caching.md, kv-cache.md, disaggregated-serving.md, continuous-batching.md, attention-kernels.md, moe-inference.md
- concepts/model-parallelism.md (new), concepts/production-operations.md (new)
- claims/technique-performance.md (new), claims/hardware-specs.md
- entities/nvidia-dynamo.md, tensorrt-llm.md, vllm.md, sglang.md

## Wiki pages touched (part-1 ingest, Ch 3.2–4)

- concepts/inference-engines.md (new), concepts/hardware-landscape.md (new)
- concepts/inference-metrics.md (benchmarking + profiling methodology), concepts/attention-kernels.md (kernel ecosystem)
- claims/hardware-specs.md (GPU spec table, interconnects, MIG, architecture cadence)
- entities/vllm.md, sglang.md, tensorrt-llm.md, nvidia-dynamo.md (histories, engine comparison)
