# nvidia-dynamo

## What it is
NVIDIA's open orchestration layer for disaggregated serving, announced GTC March 2025 (Apache 2.0). Backend-agnostic over vLLM/SGLang/TRT-LLM: prefix-match KV routing, disaggregated prefill/decode with independent scaling, multi-node parallelism (usually EP), and an SLA-based planner that autoscales prefill/decode workers against TTFT/TPS targets. Suits trillion-param models at high concurrency; below that scale it's unnecessary overhead.

## Timeline
- 2025: announced as NVIDIA's open orchestration layer for large-scale distributed serving.
- Capabilities per Kiely §5.3/§5.5/§7.2: KVBM (KV Block Manager — moves KV blocks across VRAM/CPU/SSD/network tiers), NIXL-based KV transfer between prefill and decode engines (incl. transposing between differing TP layouts), conditional disaggregation with configurable ISL thresholds and prefill queues, runtime-adjustable xPyD ratios, KV-cache-aware and LoRA-aware routing.

## Relation to the series
- Week 17

## Sources
- (pending first ingests)
