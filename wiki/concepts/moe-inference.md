# moe-inference

## What it is
Serving mixture-of-experts models - expert parallelism, all-to-all, load balancing.

## Key numbers
- Qwen3-235B-A22B: 22B of 235B params active per token; router picks 8 of 128 experts at each of 94 layers, per token (Kiely §2.2.4). [sourced]
- Key serving asymmetry: MoE sparsity is real at batch 1 (local inference), but in batched serving different requests activate different experts — expect nearly all params hot unless doing large-scale expert parallelism. [sourced]

Expert Parallelism (Kiely §5.4.2): whole experts sharded per GPU (128 experts at EP8 → 16/GPU); tiny router replicated on every GPU; communication is token-passing between experts, not per-layer all-reduce, so EP scales across nodes on slower interconnects. EP buys throughput, not per-token latency; common to mix TP (attention) + EP (MoE layers). Multi-node MoE: EP16 for throughput vs TP8PP2 for per-user latency.

## Open questions
- DeepSeek-V3 / expert load-balancing papers for Week 14 primary sources.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §2.2.4.

## Series mapping
- Week 14
