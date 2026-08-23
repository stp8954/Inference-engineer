# Model parallelism

## What it is
Splitting a model across GPUs when weights + KV cache exceed one device. Three inference-relevant forms (Kiely §5.4): Tensor Parallelism (splits matmuls within each layer; needs per-layer all-reduce → intra-node only, over NVLink/NVSwitch; the default, improves per-user latency), Pipeline Parallelism (layers across GPUs; poor latency/utilization, use only across nodes), Expert Parallelism (MoE experts sharded per GPU; throughput-oriented, tolerates slower interconnects — see [moe-inference](moe-inference.md)). Context Parallelism (ring attention over context slices, weights replicated) is rare for LLMs but standard for video generation. Multi-node recipes: dense → TP8PP2; MoE → EP16 (throughput) vs TP8PP2 (latency). Advice: multi-node is a poor use of hardware unless the model truly requires it — prefer horizontal replicas or disaggregation.

**The AllReduce arithmetic (Vizuara Ch 16).** Tensor parallelism alternates column-parallel and row-parallel matmuls so each transformer block needs exactly **2 AllReduces** (one after attention, one after the MLP) — 160 per forward pass for an 80-layer model. At batch 32 a 32 MB activation AllReduce takes ~36 µs on NVLink (900 GB/s) but ~640 µs on InfiniBand (50 GB/s). That's **~5.7 ms per decode step intra-node versus ~100 ms cross-node** — the difference between a 15% ITL overhead and a broken deployment. This single calculation is why TP must stay inside one node.

Pipeline parallelism moves only stage-boundary activations (~8 MB per stage, ~160 µs over InfiniBand), so it *is* cross-node viable; its cost is the bubble, which matters far less at inference than in training (no backward pass). Expert parallelism does two all-to-alls per MoE layer, but only the top-K experts' tokens move, so it's much cheaper than TP's AllReduce and tolerates slower links. Sequence parallelism splits token-independent ops (LayerNorm, dropout, residuals); context parallelism splits the sequence axis for attention itself via ring passing — the only way to fit a 100K+ token KV cache, and the enabler for ~1M-token contexts.

**Recipes:** 70B dense → TP=8 on one node, then replicate. DeepSeek-V3 671B MoE → TP=2 × EP=16. Llama-3-405B dense (810 GB at FP16 > 640 GB node) → TP=8 × PP=2, or quantize.

## Key numbers
- Sizing: VRAM ≈ (precision_bits/8) × params_B × kv_allocation_factor; with factor 1.8, DeepSeek-V3.1 (671B) at FP8 ≈ 1,200 GB → 8×B200 node (1,440 GB). [sourced] — Kiely §5.4.
- FP8 rule of thumb: ~1 GB VRAM per 1B params (weights only). [sourced] — Kiely §5.4.
- KV cache commonly takes 80%+ of VRAM remaining after weights in production LLM serving. [sourced] — Kiely §5.4.

Why TP must stay intra-node (Vizuara §6.8, worked): TP does an AllReduce after **every** transformer block — 80 of them per forward pass for a 70B model, tens of MB each. On NVLink (900 GB/s) that's tens of µs each, trivial; over InfiniBand (~50 GB/s effective) it's hundreds of µs × 80 layers = milliseconds per step, which destroys ITL. PP communicates only at stage boundaries (once per forward, a few MB) so it crosses nodes fine. EP is hybrid: routing is high-frequency but only tokens whose experts are remote pay. Production examples: Llama-3.1 405B and DeepSeek-V3 both run 8-way TP × 2-way PP over 16 GPUs / 2 nodes.

- TP AllReduce cost: 160 AllReduces per forward pass (80 layers × 2); ~36 µs each on NVLink → **~5.7 ms per decode step** (vs ~30–40 ms total decode time); ~640 µs each on InfiniBand → **~100 ms**, i.e. a regression. [sourced] — Vizuara §16.2, §16.8.
- Under TP=N each GPU loads 1/N of the weights for unchanged FLOPs → **arithmetic intensity rises ~N×**, moving each GPU toward the ridge. TP doesn't speed up a single forward pass; it makes each GPU less bandwidth-starved. [sourced] — Vizuara §16.7.
- Single-GPU ceiling for 7–70B at batch 32 on H100: **2,000–5,000 tok/s**. Data parallelism at inference needs *zero* communication — it's just replication. [sourced] — Vizuara §16.1.
- Context parallelism example: 128K tokens ÷ 8 GPUs = 16K-token chunks in a ring-attention pass. [sourced] — Vizuara §16.5.

## Open questions
- Megatron-style TP math walk-through for Week 15 (primary source: Shoeybi et al.).

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §5.4.
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 6 §6.8–6.9 (interconnect arithmetic, node topology), Ch 16 (TP/PP/EP/SP/CP mechanics, AllReduce math, decision matrix).

## Series mapping
- Week 15 (primary), Week 14 (EP), Week 23 (interconnects).
