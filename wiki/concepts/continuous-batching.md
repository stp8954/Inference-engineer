# continuous-batching

## What it is
Iteration-level scheduling that admits/evicts requests every decode step instead of per-batch. Core idea from Orca; implemented by all modern engines.

Taxonomy (Kiely §7.2.1): static batching (wait for a full batch), dynamic (full batch OR timeout — right for non-autoregressive components like TTS audio decoders), continuous/in-flight (token-level admission and eviction — vLLM, SGLang, TensorRT-LLM). Batching is also why quantization/speculation interact: bigger batches soak up the idle compute speculation relies on. Ops tie-in: autoscaler concurrency targets should match replica batch size — scale up when all replicas hit max concurrency.

Roofline view (Vizuara Ch 3): batch size multiplies decode arithmetic intensity almost directly (weights load once per pass regardless of batch) — batch 1 → AI≈1, batch 64 → AI≈64, crossing the H100 ridge between batch 16–64 for 8B-class models. Below the ridge, added batch is ~free throughput; above it, throughput trades linearly against per-user ITL. vLLM knobs: `max_num_seqs` + `max_num_batched_tokens`, tuned to hold median batch near the ridge.

**The scheduler (Vizuara Ch 14).** It runs **once per forward pass**: decide the active set, run, evict finished sequences (EOS/max length), admit from the queue. Two governing knobs — `max_num_seqs` (concurrency cap; vLLM default 256; too low wastes the GPU, too high causes KV OOM or thrashing) and `max_num_batched_tokens` (per-step token budget = 1 per decoding sequence + up to chunk_size per prefilling one). Prefill and decode routinely share a pass: the attention kernel doesn't care which mode a sequence is in, so **prefill/decode contention stops being a scheduling fight** once mixed-mode batches exist.

Static batching fails on LLMs because output lengths vary 100× — a batch ends with its longest member, so most slots idle. At batch 1 the GPU waits ~99% of the time, which is the regime of nearly every naive PyTorch inference script.

## Key numbers
- (pending: throughput-vs-batch-size curve for Week 5 — predict the knee from the ridge point with KVScope first, then verify)
- Batch 16–32 ≈ production sweet spot on H100 (ridge vicinity). [sourced] — Vizuara Ch 2–3.
- Throughput vs same-size static batching: **2–4× tokens/GPU-hour** on realistic chat, 5×+ with very heterogeneous output lengths. [sourced] — Vizuara §14.3 citing Kwon et al. 2023 (vLLM paper).
- Tensor-core utilization: static ~22% → continuous ~73% (**3.3×**) on mixed workloads. [sourced] — Vizuara §14.8.
- A tuned H100 serves ~**50× more tokens/s** than a naive one-sequence-at-a-time loop; naive scripts run at ~5–10% of peak. [sourced] — Vizuara Ch 14 intro, §14.1.
- **Co-design fact:** continuous batching requires PagedAttention (a contiguous cache fragments the instant sequences join and leave) and PagedAttention's 8× concurrency claim is unrealizable under static batching. vLLM shipped both together — that pairing is why it became the default. [sourced] — Vizuara §14.9.

Amortization arithmetic (Vizuara §6.6): a 7B FP16 model reads 14 GB of weights per token → 4.18 ms ITL floor at batch 1 on H100. Batch 32 amortizes that same load across 32 users → **effective 0.13 ms/token**. This is the single largest lever in the runtime layer, and the reason speculative decoding (which spends the idle compute instead) conflicts with large batches.

## Open questions
- Orca paper ingest for the iteration-level scheduling mechanism (Week 5 primary source).

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §7.2.1 (batching taxonomy), §5 intro (technique interactions).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 3 §3.8 (batch multiplies AI).

## Series mapping
- Week 5, Week 27 (Rust implementation)
