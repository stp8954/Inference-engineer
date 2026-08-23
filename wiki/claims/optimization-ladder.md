# The optimization ladder (measured stack-up)

Source: [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) Ch 24 (Capstone 1) — Llama-3-8B on 1× H100, ~30 min load test per configuration. **This is the closest published analog to our Week 30 showdown; our own numbers should be compared against it directly.**

| Step | Configuration | tok/s | Gain |
|---|---|---|---|
| 1 | HF Transformers baseline, batch=1 (`transformers.generate()`, ~3% GPU util) | 15 | — |
| 2 | + torch.compile & CUDA graphs | 35 | 2.3× |
| 3 | + FP8 W8A8 quantization (16 GB → 8 GB weights) | 85 | 2.4× |
| 4 | + FlashAttention 3 | 140 | 1.6× |
| 5 | + speculative decoding (EAGLE head, 3–4 drafts, ~75% acceptance) | 380 | 2.7× |
| 6 | + continuous batching at batch 32 | 850 | 2.2× |

- **Net ≈ 55× more tokens per H100-hour, same model, same prompts, same quality.** [sourced] Recorded 2026-08-22.
- Naive multiplication of the individual gains predicts ~1,050 tok/s; the measured stack reaches 850 — **overheads compound nonlinearly**, which is itself a finding worth reproducing. [sourced]
- Final capstone benchmarks: median TTFT 120 ms / P99 280 ms (SLO 500); median ITL 22 ms / P99 48 ms (SLO 100); 850 tok/s at batch 32; peak HBM utilization 76%; **$0.38 per million tokens**. Stack is ~300 lines of Python (mostly vLLM config) plus a ~10 MB EAGLE head. [sourced]
- ⚠️ Internal inconsistency in the source: prose says ~80× cheaper than GPT-4-class API pricing, a figure panel says 400×. Do not cite either without recomputing. [hearsay]
- Scale-out (Ch 25): the book gives conflicting figures — 200 replicas × ~5,000 users vs an extrapolation of 1M ÷ 80 users = 12,500 replicas at ~$50K/hour. Treat the per-replica concurrency claim as unverified. [hearsay]
- Conclusion's anchor number: one Llama-3-70B decode step on H100 spends **~4 ms loading 140 GB of weights against ~14 µs of arithmetic — a 280× imbalance**. Every runtime technique is a way to amortize that 4 ms. [sourced] — an excellent closing line for Week 1 or Week 24.
- Compounding claim: GQA × FP8 × FlashAttention × PagedAttention × speculative decoding × continuous batching = **100–1000× more useful work per GPU-hour**; frontier per-token prices have fallen ~2 orders of magnitude since 2023 with another expected by 2028. [hearsay] — directionally consistent with Kiely; verify prices before citing.
