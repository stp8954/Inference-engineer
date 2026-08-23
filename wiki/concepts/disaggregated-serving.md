# disaggregated-serving

## What it is
Separate prefill and decode fleets with KV transfer between them; the current large-scale serving architecture shift.

Mechanics: prefill engine builds KV cache + first token → KV transferred over interconnect → decode engine generates the rest. Each fleet is tuned separately (prefill typically wants lower TP than decode). Production reality is conditional disaggregation: requests land on decode engines, which prefill locally when input is short or cached, forwarding only long uncached inputs to the prefill fleet. Dynamo notation xPyD (5P3D = 5 prefill, 3 decode), adjustable at runtime. New bottlenecks it introduces: prefill queue depth and decode-side KV exhaustion (mitigated by KV quantization + offloading).

**Why it exists (Vizuara Ch 17).** Chunked prefill smooths head-of-line blocking but doesn't remove it: a 16K-token prefill arriving mid-stream still spikes another user's ITL from 50 ms to 850 ms. That's an infrastructure problem, not a runtime one — and no single-GPU tuning removes it, because compute-bound and memory-bound workloads simply want *different GPUs*. Separate the pools and each sits at its own ideal operating point: prefill batches ~8 concurrent requests to saturate compute, decode batches ~32 to saturate bandwidth.

**The dashboard signature to teach:** flat ~50 ms ITL punctuated by occasional 400–900 ms spikes. If your SLO is P99 ITL < 100 ms, those spikes are the failure, and tuning `max_num_batched_tokens` only trades them against TTFT.

**KV transfer** is a one-time cost amortized over hundreds of decode steps: a 4K-token prefill on Llama-3-70B produces ~5 GB of KV, which is ~6 ms over NVLink (0.04% of a 500-token generation) or ~100 ms over InfiniBand. Keep pools in one node where possible; MLA-compressed KV shrinks the transfer ~8× (DeepSeek's approach). The connector does four things — publish paged blocks, transport (NVLink / RDMA / TCP, optionally FP8-compressed), receive into the decode pool's own block table, activate — essentially OS page migration; NIXL is the emerging standard.

**Hardware specialization is the deeper payoff:** H100s for prefill, H20 or B200 for decode (reduced compute, similar bandwidth), or even Cerebras/Groq for prefill at datacenter scale.

## Key numbers
- When to use (all three): ~100M–1B+ tokens/day volume, model ≥ ~100B params, prefill-heavy traffic. Otherwise scale replicas horizontally. [sourced] — Kiely §5.5.2.
- Textbook workload: frontier-model code editor (huge, varied, mostly-prefill contexts). [sourced] — Kiely §5.5.2.

- **P:D ratio sizing:** chat (short prompts, long outputs) → ~1:4 prefill:decode; RAG/document-heavy (long prompts, short outputs) → 4:1 or higher. Measure your own prefill:decode compute split; there is no universal ratio. [sourced] — Vizuara §17.5.
- Latency impact (TTFT, aggregated → disaggregated): P50 200→180 ms, P90 450→240 ms, **P99 2200→310 ms (~7×)**. Realistic gains are 5–10× on P99 TTFT and similar on P99 ITL — medians barely move. [sourced] — Vizuara §17.9.
- KV transfer: ~5 GB for a 4K-token Llama-3-70B prefill → ~6 ms on NVLink (**0.04% overhead** across a 500-token generation) vs ~100 ms cross-node on InfiniBand. [sourced] — Vizuara §17.4.
- When to adopt: >100 concurrent users with a P99 SLO and budget for 2+ GPUs. Below ~10 users, one GPU with chunked prefill is simpler and roughly equivalent. Costs ~5 ms added TTFT plus real orchestration complexity. [sourced] — Vizuara §17.7.
- Production shapes: NVIDIA NIM (4×H100 prefill TP=4 → NVLink → 16×H100 decode, NIXL connector); DeepSeek-V3 (2 prefill + 8 decode nodes, MLA cutting transfer ~8×, FP8 throughout, composing EP=16 × TP=2 — three orthogonal parallelism dimensions at once). [sourced] — Vizuara §17.10–11.

## Open questions
- DistServe/Mooncake papers as primary sources for Week 17.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §5.5.
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 17 (P99 motivation, KV transfer math, P:D sizing, connector anatomy, real architectures).

## Series mapping
- Week 17
