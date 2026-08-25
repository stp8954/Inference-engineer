# Technique performance claims

Source key: Kiely = [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md)

- Quantization: one precision level down ≈ 30–50% real-world perf gain (theoretical 2× FLOPS/bandwidth). [sourced] — Kiely §5.1. Recorded 2026-08-22.
- Speculation improves TPS/ITL only, never TTFT; must be dynamically disabled at high batch sizes. [sourced] — Kiely §5.2. Recorded 2026-08-22.
- Draft model ≥10× smaller than target (rule of thumb); EAGLE drafters <1B params, up to ~8 draft tokens; n-gram drafts 10+ tokens for code-editing workloads. [sourced] — Kiely §5.2. Recorded 2026-08-22.
- Disaggregation entry bar: ~100M–1B tokens/day AND model ≥ ~100B params AND prefill-heavy traffic. [sourced] — Kiely §5.5.2. Recorded 2026-08-22.
- KV storage tiers: G1 VRAM (TB/s) / G2 CPU RAM (10s–100s GB/s) / G3 local SSD (5–10 GB/s) / G4 network SSD (GB/s). [sourced] — Kiely §5.3.2. Recorded 2026-08-22.
- VLM: one high-res image ≈ ~1,000 visual tokens; 4 s of 24 fps video ≈ ~100K tokens before downsampling. [sourced] — Kiely §6.1. Recorded 2026-08-22.
- Diffusion: attention ≈ 70–80% of video-gen compute; attention caching → 30–40% speedup; guidance-skipping (last 20 of 50 steps) → 80 forward passes instead of 100. [sourced] — Kiely §6.5–6.6. Recorded 2026-08-22.
- ASR: optimized Whisper pipeline ≈ 1000× real-time factor (1 hr audio < 4 s) via VAD chunking + parallel MIG replicas; RTF scales ~linearly with GPU count. [sourced] — Kiely §6.3.2. Recorded 2026-08-22.
- TTS: real-time needs only ~80–100 tok/s — optimize concurrent streams per GPU, not TPS; Orpheus TTFB ~150 ms on one H100. [sourced] — Kiely §6.4. Recorded 2026-08-22.

## From Vizuara Ch 10–14 (2026)

- FlashAttention: ~10× less attention HBM traffic at long context, but end-to-end decode gain is only 5–10% (7–13B @4K) to 15–25% (70B+ @32K) because projections/FFN dominate. FLOPs unchanged. [sourced] — Vizuara §10.6, §10.8. Recorded 2026-08-22.
- FA version utilization on H100: FA-1 40% → FA-2 65% → FA-3 85% of tensor-core peak. [sourced] — Vizuara §10.7. Recorded 2026-08-22.
- PagedAttention: HBM utilization ~40% → >96%; concurrent users per H100 (Llama-3-70B) 3–5 → 20–40 (~8×). Prefix sharing typically saves 30–50%. [sourced] — Vizuara §11.8, §11.7. Recorded 2026-08-22.
- Prefix caching + chunked prefill: median TTFT 5×, P99 TTFT 10×, **P99 ITL 15×** (850→55 ms), medians ~flat. Scheduling optimizations show up in tails, not medians. [sourced] — Vizuara §12.10. Recorded 2026-08-22.
- Continuous batching: 2–4× tokens/GPU-hour vs static (5×+ on heterogeneous lengths); tensor-core utilization 22% → 73%. [sourced] — Vizuara §14.3, §14.8. Recorded 2026-08-22.
- Quantization measured e2e on H100 Llama-3-70B: INT4 ≈ 2.5× tokens/GPU-hour vs FP16; FP8 ≈ 2×. [sourced] — Vizuara §13.11. Recorded 2026-08-22.
- Stacked effect claimed across Ch 5–14: naive decode (~1% GPU utilization, ~3 concurrent users per H100) → modern engine (~60% utilization, 20–40 users); per-token costs 50–100× lower than naive baselines. [sourced] — Vizuara Ch 12 breadcrumb, §13.12. **This is the Week 30 showdown's target framing — verify our own numbers against it.** Recorded 2026-08-22.

## From Vizuara Ch 15–18 (the previously-missing chapters)

- Speculative decoding speedup formula: `(1 + α + α² + … + α^K) / (1 + draft_cost_fraction)`. At K=4, 5% draft cost: n-gram α=0.30 → 1.4×; Medusa α=0.55 → 2.0×; EAGLE α=0.75 → 2.9×. **+10 pp acceptance ≈ +30% speedup.** [sourced] — Vizuara §15.5. Recorded 2026-08-22.
- Speculation is distribution-exact (Leviathan et al. Theorem 1) — the only major technique with literally zero quality cost. [sourced] — Vizuara §15.3. Recorded 2026-08-22.
- TP AllReduce: 160 per forward pass on an 80-layer model; ~5.7 ms/decode step on NVLink vs ~100 ms on InfiniBand — the quantitative reason TP stays intra-node. [sourced] — Vizuara §16.2, §16.8. Recorded 2026-08-22.
- Disaggregation P99 TTFT: 2200 → 310 ms (~7×); realistic range 5–10× on P99, medians ~unchanged. KV transfer overhead 0.04% intra-node for a 4K-prefill/500-token-output request. [sourced] — Vizuara §17.4, §17.9. Recorded 2026-08-22.
- KV-cache-aware sticky routing cuts average TTFT 2–3× for multi-turn chat vs least-busy. [sourced] — Vizuara §18.3. Recorded 2026-08-22.
- Cold start ≈ 4 min, ~140 s of it loading 140 GB of weights; warm pools cost ~20–30% extra GPUs. [sourced] — Vizuara §18.2. Recorded 2026-08-22.
- Book's full-stack tally (Ch 7–15 multiplicative): naive PyTorch ~10 tok/s/GPU → production vLLM 2,000–5,000 tok/s/GPU ≈ **200–500×**, purely engineering. [sourced] — Vizuara §15.9, Ch 16 recap. Recorded 2026-08-22. *(Note: the Ch 24 capstone measures 55× on a single 8B configuration — the difference is batch size and baseline choice. Reconcile carefully before citing either.)*

## The single-machine optimization ladder, measured (YALM, 2026)

One implementation, one machine, every step measured — Mistral-7B-Instruct-v0.2 FP16 (~15 GB), RTX 4090
(1,008 GB/s) and AMD EPYC 7702P (204.8 GB/s). All [sourced] — Chan, *YALM* (2026)
(see ../../sources/2026-08-24-chan-yalm.md). Recorded 2026-08-25. This is the closest published analogue
to the series' own Week 1 → Week 30 arc, and a useful sanity check on the Vizuara tallies above, which
measure a different baseline.

| Step | tok/s |
|---|---|
| Naive CPU | 0.6 |
| + OpenMP on matmul | 4.2 |
| + attention-head parallelism | 4.4 |
| + F16C vectorized FP16 weights | 8.2–8.4 |
| Naive GPU port | 2.9 |
| + warp-stride matmul | 51.7 |
| + kernel fusion | 54.1 |
| + matmul coalescing | 56.1 |
| + attention-mixing kernel redesign | 63.7 |
| + FP16 KV cache (final) | 63.8 short / 58.8 long |

- **Shape of the ladder, which matters more than the totals:** ~14× from CPU work, then a *regression* to 2.9 tok/s on the naive GPU port, then ~22× from GPU kernel work. The single largest jump (2.9 → 51.7, ~18×) is one change — giving each warp a strided slice of the matmul instead of one block per output element. Everything after it is single-digit percentages. [sourced] — Chan, *YALM* (2026). Recorded 2026-08-25.
- Peer comparison on the same box: llama.cpp 61.0 tok/s GPU / 8.7 CPU; `calm` 66.0 GPU. A careful from-scratch implementation lands between them, and all three sit within ~10% of the weight-bandwidth ceiling. [sourced] — Chan, *YALM* (2026). Recorded 2026-08-25.

## Edge MoE serving (FreeToken, 2026)

All [sourced] — FreeToken §5 (see ../../sources/2026-08-25-yang-freetoken.md). Recorded 2026-08-25.
Baselines are llama.cpp, Ollama, KTransformers and MoE-Infinity, with weight formats bit-exactly aligned.

- Decode throughput on RTX 5090: **77–83 tok/s** on Qwen3.6-35B-A3B (BF16) and **22–25 tok/s** on DeepSeek-V4-Flash (284B total / 13B active, MXFP4) — **1.8–2.3×** and **1.5–1.9×** the strongest baseline per workload.
- **Agentic stability is the real result.** FreeToken's decode rate stays within **12%** of its single-turn value across three agent workloads; KTransformers on DSV4-Flash has already lost **31%** by the second. Single-stream benchmarks systematically overstate baseline agentic performance — a methodological warning worth repeating in Week 4.
- **Tail TTFT as an availability boundary, not a latency statistic.** Worst-turn TTFT stays below **44 s** in every cell; each baseline crosses 150 s somewhere (llama.cpp 232 s, Ollama 179 s, KTransformers 946 s). Real clients abandon first: OpenClaw ships a 120 s idle watchdog, Claude Code defaults to roughly a ten-minute request timeout.
- Expert-cache locality at equal capacity (37% of the Qwen3.6 pool, 11% of DSV4-Flash's) — decode-time expert miss rate: **global LRU 16% / 39%**, KTransformers' prefill-updated placement 41% / 59%, llama.cpp's routing-blind static split 62% / 89%. Ordering holds at every capacity short of the full pool.
- Prefill double-buffering ablation: each 8,192-token chunk completes in 1.19–1.22 s — exactly the time to stream the 64.4 GB expert pool once at 52.7 GB/s, so computation is fully hidden behind transfer. Disabling the second buffer costs **19% at 4K, 25% at 8K, 26% at 16K**, the penalty growing with prompt length. Prefill reaches 6.7K tok/s at 16K tokens.
- Cross-hardware speedup over the strongest baseline (coding-agent workload): 1.3× on RTX 3090 and 4090, 1.9× on the 5090 server, 2.1× on the 5090 desktop, 1.8× on the 4060 laptop.
- Capability results: an **8 GB RTX 4060 laptop on PCIe ×8 serves a 35B model at 39.3 tok/s** — 92% of the RTX 4090 rate, and above the 33 tok/s median decode speed of Codex in production traces. A single RTX PRO 6000 serves **GLM-5.2 (753B / 40B active, a 433 GB checkpoint) at 14.9 tok/s vs llama.cpp's 7.3** (2.0×) with bit-identical weights and comparable TTFT (7.5 s vs 7.8 s).
