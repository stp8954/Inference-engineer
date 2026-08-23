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
