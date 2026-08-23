# KV Cache

## What it is
The cached key/value projections of all past tokens, kept so each decode step only computes attention for the new token instead of re-running the whole sequence. Turns quadratic recomputation into linear work at the cost of memory that grows with sequence length × layers × heads × head_dim. The binding constraint of decode is usually KV memory, not compute.

**Derivation (Vizuara Ch 5–7).** Naive decode recomputes every past row of K and V each step from frozen weights — bit-identical work. Cache K and V, *not* Q: Q[i] is single-use (it asks the question for position i and is never consulted again), while K/V describe how past tokens answer future queries and are read forever. That asymmetry in `attn_i = softmax(Q[i]·Kᵀ/√d)·V` is the whole justification. Per-step cost drops from O(N²d) to O(Nd).

**The good/evil pairing (Ch 7).** The same structure that removes the FLOPs creates the bandwidth problem: the entire cache is re-read from HBM every decode step, and since attention FLOPs and bytes both scale with N, arithmetic intensity stays pinned near 1–2 regardless of context. The KV cache *solves the FLOPs problem and creates the bandwidth problem* — it moves decode LEFT on the roofline, deeper into memory-bound territory. Everything in Weeks 6–14 is a counter-attack on one of its terms.

## Key numbers
- **Size formula: `KV bytes = 2 (K,V) × N × H × D × L × bytes_per_elem` = `4·N·H·D·L` at FP16.** Only N grows at inference; H, D, L are architectural. [sourced] — Vizuara §5.13/§7.4.
- Llama-3-70B (H=8 KV heads, D=128, L=80) at N=32K: **43 GB per user** — over half an H100's HBM for one session. Same model at N=4K: 5.4 GB. [sourced] — Vizuara §7.4–7.5.
- Concurrency math: 70B FP16 = 140 GB weights → 2-way TP = 70 GB/GPU, ~5 GB activations → **~5 GB left for KV → under 1 user per GPU at 32K**. "Every byte saved per user is a new concurrent user." [sourced] — Vizuara §7.6.
- Cross-model at N=4K FP16: GPT-2 124M = 12 MB; GPT-3 175B = 2.4 GB; Llama-3-70B = 5.4 GB; DeepSeek-V3 (MLA) = 0.6 GB vs ~15 GB if it used MHA. [sourced] — Vizuara §7.5.
- Bandwidth floor, Llama-3-70B @32K on H100: 43 GB cache ÷ 3.35 TB/s = 12.8 ms + weights ~21 ms → **~34 ms single-user ITL**, cache traffic dominating at long context. Bytes-per-FLOP crosses the H100 bandwidth line around **N ≈ 4K**. [sourced] — Vizuara §7.7.
- FLOP savings scale as ~N: N=128 → ~100×; N=2,048 → ~2,000×; N=32K → ~32,000×. [sourced] — Vizuara §7.3.
- When NOT to compress: plain MHA is fine below ~4K context, <20 concurrent users/GPU, models <13B. Measure the binding constraint first. [sourced] — Vizuara §7.11.
- Rule of thumb (Kiely §3.1.2): provision VRAM = weights + ≥50% headroom for KV cache; more for long context, high batch, or video models. [sourced]

Storage is tiered (Kiely §5.3.2): G1 GPU VRAM (TB/s), G2 CPU RAM (10s–100s GB/s), G3 local SSD (5–10 GB/s), G4 networked SSD (GB/s) — keep hot blocks high, demote cold ones; NVIDIA Dynamo's KVBM manages block movement across tiers. Engine allocation is explicit, e.g. TensorRT-LLM `free_gpu_memory_fraction=0.8`: B200 (180 GB) − 100 GB weights/buffers → 64 GB KV cache. "Long context" ≈ when the KV cache itself causes problems — typically past 32K/64K/128K tokens depending on stack; mitigations: sliding-window/compressed/sparse attention (model-side), FlashAttention, PagedAttention, chunked prefill (engine-side).

## Open questions
- Which KV quantization / eviction methods to cover in Weeks 18–19.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §2.2.3 (KV cache makes attention linear-in-practice; built during prefill, used/updated during decode, lives in GPU memory; §5.3 for re-use).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 5 (derivation, cache-vs-recompute, why not Q), Ch 7 (size formula, concurrency math, good/evil trade-off table, roofline shift).

**Reframing note (Vizuara §3.9):** decode-attention itself is compute-bound (AI≈N); the true memory bottleneck is weight reloading for projections/FFN. KV-shrinking techniques (MLA/GQA/quantized KV) therefore win mostly by fitting more concurrent sequences in VRAM → bigger batches → higher AI — not by direct bandwidth relief on attention reads. Carry into Weeks 3 and 13.

## Series mapping
- **Anchor figure debuts here** (Week 3): `drafts/figures/fig4-anchor.*` — the byte budget of a decode step plus the five families of attack on it. Recurs for the rest of the series. See `planning/visual-strategy.md`.
- Week 3 (derivation), Week 6 (PagedAttention), Week 18 (offloading/tiered storage), Week 26 (Rust implementation).
