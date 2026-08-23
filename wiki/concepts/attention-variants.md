# Attention variants (compressing the KV cache)

## What it is
Two orthogonal axes for shrinking `KV bytes = 4·N·H·D·L` (FP16): compress **across heads** (shrink H) or **across tokens** (shrink N). Head compression is close to lossless; token compression is inherently lossy — capped or collapsed history cannot be recovered.

**Across heads (Vizuara Ch 8), historical order MHA → MQA → GQA → MLA:**
- **MHA** (2017): per-head K and V. `cache = 4·H·D·L`.
- **MQA** (Shazeer 2019): one shared K and V for all query heads. `cache = 4·D·L` — H× smaller. Costs ~0.5–2 perplexity points, worse on needle-in-a-haystack. Used in PaLM, StarCoder; rare in frontier models today.
- **GQA** (2023): G groups of K/V, each serving H/G query heads (G=1 → MQA, G=H → MHA). `cache = 4·G·D·L`. The 2024–2026 default: Llama-3 all sizes G=8, Qwen-2/3 G=4, Mistral-Large G=8, Command-R G=4.
- **MLA** (DeepSeek-V2 2024, refined in V3): down-project K/V into a shared low-rank latent `c_KV` of rank R≈512 and cache *only the latent*; up-projections are folded into the query weights via the **absorption trick** — `Q·Kᵀ = x·(W_Q·W_UKᵀ)·c_KVᵀ`, so K is never materialized at decode. `cache = 2·R·L`. Preserves per-head diversity (the up-projection still yields distinct per-head K), so quality matches or beats MHA at equal parameters.

**Across tokens (Vizuara Ch 9), increasing aggressiveness:**
- **Sliding window**: attend to the last W tokens; `cache = 4·W·H·D·L`, a circular buffer of constant size. Receptive field ≈ W×L via layer stacking (Mistral-7B: 4K×32 = 128K nominal) but indirect ≠ direct — retrieval degrades past ~10K in practice. Gemma-2/3 interleave SW with periodic full-attention layers; Mistral-Large dropped SW for GQA because long-context retrieval was a product requirement.
- **Linear attention**: drop softmax, use a kernel feature map, exploit associativity to keep a running D×D state — constant memory at any length, but all past tokens contribute equally and opposing keys can cancel, erasing information. RetNet (decay) and GLA (learned gating) patch this.
- **SSMs**: `h_t = A·h_{t-1} + B·x_t` with |eigenvalues|<1 giving recency decay; trainable in parallel via the convolution view (FFT, O(N log N)), O(1) per step at inference. Each state dimension is its own memory timescale.
- **Mamba** (Gu & Dao 2024): makes A, B, C functions of the input — *selective* remembering. Content decides importance, not just position; trained with a parallel scan.
- **Hybrids win**: mostly SSM layers with a few full-attention layers (best placed mid-stack) — Jamba, Zamba.

## Key numbers
- See [claims/kv-compression.md](../claims/kv-compression.md) for the full comparison table and retrieval-quality numbers.

## Open questions
- Primary sources still to ingest: GQA paper (Ainslie et al.), DeepSeek-V2/V3 tech reports (MLA), Mamba paper — needed before Weeks 8–9 and 13.
- Reproduce the needle-in-a-haystack comparison ourselves? Would be an original contribution for Week 19.

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 8 (head compression, absorption trick, Pareto frontier), Ch 9 (token compression, Mamba trace, hybrid results).

## Series mapping
- Week 13 (primary — architectural efficiency), Week 3 (KV cache motivation), Week 19 (long context).
