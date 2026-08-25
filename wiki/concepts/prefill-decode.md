# Prefill vs. Decode

## What it is
The two computational phases of LLM inference. Prefill processes all prompt tokens in one pass — weights stream through HBM once for the whole prompt, giving high arithmetic intensity (compute-bound). Decode generates one token at a time — every parameter must move from HBM to compute per token, giving very low arithmetic intensity (memory-bandwidth-bound). This asymmetry explains TTFT vs. per-token latency behavior and motivates most serving architecture, up to and including disaggregated serving.

## Key numbers
- See [claims/decode-bandwidth-ceilings.md](../claims/decode-bandwidth-ceilings.md) for the weights÷bandwidth decode ceilings and the H100 ops:byte ratio (~295).
- Worked example (Kiely §2.4.2): standard attention at decode, d=128, N=4096, FP16 → arithmetic intensity ≈ 62 « H100's ~295 ops:byte → decode is memory-bound. Prefill loads weights once for the whole sequence and does large matrix-matrix work → high arithmetic intensity → compute-bound.
- Vizuara Ch 3 derivations: AI_prefill ≈ d/2 (Llama-3-8B, d=4096, N=2048 → AI ≈ 1,024, deeply compute-bound); AI_decode ≈ 1 for projections/FFN → achieved ≈ 3.35 TFLOPs = 0.34% of H100 FP16 peak — "99.66% of compute idle" at batch 1. Ridge point (H100 FP16) = 989 ÷ 3.35 = 295 FLOPs/byte. [sourced]
- **Nuance worth a post section (Vizuara §3.9):** attention itself at decode has AI ≈ N (≈4,096 at 4K context) — attention is COMPUTE-bound; the memory-bound bottleneck of decode is reloading weights for the projection matmuls + FFN every token. Reframes MLA/GQA: their primary win is smaller KV → more sequences fit → bigger batches → climb the roofline slope, not direct per-token bandwidth relief. (Note: Kiely's §2.4.2 example computes standard attention WITH the intermediate S/P matrices in HBM, which is what FlashAttention eliminates — the two books' numbers describe different implementations; reconcile carefully in Week 2/12 drafts.)

**Counterintuitive placement (Vizuara §5.11):** *naive* decode (no KV cache) is O(N²d) FLOPs over O(Nd) bytes → AI ≈ N, i.e. compute-bound like prefill. Adding the KV cache collapses FLOPs to O(Nd) while bytes stay O(Nd) → AI ≈ 1. The optimization that made decode viable is precisely what made it memory-bound. Good narrative beat for Week 3.

**Week 2 derivation plan.** Per layer, per forward pass over N tokens (Vizuara §5.2):
- Q/K/V projections `6Nd²` + output projection `2Nd²` + FFN `16Nd²` = **`24Nd²`**
- attention scores `2N²d` + context `2N²d` = **`4N²d`**
- Total ≈ **`24Nd² + 4N²d`** per layer.

Two teaching beats fall out of this:
1. **The Week 1 rule of thumb is the N=1 case.** At decode, 24d² per layer ≈ 2 × (12d² params per layer) — "2 FLOPs per parameter" isn't a separate fact, it's this formula with N=1. Deriving it in Week 2 retroactively justifies what Week 1 asserted.
2. **When does attention actually dominate?** Set `4N²d = 24Nd²` → **N = 6d**. For an 8B model (d=4096) that's **~24,600 tokens**. Below that context length the linear layers dominate the FLOP budget, so "attention is quadratic" is true but misleading at typical context lengths — which dovetails with the §3.9 finding that decode-attention is compute-bound and the real bottleneck is weight reloading. [verified] — algebra.

**MoE inverts the asymmetry in a second way (FreeToken §2.1–2.2).** The prefill/decode split is usually
taught as compute-bound versus bandwidth-bound. For mixture-of-experts models there is a *sparsity*
split layered on top of it, running in the opposite direction. At decode each token routes through only
k of E experts, so the working set is genuinely sparse — but at prefill the **union** of routes across a
long prompt covers most experts in every layer, so the expert working set goes effectively dense. The
same model is sparse one token at a time and dense a prompt at a time.

That has a sharp consequence for anything serving MoE from host memory: prefill has to stream nearly the
whole expert pool regardless of sparsity, while decode has to serve a handful of scattered misses. They
are not the same problem and do not have the same solution — FreeToken double-buffers whole layers for
prefill (transfer-bound, hide compute behind it) and splits individual misses between PCIe and CPU
execution for decode (latency-bound, hide transfer behind compute). Worth a section in Week 2, because
it shows the prefill/decode frame generalizing past the dense case the series starts from.

## Open questions
- KVScope trace for the Week 2 chart (TTFT vs. prompt length; per-token latency flat) — needs to be run.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — Ch 2 §2.4 (ops:byte, arithmetic intensity, roofline, worked attention example); Ch 1 §1.4 (TTFT from prefill, TPS from decode).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 3 (roofline, AI derivations, three optimization directions: right/up/up-right).
- [Yang et al., *FreeToken* (2026)](../../sources/2026-08-25-yang-freetoken.md) — §2.1–2.2, the MoE sparsity inversion between prefill and decode.

## Series mapping
- Week 2 (primary), Week 1 (napkin math), Week 17 (disaggregation). Draft: [../../drafts/week-01-life-of-a-token.md](../../drafts/week-01-life-of-a-token.md)
