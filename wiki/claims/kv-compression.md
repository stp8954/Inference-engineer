# KV cache compression claims

Source key: Vizuara = [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) (Ch 8–9). All shapes use the reference model Llama-3-70B-like: H=64 query heads, D=128, L=80, FP16, N=32K unless stated.

## Head compression (Ch 8)
- MHA: `4·H·D·L` = 2.6 MB/token → **84 GB @ 32K**. [sourced]
- MQA (G=1): `4·D·L` = 40 KB/token → **1.3 GB @ 32K = 32× reduction**; costs ~0.5–2 perplexity points, worse on needle tasks. [sourced]
- GQA (G=8, Llama-3-70B actual): `4·G·D·L` = 320 KB/token → **10 GB @ 32K = 8× reduction**, quality gap vs MHA within noise. [sourced]
- MLA (R=512, DeepSeek-V3 L=61): `2·R·L` = 62 KB/token vs 4 MB/token for the same architecture with MHA → **~64× reduction**; DeepSeek-V3 is 671B total / 37B active. [sourced]
- Per-layer bytes/token at H=32, D=128: MHA 16,384 / GQA(G=8) 4,096 (4×) / MQA 512 (32×) / MLA(R=512) 1,024 (16×). [sourced]
- Why G=8: perplexity drops steeply G=1→4 and plateaus from 8→32 while cache grows linearly in G — pick the **left edge of the perplexity plateau**. Read any model's `num_key_value_heads / num_attention_heads` to see how hard it compressed. [sourced]
- Parameter counts barely move (d=4096, H=32 attention block): MHA ~4.2B, GQA(G=8) ~3.0B, MLA ~2.5B, MQA ~2.3B — only **30–45% parameter reduction for 4–64× cache reduction**, because W_Q is unchanged and is the largest matrix. Optimize per-token cache bytes, not parameters. [sourced]
- Pareto frontier passes through MHA, MLA, GQA — **MQA is dominated by GQA**. [sourced]

## Token compression (Ch 9), per layer at d=4096, N=32K
- MHA full: O(N·d) ≈ 256 KB/layer. Sliding window W=4K: ~32 KB/layer (~8×). Linear attention: ~32 KB/layer, N-independent (D×D state ≈ 1 MB/session across 32 layers). SSM/Mamba (state 64–256): ~2 KB/layer — Mamba ≈ **128× smaller** than full attention at long context; an SSM session state can be ~4 KB total vs ~43 GB for MHA at 32K. [sourced]
- **Needle-in-a-haystack retrieval @ N=32K: full attention 95%, Mamba 78%, sliding window (W=4K) 45%, linear attention 40%, hybrid (Mamba + a few attention layers) 92%.** The hybrid result is the load-bearing number — near-full quality at a fraction of the memory. [sourced]
- Selectivity is what buys Mamba its quality: input-dependent gating lifts needle retrieval from ~40% (fixed SSM / linear attention) to ~78%. In the book's traced example the final state is 65.5% "France" and 0.5% "of" — content-weighted, not recency-weighted. [sourced]
- Roofline direction: SSM/linear attention raise arithmetic intensity ~N×, making long-context decode **compute-bound** — the opposite regime from every other technique in the book. [sourced]

## Practical selection (Ch 8.8, 9.7)
- ≤13B model, ≤8K context, modest concurrency → plain MHA is fine.
- ~70B, ~32K, dozens of users per node → GQA G=8 (what Llama-3-70B does).
- Very long context / high concurrency → MLA, if the model was trained with it.
- MQA only for heavily memory-constrained edge deployments.
- Engine reality check (2026): if you run vLLM, **GQA or MLA are the safe bets**; Mamba variants need experimental engines (FlashSSM is new; continuous batching with Mamba is limited to specific codebases). [sourced]
