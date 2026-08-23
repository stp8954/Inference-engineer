# prefix-caching

## What it is
Reuse of shared prompt prefixes (system prompts, multi-turn, agent loops) via radix trees or block hashing.

Re-use KV cache across requests sharing a prefix → skip prefill on shared tokens → TTFT wins (why APIs discount cache-hit input tokens). Prefix ends at the first non-matching token, so prompt design is performance engineering: put novel tokens as late in the context as possible. High-value workloads: agent system prompts, code completion, doc QA, multi-turn chat. Non-prefix KV re-use (positional-embedding correction + selective recompute) is active research — CacheBlend, LMCache. At fleet scale this becomes routing: cache-aware routing sends a user's follow-ups to the replica already holding their prefix; alternatively a global networked KV store serves all replicas (and survives autoscaling churn), though hot-in-VRAM still beats network fetch.

**Mechanism (Vizuara Ch 12).** Three additions to a paged engine: a fast hash (xxhash-class, keyed on the *token ID sequence*, not the string), a hash→block-table lookup, and reference counting for eviction (LRU over zero-refcount entries — which works because shared system prompts stay "recent" by construction). Lookup hashes block-aligned prefixes of increasing length and takes the longest hit, then prefills only the remainder.

**Chunked prefill** is the scheduling half of the same chapter and solves head-of-line blocking: one 4,000-token prefill monopolizes a forward pass and spikes every other user's ITL (the book's example: 50 ms → 850 ms, a 17× spike). Slicing it into 512-token chunks interleaved with decodes keeps ITL flat at a small TTFT cost. Note the relationship to FlashAttention: chunked prefill is *scheduler-level* tiling, FlashAttention is *kernel-level* tiling — a 4,096-token prompt is 8 chunks × 8 SRAM tiles, and the two compose multiplicatively.

## Key numbers
- (pending: measure prefix-cache TTFT deltas with KVScope for Week 7)
- Motivating scale: a 500-token system prompt × 10K users × 30 messages ≈ 150M redundant prefix tokens/day ≈ **1,875 GPU-hours/day** of identical work at ~45 ms per 500-token prefill on Llama-3-70B/H100. [sourced] — Vizuara §12.1.
- Chunk size sweet spot on H100: **512–1024 tokens** (512 typical default, ~10–20 ms compute per chunk). vLLM knobs: `max_num_batched_tokens` (v1 default 8192), `long_prefill_token_threshold` (default 2048). [sourced] — Vizuara §12.6.
- Combined worked example (5,000-token prompt, 3,500 cached, 1,500 new in 3 chunks): 800 ms → 240 ms (cache only) → **140 ms stable** (cache + chunking) = 5.7× TTFT, with no P99 spikes for other users. [sourced] — Vizuara §12.8.
- **The production signature:** medians improve modestly, tails dramatically — median TTFT 400→80 ms (5×), P99 TTFT 2200→220 ms (10×), median ITL 50→48 ms (flat), **P99 ITL 850→55 ms (15×)**. Fine medians plus painful tails almost always means a scheduling problem. [sourced] — Vizuara §12.10.
- Prefix caching helps TTFT only — prefill time falls ~linearly with cache-hit ratio (80% hit → 5× cheaper prefill); decode/ITL is untouched. Chunked prefill changes no FLOPs or bytes at all, only scheduling. [sourced] — Vizuara §12.11–12.

## Open questions
- RadixAttention specifics (SGLang paper) still to be ingested from primary source.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §5.3.1, §5.3.3.
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 12 (hashing/refcounting/eviction, chunked prefill, combined metrics table).

## Series mapping
- Week 7, Week 29 (Rust implementation)
