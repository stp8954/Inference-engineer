# Anatomy of a vLLM step

## What it is
The engine walkthrough (Vizuara Ch 19) — the single most useful page for both the Week 9 engine tour and the Phase 6 Rust build, because it names every component we will have to reimplement.

**Structure.** `LLMEngine` owns three collaborators: the **Scheduler** (chooses which sequences run each forward pass), the **Block Pool** (KV memory, PagedAttention), and **Workers** (one per GPU, runs the forward pass). Around them sit a Processor (tokenize/detokenize) and an Output Streamer (SSE/WebSocket).

**Startup partitions HBM**, and that calculation sets the concurrency ceiling: load weights → run a dummy forward pass to measure peak activation memory → `total_hbm − weights − activations = KV pool` → carve into blocks → pre-capture CUDA graphs for common decode batch sizes (1, 2, 4, 8, 16, 32). Decode has fixed shapes so it can replay captured graphs; prefill shapes vary, so it skips capture.

**A step** = build batch → forward → sample → stream → free finished. The scheduler runs *once per forward pass*, and the forward pass is ~95% of the step's time — scheduling ~100 µs, sampling ~1 ms, detokenization ~500 µs against tens of ms of GPU work. The admission policy is the load-bearing detail: **decode tokens are admitted first** (they cannot be postponed or chunked), then the remaining token budget is greedily packed with prefill chunks. That single rule is what keeps ITL stable under heavy prefill load.

**Block pool** is a FIFO free-block deque plus a refcount table for prefix-cached blocks — O(1) alloc/free, lock-free per GPU. If allocation were milliseconds instead of microseconds, the scheduler would become the bottleneck.

**Detokenizer subtlety worth stealing for the series:** it must buffer partial multi-byte UTF-8 and flush only at character boundaries, or users see half an emoji.

## Key numbers
- Startup on an 80 GB H100 (70B across 2 GPUs with TP): weights 40 GB, activations 5 GB, KV pool 35 GB → block bytes ≈ 256 KB at block_size=16 → **~140,000 blocks**. Weight load for 140 GB ≈ 140 s — the dominant cold-start cost. [sourced] — Vizuara §19.2.
- Step time ≈ 30–80 ms on one H100 (7B at batch 32; ~30 ms for Llama-3-8B) → **~33 scheduler iterations/sec per replica**. Per-token budget: 30 ms GPU + 1 ms detokenize + 2 ms network ≈ **33 ms end-to-end ITL**. [sourced] — Vizuara §19.5, §19.8.
- Tokenization ≈ 1 ms per 1K tokens (CPU). [sourced] — Vizuara §19.4.
- The two caps are independent: 8 sequences prefilling 4,096 tokens each = 32,768 tokens → OOM; 16 decodes + 4 chunked prefills of 512 = 2,064 tokens → healthy. vLLM v1 default `max_num_batched_tokens=8192`. [sourced] — Vizuara §19.6–19.7.
- **The eight knobs:** `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `block_size` (16 or 32), `gpu_memory_utilization` (0.9 typical), `enable_chunked_prefill`, `enable_prefix_caching`, `tensor_parallel_size`. Rule of thumb: **~2× throughput from parameter tuning alone is common on a fresh deployment.** [sourced] — Vizuara §19.10.

## Open questions
- Read the actual vLLM v1 source against this description before Week 9 — the book is a secondary source and the engine moves fast.
- Week 25–30 mapping: our Rust engine needs Scheduler, BlockPool, Worker, Sampler, Detokenizer. This page is the component checklist.

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 19.

## Series mapping
- Week 9 (engine tour, primary), Weeks 25–30 (Rust engine component map), Week 5 (scheduler), Week 6 (block pool).
