# PagedAttention

## What it is
Operating-system virtual memory applied to the KV cache, introduced by vLLM in 2023. The naive approach pre-allocates a contiguous slab per user sized for the maximum expected sequence, which forces defensive over-allocation and leaves **40–70% of HBM unused**: both *external* fragmentation (enough free bytes overall, but split into unusable holes) and *internal* fragmentation (reserve 25 GB, use 12). No contiguous scheme avoids both.

PagedAttention instead splits the cache into fixed-size blocks (vLLM default: **16 tokens**) that live anywhere in HBM, plus a per-sequence **block table** mapping logical token positions to physical block IDs. Lookup is `logical_block = pos // 16`, `offset = pos % 16`, then one table read — a few tens of nanoseconds against HBM latency of hundreds of cycles. Blocks are allocated on demand as the previous one fills and returned to the free pool the moment a session ends.

The kernel is unchanged in structure: for each physical block, fetch into SRAM, compute `Q·K_blockᵀ/√d`, apply the same online-softmax merge rule as FlashAttention, accumulate. **PagedAttention is FlashAttention applied to non-contiguous KV blocks** — the two compose additively.

**Prefix sharing falls out for free.** Users with a common prompt prefix point at the same physical blocks, tracked by reference count and freed at zero; copy-on-write handles mutation, though in practice generation appends *new* blocks so COW rarely triggers and sharing stays read-only. This is the foundation prefix caching (Week 7) builds on.

## Key numbers
- Block size formula: `16 tokens × H_kv·D × 2 bytes × 2 (K,V) × L`. Llama-3-70B (GQA H_kv=8, D=128, L=80) → **6.5 MB per block**. [sourced] — Vizuara §11.3.
- HBM utilization: contiguous ~40% → paged **>96%**. Concurrent users per H100 on Llama-3-70B: **3–5 → 20–40 (~8×)** with no change in quality or per-request latency. [sourced] — Vizuara §11.1, §11.8.
- vLLM at launch (2023) was ~10× cheaper per token than HF Transformers largely because of this; concurrency went from 4–8 to 20–40 users per GPU. [sourced] — Vizuara Ch 11 intro.
- Worked fragmentation example: 100 GB HBM, 4 users pre-allocated 25 GB each, actually using 22 GB total — **78 GB wasted**, and a fifth user is refused while the GPU sits idle. [sourced] — Vizuara §11.1.
- Prefix sharing savings in production: typically **30–50%**. [sourced] — Vizuara §11.7.
- Block-size tuning (the one knob): 4–8 tokens = finer granularity, more lookup overhead (favors short/streaming workloads); 32–64 = less overhead, more internal fragmentation (favors long documents). Default 16. [sourced] — Vizuara §11.11.

## Open questions
- Read the vLLM paper (Kwon et al. 2023) as primary source before Week 6.
- Week 28 implementation: our Rust block allocator + block table; measure achieved utilization against the ~96% claim.

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 11.
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §2.5 (PagedAttention as a fragmentation fix).

## Series mapping
- Week 6 (primary), Week 7 (prefix sharing), Week 28 (Rust implementation), Week 5 (co-dependency with continuous batching).
