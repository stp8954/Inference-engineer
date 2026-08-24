# tiny-vllm: LLM Inference Engine Course & Implementation — Maczan

- **Type:** Open-source repository with integrated pedagogical course. https://github.com/jmaczan/tiny-vllm
- **Language:** C++ and CUDA with Python utilities.
- **Published:** Ongoing (GitHub repository).
- **Accessed:** 2026-08-24
- **Coverage:** COMPLETE. Full working inference engine (Llama 3.2 1B Instruct reference) plus structured course materials.

## Summary

A minimalist, pedagogically-focused inference engine that combines runnable C++/CUDA code with a learning curriculum. Rather than a production framework, it functions as "a learning tool on your learning path" — each component is explained from first principles before implementation. The engine supports the complete inference pipeline: prefill + decode phases, KV cache management, static and continuous batching, PagedAttention with paged KV cache, FlashAttention-style online softmax, RoPE, group-query attention, and optimized CUDA kernels. Targets NVIDIA GPUs; verified on RTX 5090. Uses BF16 throughout. Course progression spans floating-point theory, GPU memory management, CUDA kernel development, batching strategies (static → continuous), and advanced optimization (paged attention).

## Relevance to the series

**Week 1–3 context:** Demonstrates the concrete GPU implementation of the bandwidth-ceiling math (Week 1) and prefill/decode distinction (Week 2). Shows memory layout, CUDA thread organization, and numerical stability considerations in practice.

**Weeks 5–9 (framework survey):** Reference implementation of a minimal but correct inference engine. Unlike vLLM/TensorRT-LLM (which are production-scale), tiny-vllm is stripped down to essentials — perfect for understanding which design choices matter. A direct architectural parallel to the series' pedagogical bent: "here's what the smallest correct engine looks like."

**Weeks 25–30 (Rust engine arc):** Architectural template for the series' own Rust implementation. Though written in C++/CUDA, the design patterns (continuous batching scheduler, paged KV cache layout, attention kernel structure, cuBLAS integration tricks) are directly transferable. The "column-major to row-major transposition trick" for cuBLAS and numerical epsilon handling are practical details the Rust engine will need.

**Measurement culture:** Educational code that ships correct numbers and visible optimizations — echoes the series' philosophy of learning in public with receipts.

## Wiki pages this touches

- concepts/inference-engines.md (minimalist engine design, course-based approach, architectural decisions)
- concepts/attention-kernels.md (CUDA kernel structure, online softmax, GQA implementation)
- concepts/kv-cache.md (paged KV cache layout, memory management)
- concepts/prefill-decode.md (pipeline implementation, batching integration)
- concepts/quantization.md (BF16 format theory and practice)

## Key pedagogical points for future posts

1. **Floating-point theory matters:** The course begins with why BF16 (11-bit mantissa) works for inference despite half the precision of FP32. This foundation is non-obvious and worth explaining in the series.
2. **GPU memory management is the bottleneck:** The course makes this visceral — threads, blocks, coalescing, bank conflicts are not optional details.
3. **Batching is not free:** Static batching is simple but inefficient (padded sequences); continuous batching is correct but requires scheduler sophistication. tiny-vllm shows both.
4. **PagedAttention avoids a rewrite:** Fragmented KV pages require block tables and careful indexing, but the payoff (no reshape/concat on every batch) is enormous.
5. **Column-major layout tricks:** cuBLAS expects column-major matrices; naive row-major transposes kill performance. The "trick" (treat row-major input as transposed, compute C^T = B^T @ A^T, transpose result) is non-obvious and reusable.

## Comparison to YALM (Chan)

Both are educational engines in C++/CUDA optimizing Mistral-7B or Llama inference. **YALM (Chan)** is a performance progression narrative — start at 0.6 tok/s, optimize to 63.8 tok/s, and explain each step. **tiny-vllm (Maczan)** is a course-first approach — theory first (FP math, GPU internals), then implementation, with less emphasis on the performance story. Together they cover the full range: YALM for "here's the optimization journey," tiny-vllm for "here's the theory and minimal design."
