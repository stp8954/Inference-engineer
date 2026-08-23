# Inference Engineering: The Definitive Workshop Guide — Vizuara

- **Type:** Book/workshop guide (Vizuara, 2026). Authors: Raj Dandekar, Rajat Dandekar, Sreedath Panat. Visual-first pedagogy; every technique placed on a recurring GPU roofline diagram.
- **Raw source:** attached PDFs `visuarap11.pdf` + `visuarap12.pdf` (96 pages)
- **Ingested:** 2026-08-22
- **Coverage: COMPLETE — Ch 0–26 plus conclusion.** The Ch 15–18 gap was closed 2026-08-22 with the final two PDFs (Ch 15 speculative decoding, Ch 16 parallelism, Ch 17 disaggregated P/D, Ch 18 replication/routing/multi-region in full).

## Summary

The closest published analog to the series' own approach: depth-first, derivation-driven, roofline-as-organizing-principle. Ch 3 is the standout ingested chapter — it derives arithmetic intensity for prefill (AI ≈ d/2, compute-bound), decode (AI ≈ 1, memory-bound, 0.34% compute utilization), and — the sharpest insight — attention itself at decode (AI ≈ N, actually COMPUTE-bound), locating the true memory bottleneck in weight reloading for projections/FFN, which reframes MLA's primary benefit as batch-size headroom rather than direct bandwidth relief. Ch 0's case studies (Vizz AI tutor: $300K/mo Gemini bill vs ~$15K self-hosted, ~20×; DynaRoute model routing: 86% savings) are excellent Week 24 economics material. Ch 2 formalizes the five metrics with production targets. Ch 4's 3×/6× training-vs-inference FLOPs rule and the ~400× throughput-regime gap are strong Week 1–2 framing. Competitive intel: this book validates the series' positioning but is workshop-breadth; the series' build-a-Rust-engine arc and weekly-cadence depth remain differentiated.

## Wiki pages touched

- concepts/prefill-decode.md (AI derivations, ridge point, attention-is-compute-bound nuance)
- concepts/inference-metrics.md (five metrics, targets, $/M formula, batch sweet spot)
- concepts/continuous-batching.md (batch multiplies AI; ridge as the sweet spot)
- concepts/attention-kernels.md (kernel efficiency ladder, CUDA graphs, TMA)
- concepts/kv-cache.md (MLA reframing note)
- concepts/training-vs-inference.md (new)
- claims/decode-bandwidth-ceilings.md, claims/hardware-specs.md, claims/inference-economics.md (new)

## Wiki pages touched (Ch 5–9 ingest)

- concepts/kv-cache.md (derivation, size formula, concurrency math, good/evil framing — major expansion)
- concepts/attention-variants.md (new — head and token compression families)
- concepts/hardware-landscape.md (SM internals, memory hierarchy, spec-sheet checklist)
- concepts/prefill-decode.md (naive-vs-cached roofline inversion)
- concepts/attention-kernels.md (SRAM tiling budget), continuous-batching.md (amortization arithmetic), model-parallelism.md (AllReduce-per-layer arithmetic)
- claims/kv-compression.md (new), claims/hardware-specs.md, claims/decode-bandwidth-ceilings.md

## Wiki pages touched (Ch 10–14 ingest)

- concepts/attention-kernels.md (tiling + online softmax mechanism, FA-1/2/3 lineage, I/O complexity, honest e2e-vs-kernel gains)
- concepts/paged-attention.md (rewritten from stub: fragmentation math, block tables, prefix sharing, block-size knob)
- concepts/prefix-caching.md (hashing/refcount/LRU, chunked prefill, the tails-not-medians signature)
- concepts/quantization.md (major expansion: formats, symmetric/asymmetric, GPTQ Hessian, GGUF K-quants, QAT/STE, BitNet)
- concepts/continuous-batching.md (iteration-level scheduler, knobs, utilization, PagedAttention co-design)
- claims/technique-performance.md (Ch 10–14 measured gains section)

## Wiki pages touched (Ch 18–26 ingest)

- concepts/vllm-internals.md (new — engine anatomy, the eight knobs; component map for the Rust build)
- concepts/fine-tuning-for-inference.md (new — LoRA/QLoRA/distillation/multi-LoRA, subliminal learning)
- concepts/multimodal-inference.md (new — token rates, voice/video budgets, embodied loops)
- concepts/inference-engines.md (2026 family tree, TGI/Ray Serve/Modal/local/specialty silicon, selection heuristic)
- concepts/production-operations.md (autoscaling rules, cost breakdown, multi-region, spot, N+1)
- claims/optimization-ladder.md (new — the measured 55× stack-up, with the source's own inconsistencies flagged)
- entities/vllm.md, entities/sglang.md

## Wiki pages touched (Ch 15–18 ingest — gap closure)

- concepts/speculative-decoding.md (accept/reject rule and why it's distribution-exact, acceptance→speedup geometric series, drafter comparison, workload sensitivity)
- concepts/model-parallelism.md (AllReduce arithmetic: 160/forward pass, 5.7 ms NVLink vs 100 ms IB; PP/EP/SP/CP mechanics; deployment recipes)
- concepts/disaggregated-serving.md (P99 motivation and dashboard signature, KV transfer math, P:D ratio sizing, connector anatomy, NIM/DeepSeek architectures)
- concepts/production-operations.md (replication linearity, cold-start breakdown, sticky KV-aware routing)
- claims/technique-performance.md (Ch 15–18 section)
