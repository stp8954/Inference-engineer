# Inference engines

## What it is
The software layer that turns model weights into a production token service. Stack by abstraction (Kiely Ch 4): CUDA (kernels, graphs, driver/runtime; cuBLAS/cuDNN primitives) → frameworks (PyTorch dominant; torch.compile does auto kernel selection + fusion but cannot fuse plugin kernels like FlashAttention/DeepGEMM) → engines → orchestration (Dynamo). All three major engines are Apache 2.0 and ship the same headline features (continuous batching, quantization, speculation, prefix caching, parallelism, disaggregation); they differ in performance ceiling, ease, and breadth:

- **vLLM** (summer 2023, UC Berkeley → PyTorch Foundation): broadest model+hardware support (NVIDIA/AMD/Intel/TPU), day-zero support culture, vLLM Omni for multimodal. Pick for: almost any model, fast; smaller/older GPUs.
- **SGLang** (Dec 2023, LMSYS): composable, community-driven, engine of choice at xAI; co-develops with DeepSeek/Qwen/Kimi (MLA); deep multi-node MoE investment (GB200 NVL72); SGLang Diffusion. Pick for: large-MoE throughput, customization.
- **TensorRT-LLM** (V1 summer 2025, PyTorch-based — V0 was a TensorRT plugin; check versions): best peak performance via NVIDIA handwritten (partly closed) fused kernels, best Hopper/Blackwell + NVFP4 support, NVIDIA-only, steep curve, no image/video. Pick for: peak performance on well-supported models, pairing with Dynamo.

Engine choice embodies the book's core principle: more constraints → more performance (narrow TRT-LLM beats broad vLLM at the high end). Reference implementations (HF transformers/diffusers) are for learning and prototyping, not production.

**The 2026 landscape (Vizuara Ch 20).** Family tree: HF Transformers (2022, pre-engine) → TGI + FasterTransformer (2023) → **vLLM (late 2023, first PagedAttention — the center of gravity)** → TensorRT-LLM, SGLang, LMDeploy (2024) → vLLM v1 rewrite, llama.cpp/Ollama (2025) → Modal serverless, Ray Serve + vLLM (2026). Everything either builds on vLLM or competes with it, and its core primitives (paging, continuous batching) are now universal.

Beyond the big three: **TGI** trades leading-edge features for operational maturity (hot model swap, auth/rate limiting, Prometheus metrics, commercial support). **Ray Serve** is not an engine but the orchestration layer above one — cross-replica autoscaling, routing, composition — and **Ray Serve + vLLM is the reference architecture for large-scale serving in 2026**. **Modal** is serverless: ~2 s cold starts via pre-baked images and warm pools, scale-to-zero, billed per second — cost-effective below roughly 40% average utilization. **llama.cpp/Ollama** own local inference via GGUF. **Specialty hosted silicon** trades generality for speed: Cerebras WSE-3 (1,000–3,000 tok/s/user, model on-chip, no HBM), Groq TSP (500–1,500 tok/s, deterministic pipelines so latency variance is minimal), Taalas (~17,000 tok/s, model burned into the mask).

**Selection heuristic:** no single best engine — match to workload, then optimize. Chat/general → vLLM; structured/JSON → SGLang; lowest latency on frozen models → TensorRT-LLM or Groq; bursty → Modal; enterprise support → TGI; local/private → llama.cpp or Ollama.

## Key numbers
- vLLM ≈ 2× the GitHub stars of SGLang and TensorRT-LLM combined (at book publication, Jan 2026). [sourced] — Kiely §4.3.1.
- TensorRT-LLM is typically **1.5–2× faster per token than vLLM** on the same model and hardware (up to ~2× best case), paid for with minutes-long compiles, NVIDIA-only support, and no image/video models. [sourced] — Vizuara §20.4.
- SGLang's RadixAttention stores the prefix cache as a **radix tree rather than a flat hash table**, so it finds maximal shared prefixes across *unrelated* queries — a first-time user can hit a cache another conversation populated. [sourced] — Vizuara §20.3.
- Modal serverless: ~2 s cold start (vs ~4 min naive), scale-to-zero, per-second billing; cost-effective below ~40% average utilization. [sourced] — Vizuara §20.7.
- Dynamo announced GTC March 2025; SLA-based planner autoscales prefill/decode workers against TTFT/TPS targets. [sourced] — Kiely §4.4.

## Open questions
- Week 9 hands-on: same model, three engines, KVScope comparison — verify the performance ordering ourselves.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — Ch 4.

## Series mapping
- Week 9 (primary), Weeks 5–8 (vLLM/SGLang internals), Week 25+ (what our Rust engine borrows from each).
