# vllm

## What it is
The leading open-source inference engine; origin of PagedAttention. GitHub - vllm-project/vllm. First released summer 2023 (UC Berkeley, now a PyTorch Foundation / Linux Foundation project). Broadest model and hardware support (NVIDIA/AMD/Intel GPUs, TPUs); ~2× the GitHub stars of SGLang + TRT-LLM combined at Jan 2026; day-zero support culture; vLLM Omni extends to multimodal I/O.

## Timeline
- Per Vizuara Ch 20: the family tree's center of gravity — released late 2023 with PagedAttention, v1 rewrite 2025. Feature set: paging, continuous batching, chunked prefill, prefix caching, speculative decoding (N-gram/EAGLE/Medusa), TP + PP, experimental disaggregated P/D, AWQ/GPTQ/FP8 quantization, guided decoding, multi-LoRA, streaming. Weaknesses: less hardware-specific tuning than TRT-LLM, Python-heavy startup, rough edges on some long-context KV layouts.
- Per Kiely: fast engine startup (vs TRT-LLM's compile step); official base images are the recommended containerization starting point; vLLM Omni (new, 2025-26) extends toward diffusion/omni-modal serving.

## Relation to the series
- Weeks 5-9 reference engine; Week 30 benchmark yardstick

## Sources
- (pending first ingests)
