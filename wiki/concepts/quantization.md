# quantization

## What it is
Lower-precision weights/activations/KV (FP8, INT4, NVFP4...) — primarily a memory-bandwidth optimization for decode.

Only lossy technique among the big five (quantization, speculation, caching, parallelism, disaggregation) — everything else is quality-neutral. Helps BOTH phases: prefill gets 2× FLOPS from lower-precision Tensor Cores; decode effectively doubles bandwidth by halving bytes per value. Floating-point formats beat integer for inference (exponent handles outliers); FP8/MXFP8 is the production sweet spot, NVFP4 promising. Microscaling: MXFP8/MXFP4 use a blockwise scale factor per 32 params; NVFP4 per 16 + a global FP32 scale; Blackwell applies scale factors inside Tensor Cores. QAT (GPT-OSS in MXFP4, Kimi K2 Thinking in INT4) vs PTQ (calibration; NVIDIA ModelOpt is the leading tool, outputs run in vLLM/SGLang/TensorRT-LLM). Sensitivity order (least→most): weights < activations < KV cache < attention (softmax nearly always kept full precision; first/last layers often too). Quality bar: zero perceptible loss — check perplexity delta + standard benchmarks + custom evals vs. original weights, accepting only run-to-run-noise differences. GGUF is the distribution format for aggressive quants (Unsloth's 1.58-bit dynamic mixes).

**Why it works and how (Vizuara Ch 13).** FP32's dynamic range vastly exceeds what trained weights actually occupy (typically ±1, clustered near zero), so weights compress to ~4 bits with <1% benchmark loss; activations are more sensitive because of residual-stream outliers — hence **weight-only quantization is the production default**. Rule: symmetric (abs-max) for weights, asymmetric (zero-point) for activations. A single outlier is catastrophic for a shared scale — one 80.0 in a row of ~±10 values leaves normal weights occupying 1.6% of the INT8 grid — which is why group-wise scales (group_size=128 in GPTQ/AWQ) and SmoothQuant-style outlier migration exist.

**Two paradigms:** weight-only (GPTQ, AWQ, bitsandbytes NF4, GGUF) stores INT4 and dequantizes to FP16 on the fly — saves HBM traffic only, works on any FP16 GPU; W8A8 (LLM.int8, SmoothQuant, DeepSeek FP8) quantizes activations too and uses the INT8/FP8 tensor-core path — double bandwidth saving *plus* a higher compute ceiling, at the cost of handling activation outliers.

**GPTQ** quantizes weights one at a time and pushes each rounding error onto the remaining unquantized weights, weighted by the inverse Hessian (sensitivity), so error accumulates preferentially in tolerant weights — Llama-2-70B at ~3.5 bits/weight with <1% perplexity loss from ~128 calibration samples. **GGUF** (llama.cpp/Ollama) uses a two-level scale hierarchy — 32 INT4 weights share an INT8 sub-block scale, 256 weights share an FP16 super-block scale — which amortizes scale overhead down to ~4.3–4.6 effective bits and is what puts a 70B model on a MacBook. **QAT** simulates quantization in the training loop with a straight-through estimator (pretend round′=1 in backward, else gradients are zero everywhere); the model learns which grid point minimizes loss rather than accepting the nearest one, and it steers the optimizer into *wide* minima where rounding costs little. **BitNet 1.58b** goes to ternary {-1,0,+1}, turning matmul into additions — ~14× lower energy per matmul, ~30% natural sparsity, but requires QAT from scratch and remains research as of 2026.

## Key numbers
- One precision level down ≈ 30–50% performance gain in practice (not the theoretical 2×). [sourced] — Kiely §5.1.
- Measured end-to-end on H100, Llama-3-70B: **INT4 ≈ 2.5× more tokens/GPU-hour than FP16; FP8 ≈ 2×**. Quantization moves the roofline point up *and* right — bytes/weight fall 4× (140 GB → 35 GB) while the ceiling rises (FP16 989 → FP8 1,979 → INT4 3,958 TFLOPS). [sourced] — Vizuara §13.11.
- GGUF K-quant footprints for Llama-3-70B: Q8_0 66 GB (8.5 b/w, ~lossless) · Q6_K 54 GB · Q5_K_M 48 GB · **Q4_K_M 42 GB (4.6 b/w — the community default sweet spot)** · Q4_0 39 GB · Q3_K_M 33 GB (noticeable loss) · Q2_K 27 GB (2.6 b/w, significant loss, experimental). [sourced] — Vizuara §13.7.2.
- Q4_1 vs Q4_0 in the book's traced block: mean error 0.016 vs 0.030 — nearly half the error for 0.5 extra bits/weight, because asymmetric beats symmetric on one-sided data. [sourced] — Vizuara §13.7.2.
- Max symmetric quantization error = scale/2. [verified] — arithmetic.
- Composition is multiplicative: GQA + FP8 → 4× KV reduction; MLA (~8×) + FP8 (2×) → **16× vs MHA/FP16**. FP8 KV cache is common in 2026; FA-3 supports FP8, FA-2 does not. [sourced] — Vizuara §13.12.
- BitNet 1.58b energy per matmul relative to FP16: INT8 0.4×, ternary **0.07×** (~14×), with ternary-trained models reaching 98–99% of full-precision quality. [sourced] — Vizuara §13.10.
- Not solved by quantization: activation outliers (~10× typical magnitude) and non-linear ops (softmax, LayerNorm, GELU, residuals) which stay FP16/BF16 even in INT4-weight models. [sourced] — Vizuara §13.13.
- MXFP8/MXFP4 block size 32; NVFP4 block size 16 + global scale. [sourced] — Kiely §5.1.1.
- Format-support timeline: FP8 from Hopper (2022); MXFP8/FP6/FP4/NVFP4 from Blackwell (2024); INT4 from Turing (2018). [sourced] — Kiely §5.1.1.

## Open questions
- Which PTQ recipes to reproduce for Weeks 10–11 (candidate: ModelOpt FP8 weights+activations+KV, attention untouched, evaluated with perplexity + evals).

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §5.1 (formats, approaches, quality measurement).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 13 (bit layouts, symmetric/asymmetric traces, GPTQ Hessian mechanism, GGUF K-quant arithmetic, QAT/STE trace, BitNet, decision matrix).

## Series mapping
- Weeks 10-11
