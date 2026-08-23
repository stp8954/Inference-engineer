# sglang

## What it is
Open inference engine known for RadixAttention prefix caching. GitHub - sgl-project/sglang. First released Dec 2023 (LMSYS). Composable architecture; engine of choice at xAI; co-develops optimized implementations with DeepSeek/Qwen/Kimi (e.g., MLA); heavy investment in multi-node MoE serving on GB200 NVL72; SGLang Diffusion for image/video; Genai-bench benchmarking tool.

## Timeline
- Per Vizuara Ch 20: RadixAttention stores the prefix cache as a radix tree (not a flat hash), finding maximal shared prefixes across unrelated queries; structured generation (JSON/regex/grammar) is enforced at the token-sampling level rather than by parse-and-retry. Ships Genai-bench.
- Per Kiely: fast startup; recommended base images; SGLang Diffusion (new) packages performant image-generation engines.

## Relation to the series
- Weeks 7, 9

## Sources
- (pending first ingests)
