# Inference metrics

## What it is
The measurement vocabulary of serving. TTFT (time to first token) is set by prefill and is the latency a user feels before output starts. "Tokens per second" is ambiguous and should be split: **perceived TPS** (per-user, after first token — a latency metric), **total TPS** (whole-service throughput), and **inter-token latency** (ITL; 10 ms ITL = 100 tok/s perceived). For non-streamed calls (e.g., agent tool calls), total response time is the right latency metric, not TTFT/TPS. Latency distributions are right-skewed, so percentiles (P50/P90/P95/P99) matter more than means — tail latency drives user trust. Distinguish inference-only time (measures model-performance work) from end-to-end time including network and queueing (measures user experience); when the first is fast and the second slow, the problem is infrastructure, not the model.

Vizuara Ch 2 formalizes "five numbers you will live by": TTFT, ITL, TPS, P99, $/M tokens — "TTFT and ITL are coupled through the prefill/decode boundary; TPS = 1/ITL; P99 is the shape of the distribution; $/M is the amortization across all of it." Stakeholder mapping: end users feel TTFT+TPS, CFO sees $/M, SRE lives on P99, researchers over-index on per-user TPS.

## Key numbers
- ITL ↔ perceived TPS conversion: perceived TPS = 1000 / ITL(ms). [verified] — arithmetic.
- Interactive targets (Vizuara Ch 2): TTFT P99 < 500 ms / P50 < 300 ms (>1 s "feels broken"); ITL < 50 ms interactive, < 30 ms voice. [sourced]
- Decode ITL floor from bandwidth: 7B @ FP16 = 14 GB ÷ 3.35 TB/s ≈ 4.18 ms/token pure memory traffic. [sourced] — Vizuara Ch 2.
- $/M tokens ≈ (GPU $/hr ÷ tok/s) × 278; at $2.50/hr: 100 tok/s → $6.95/M, 3,000 tok/s → $0.23/M. [sourced] — Vizuara Ch 2.
- Batch sweet spot ≈ 16–32 on H100 for most production models (at the roofline ridge). [sourced] — Vizuara Ch 2–3.
- P99 tail ≈ 9× median in the book's example (P50 250 ms / P99 2,200 ms); report percentiles, never means. [sourced] — Vizuara Ch 2.

Benchmarking methodology (Kiely §4.5): best benchmark = shadowed production traffic; if simulating, match sequence lengths (ISL/OSL), volume/pattern (with jitter), request contents (drives cache hits and draft acceptance), and inference params (temperature, reasoning effort). Baseline before optimizing; change one variable at a time; test optimizations individually AND together (they interact — e.g., speculation × batch size). Benchmarking says how fast; profiling says why: PyTorch Profiler (step-level CPU/GPU/memory), Nsight Systems (system-wide traces), Nsight Compute (per-kernel) — profiling is for engine contributors and new modalities, not daily work. Tools: SGLang Genai-bench, NVIDIA GenAI-Perf, Locust; eval datasets (MMLU, gsm8k, HumanEval) double as realistic inputs + quality spot-checks.

## Open questions
- Goodput definition and SLO-attainment metrics (not in Kiely Ch 1; from the disaggregation literature) — fill in when ingesting serving papers for Week 4.
- KVScope positioning vs Genai-bench/GenAI-Perf: those are load generators; KVScope's niche is phase-level attribution (prefill/decode split, KV growth, arithmetic intensity). Sharpen this in the Week 4 post.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §1.4 (TTFT/TPS/ITL, percentiles, end-to-end vs. inference-only).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 2 (five metrics, targets, $/M formula, stakeholder map).

## Series mapping
- Week 4 (primary — this page is the draft skeleton for the metrics post), Week 2 (TTFT/TPS phase mapping).
