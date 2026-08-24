# LLM Inference Blog Series — Master Plan

**Audience:** ML engineers who deploy and serve models — comfortable with PyTorch, CUDA-adjacent concepts, and systems thinking, but who want a rigorous ground-up treatment of inference.

**Platform:** one Substack publication (email-first) with two sections: **Deep Dive** and
**This Week in Inference**, so subscribers can opt into either or both.

**Publication model:** a weekly digest plus a deep dive every other week during the initial season.
The curriculum is 30 installments, not a promise to finish in 30 consecutive calendar weeks. After
the Week 4 checkpoint, the deep dive may move to weekly only if the evidence workflow is stable, at
least four finished deep dives remain buffered, and KVScope work is not competing with publication.

- **Deep Dive** (educational track) — publishes every other Tuesday. ~1,500–2,500 words, code + diagrams. Evergreen.
- **This Week in Inference** (news track) — publishes Friday or Monday. ~600–1,000 words. Timely, skimmable, link-heavy.

The two tracks reinforce each other: the digest cites your own deep dives ("we covered paged attention in Week 5"), which drives archive traffic; the deep dives stay relevant because readers see the concepts appear in the news.

---

## Track A: The Deep Dive Curriculum (Beginner → SOTA)

A 30-installment arc in six phases. Each post ends with a "what's next" teaser and links back to prerequisites, so the archive works as a self-serve course.

### Phase 1 — Foundations (Weeks 1–4)

**Week 1: What actually happens when an LLM generates a token.**
The forward pass viewed as a serving problem. Autoregressive decoding, logits → sampling, temperature/top-p. Build a naive `generate()` loop in ~50 lines of PyTorch. Establish the series' framing: inference is a *systems* problem, not a modeling problem.

**Week 2: Prefill vs. decode — the two-phase workload.**
Why processing the prompt and generating tokens tend to occupy different computational regimes.
Prefill can become compute-bound as sequence length and batch grow; low-batch decode is usually
dominated by moving weights and, at long context, KV data. Make the dependencies on sequence length,
batch, model architecture, kernels, and hardware explicit. Introduce arithmetic intensity and the
roofline model, while keeping the linear-layer/attention FLOP crossover separate from the hardware
roofline crossover. First KVScope cameo: show a prefill/decode timing split under one fully specified
configuration ("see this yourself" — full tool introduction comes in Week 4) and include the minimal
measurement contract: model revision, hardware, software versions, prompt/output lengths, batch or
concurrency, warm-up, and sampling settings.

**Week 3: The KV cache — why it exists and what it costs.**
Derive the KV cache from the attention equation. Compute its memory footprint by hand for Llama-class
and MoE models. Distinguish per-step from cumulative complexity: caching avoids recomputing prior
keys and values, while attention over the cached prefix still grows with context length. Show when
weight traffic dominates low-batch decode and when KV traffic becomes material or dominant.

**Week 4: Measuring inference — the metrics that matter (KVScope debut).**
TTFT, TPOT/ITL, throughput vs. goodput, latency percentiles, tokens/s/GPU, cost per million tokens.
How benchmarks lie (batch size games, input/output length mixes). Official introduction of
**KVScope** ([github.com/stp8954/KVScope](https://github.com/stp8954/KVScope)) as the series'
companion profiler. The Week 4 MVP must separate prefill and decode timing, record TTFT and per-token
latency, track KV growth, emit a versioned JSON report with the benchmark contract, and run one
repeatable local backend end to end. Arithmetic-intensity estimates and additional backends are
valuable but do not block publication. Later posts prefer KVScope reports; if a required backend is
unavailable, use a checked-in reference harness with the same report fields and disclose the
limitation. (Tease it in Week 2, but keep Week 1 tool-free; the naive loop must feel primitive.)

### Phase 2 — The Serving Engine (Weeks 5–9)

**Week 5: Batching — from static to continuous.**
Why naive batching wastes GPUs, how continuous (in-flight) batching works, iteration-level scheduling. Walk through the Orca paper's core idea and how vLLM/SGLang implement it.

**Week 6: PagedAttention and KV cache memory management.**
Virtual-memory-for-KV-cache: blocks, block tables, fragmentation, copy-on-write for beam search/parallel sampling. Read the vLLM paper together.

**Week 7: Prefix caching and RadixAttention.**
Shared system prompts, multi-turn chat, agentic loops — why prefix reuse became critical. SGLang's radix tree approach, vLLM's automatic prefix caching, cache-aware routing.

**Week 8: Scheduling and preemption.**
Chunked prefill, priority scheduling, preemption/swapping/recomputation, fairness. How schedulers trade TTFT against throughput, and SLO-aware serving.

**Week 9: Anatomy of a modern engine — vLLM vs. SGLang vs. TensorRT-LLM.**
Architecture tour of the major open engines: where the scheduler, model runner, and kernels live;
CUDA graphs; when to pick which. Compare capabilities from source, documentation, and one reproducible
benchmark where the available hardware and backends permit it. A KVScope vLLM backend is the target,
not a publication blocker; use the reference harness and disclose any missing engines rather than
delaying the installment or presenting an uneven comparison.

### Phase 3 — Making the Model Cheaper (Weeks 10–14)

**Week 10: Quantization I — the ideas.**
Number formats (FP16/BF16 → FP8 → INT4/NVFP4/MXFP4), weight-only vs. weight+activation,
calibration, outliers. Explain why reducing weight traffic can substantially accelerate memory-bound
decode, while dequantization, kernel availability, batching, and hardware support determine the real
gain. Treat model-quality effects separately from the different performance tradeoffs in prefill.

**Week 11: Quantization II — the methods in practice.**
GPTQ, AWQ, FP8 (per-tensor vs. per-block scaling), KV cache quantization, QAT vs. PTQ. What actually ships in production and how to evaluate quality degradation properly.

**Week 12: Attention kernels — FlashAttention and friends.**
How standard attention materializes and moves avoidable intermediate data; tiling and online softmax;
the FlashAttention 1→4 lineage; FlashInfer; and decode-specific kernels (flash-decoding, split-KV).
Use roofline evidence for the exact workload instead of labeling all attention memory-bound.

**Week 13: Architectural efficiency — MQA, GQA, MLA, sliding windows, and hybrids.**
How model architecture co-evolved with inference: grouped-query attention, DeepSeek's multi-head latent attention, sparse/linear attention hybrids (Mamba-style SSM blocks in production models), and what each does to the KV cache.

**Week 14: MoE inference.**
Why mixture-of-experts changed the serving calculus: expert parallelism, all-to-all communication, load balancing, memory vs. compute asymmetry. Serving DeepSeek/Qwen/Kimi-class MoEs.

### Phase 4 — Scaling Out (Weeks 15–19)

**Week 15: Parallelism for inference.**
Tensor, pipeline, expert, and data parallelism — inference-specific tradeoffs (latency vs. throughput), communication costs, and how this differs from training parallelism.

**Week 16: Speculative decoding.**
Draft models, self-speculation (Medusa, EAGLE lineage), acceptance rates, when it helps and when it hurts. Implement basic speculative sampling from scratch.

**Week 17: Disaggregated serving — splitting prefill and decode.**
The architecture shift toward separate prefill and decode fleets, KV cache transfer (NIXL/UCX-style),
and orchestration layers like NVIDIA Dynamo and llm-d. Examine where disaggregation is deployed,
where colocated serving still wins, and what evidence supports broader adoption without claiming
universal convergence.

**Week 18: The KV cache as infrastructure — offloading and tiered storage.**
CPU/NVMe/remote KV offload, cache hierarchies (LMCache, Mooncake-style designs), cross-instance cache sharing, and the economics of cache hit rates for agentic workloads.

**Week 19: Long context serving.**
Serving 128K–1M+ token contexts: context parallelism, KV compression/eviction (H2O, SnapKV lineage), hybrid approaches, and honest discussion of quality tradeoffs.

### Phase 5 — The Frontier (Weeks 20–24)

**Week 20: Reasoning models and test-time compute.**
Inference when output length explodes: serving long chains of thought, the throughput/latency implications of test-time scaling, adaptive compute, and why reasoning made inference the dominant AI cost center.

**Week 21: Structured output and constrained decoding.**
Grammar-constrained generation, JSON mode internals (Outlines/XGrammar-style FSMs), function calling, and their scheduler interactions.

**Week 22: Serving agents.**
Agentic traffic patterns: bursty multi-turn loops, tool-call stalls, massive prefix reuse, session affinity and cache-aware routing, and how engines are adapting.

**Week 23: The hardware landscape.**
The hardware landscape at publication time: current NVIDIA and AMD accelerators, rack-scale systems,
TPUs/Trainium, and dataflow-style systems. Lock the comparison questions now—memory hierarchy,
bandwidth, interconnect, supported precisions, availability, power, and benchmark methodology—but
select exact products and current benchmark results four to six weeks before this installment ships.

**Week 24: The economics of inference.**
Cost per million tokens end-to-end: utilization, batching, quantization, caching, and hardware choices composed into a full cost model. Sizing and configuring a serving deployment from scratch. Sets up the finale: "now that you understand every layer, let's build one."

### Phase 6 — Capstone: Build a Mini Inference Engine in Rust (Weeks 25–30)

The payoff arc: reimplement selected serving-engine ideas in Rust, using candle for tensor operations
and the kernels it supports while making scheduler, batching, and KV memory management visible. The
Week 1 naive loop is the baseline; vLLM is the reference implementation, not a promise of performance
parity.

**Feasibility gate (complete before launch):** build a small spike in `inference-from-scratch` that
loads the exact anchor checkpoint, performs cached decoding, handles at least two variable-length
sequences, and confirms that candle exposes enough tensor and memory control for the planned cache
layout. Record unsupported operations, required custom kernels, platform constraints, and a fallback
scope. Do not advertise the detailed Week 25–30 implementation promise until this gate passes.

**Capstone acceptance criteria:** one documented model and hardware path; deterministic correctness
checks against a reference implementation; versioned benchmark inputs; continuous batching with
request admission and eviction; an explicit KV allocator whose fragmentation can be measured; raw
benchmark artifacts; and an honest accounting of features intentionally omitted. Line count is not
an acceptance criterion.

**Week 25: Skeleton + forward pass.** Project layout, candle model loading (Qwen/Llama-class), tokenizer, greedy decode loop in Rust. Feature parity with the Week 1 Python loop — and a first honest benchmark against it.

**Week 26: KV cache + sampling.** Implement the cache derived in Week 3; temperature/top-p sampling; streaming token output over an HTTP/SSE endpoint. The engine becomes a usable server.

**Week 27: Continuous batching.** The iteration-level scheduler from Week 5, in real code: request queue, in-flight batch assembly, finished-sequence eviction. Throughput curve vs. batch size.

**Week 28: Paged KV cache.** Block allocator, block tables, and fragmentation measurement. Separate
the allocator and addressing semantics that can be demonstrated without custom kernels from the
physical layout and kernel support required for real PagedAttention performance.

**Week 29: Prefix caching + scheduling polish.** Radix-style prefix reuse (Week 7), preemption, and a chunked-prefill pass (Week 8). Agent-style workloads as the test case.

**Week 30: The showdown + retrospective.** Full benchmark: our ~3,000-line engine vs. the Week 1 naive loop vs. vLLM, across TTFT/throughput/goodput, all measured with KVScope. What the remaining gap is made of (kernels, CUDA graphs, years of scheduler tuning), what we learned, and where the series goes next.

**After Week 30:** the series continues with irregular deep dives driven by what the weekly digest surfaces (new papers, new engine subsystems, case studies, interviews with engine maintainers) — plus community-requested extensions to the Rust engine.

---

## Track B: "This Week in Inference" (Weekly Digest)

### Format (keep it identical every week — readers love predictability)

1. **TL;DR** — 3 bullets: the week's most important developments.
2. **Papers** — 2–4 arXiv papers, one-paragraph each: what it claims, why it matters, what to be skeptical of.
3. **Engines & releases** — vLLM / SGLang / TensorRT-LLM / Dynamo / llm-d release notes, notable PRs and RFCs.
4. **Hardware & benchmarks** — MLPerf, InferenceMAX, vendor announcements (with a skeptical eye).
5. **Industry** — pricing changes, provider launches, notable engineering blog posts.
6. **From the archive** — one link to a relevant deep dive of yours.
7. **KVScope corner** *(occasional)* — a short note when the tool ships something: a new metric, a new backend, a reader-contributed experiment. Build-in-public beats a changelog.

### Weekly research checklist (the repeatable pipeline)

- arXiv: cs.DC + cs.LG filtered for serving/inference keywords (KV cache, speculative, quantization, disaggregat*, scheduling)
- GitHub releases/RFCs: vllm-project/vllm, sgl-project/sglang, NVIDIA/TensorRT-LLM, ai-dynamo/dynamo, flashinfer-ai/flashinfer, llm-d
- Engineering blogs: NVIDIA Developer, PyTorch, DeepSeek/Qwen/Moonshot releases, Anyscale, Modal, Together, Fireworks, Baseten, Character/Anthropic/Google infra posts
- Analysis: SemiAnalysis, The Next Platform, MLPerf result drops
- Community: r/LocalLLaMA, Hacker News "inference/vLLM/SGLang" hits, X accounts of engine maintainers
- Curated lists for backfill: Awesome-LLM-Inference, KV-cache optimization surveys

**Automation:** a scheduled task (set up separately) runs weekly, sweeps these sources, and drafts a
candidate digest. Every included item still needs a human check against the linked paper, release,
source code, or official announcement. Budget up to 90 minutes; reduce item count rather than publish
unverified coverage to satisfy a time target.

---

## Production Workflow & Growth

**Publishing rhythm:** Digest draft lands automatically → verify/edit Monday (up to 90 minutes) →
publish. Deep dives run on a two-week cycle: scope and evidence plan in week one; implementation,
measurements, draft, and figures across both weeks; final verification and publication every other
Tuesday. Track actual hours for the first four installments. Move to weekly only at a checkpoint and
only while the buffer and evidence quality remain intact.

**Voice: learning in public, receipts over résumé.** The narrator is not a veteran of production serving — and the series never pretends otherwise. Authority comes from three things the reader can verify: derivations, runnable code, and KVScope measurements. Concretely: (a) first person and honest provenance — "I measured," "this surprised me," "I got this wrong last week," never "in my experience running large fleets"; (b) confidence calibrated to evidence — state verified math and reproduced measurements plainly, hedge only what's actually untested; (c) surprise as a narrative engine — the moment something didn't behave as predicted is the most engaging paragraph in any post; (d) a visible corrections policy — a changelog on each post, and public thanks when maintainers or readers correct something (a correction from a vLLM contributor is a trophy, not a failure); (e) questions posed before answers — walk the reader through the confusion you had, then resolve it. The 30-installment arc reinforces this: the reader is watching someone actually make the journey, which is why the Rust engine finale has real stakes.

**Quality bar for deep dives:** every post has (a) at least one original diagram, (b) runnable code or
a napkin-math calculation, and (c) links to the most authoritative source available for each claim:
papers for algorithms, official documentation or source/RFCs for engine behavior, vendor
specifications for hardware, and committed benchmark artifacts for measurements. Secondary analysis
can provide context but cannot be the sole support for a plain quantitative claim. Receipts are the
substitute for tenure—this is what makes the learner voice credible rather than merely modest.

**Visuals:** static SVG figures are the default (see `planning/visual-strategy.md` for the three-tier decision and the animation shortlist). Animation is reserved for 4–6 concepts across the whole series where motion is load-bearing — the KV cache and continuous batching first — and is not touched until the cadence is proven past Week 4.

**Companion repos:** two of them. (1) The series repo — a folder per installment with code, benchmark
scripts, and diagrams; the Phase 6 Rust engine lives here as its own crate. (2) **KVScope** — the
standalone profiler, developed in public alongside the series. Every original measurement published
in the blog has a versioned JSON report committed to the series repo, produced either by KVScope or
the compatible reference harness. KVScope's roadmap phases (profiler → dashboard → optimizer
advisor) may ship near series milestones, but posts do not block on roadmap features beyond the
Week 4 MVP contract defined above.

**Benchmark contract:** every measured result records the exact model repository and revision,
precision/quantization, tokenizer, hardware and memory configuration, OS/driver/runtime and engine
versions, prompt/output length distribution, batch size or arrival/concurrency model, sampling
settings, warm-up and repetition policy, and the statistic reported. Publish TTFT and inter-token
latency distributions alongside throughput where relevant; distinguish theoretical ceilings from
measured results; commit raw reports and the command/config needed to rerun them. Cross-engine
comparisons must use equivalent model semantics and workloads, or call out the mismatch prominently.

**Running example (decided 2026-08-23):** every number, figure, and derivation is anchored to one
exact, pinned Llama-3.1-8B-class checkpoint at BF16—roughly 16 GB for weights alone. The
~210 tok/s H100 SXM batch-1 figure is a weight-bandwidth theoretical ceiling, not a promised
measurement; runtime overhead, KV traffic, and unused bandwidth lower it. Record the exact Mac model
and memory capacity before claiming local fit: an 8B BF16 checkpoint does not fit comfortably on
every Mac once runtime state and KV memory are included. A **70B is used only as a contrast**, where
scale carries a point the 8B cannot: single-device capacity, KV-cache growth at long context, and the
motivation for parallelism in Week 15. For the simplified weight-streaming derivation, the batch-1
compute-to-bandwidth imbalance is independent of parameter count because bytes and FLOPs scale
together; do not extend that simplification to workloads where KV traffic, architecture, kernels, or
parallel communication are material.

**Naming decision (open):** the KVScope README references "The Inference Engineer" as the series name; this plan uses "Inference from Scratch." Both work — "The Inference Engineer" names the reader (strong publication brand, Pragmatic-Engineer-style), "Inference from Scratch" names the promise (strong curriculum brand). Recommended: use **The Inference Engineer** as the Substack publication name and **Inference from Scratch** as the flagship series/course within it, with "This Week in Inference" as the digest. Decide before post #1 ships, then align the KVScope README to it.

**Substack setup:** two sections (Deep Dives / This Week in Inference) so readers can manage email volume; free everything for the first 3 months to build the list; consider paid tier later for the digest or "office hours" threads, keep the curriculum free forever.

**Distribution:** cross-post threads on X/LinkedIn summarizing each deep dive; submit standout posts to Hacker News and r/LocalLLaMA (genuine, not spammy); the digest is inherently shareable — end each with "forward this to someone on your infra team."

**Titles:** curriculum posts get stable numbering ("Inference from Scratch #6: PagedAttention") for course-like coherence; digests get dates ("This Week in Inference — Aug 24, 2026").

**Launch plan:** finish Weeks 1–4 before launching, have Weeks 5–6 outlined with evidence sources and
code risks identified, and complete the Rust/candle feasibility gate. Launch with Week 1 and the first
digest in the same week. Keep at least four finished deep dives buffered before considering a weekly
cadence; at the default fortnightly cadence, never let the buffer fall below two finished posts.

**Checkpoints:** pause after Weeks 4, 9, 19, and 24. Review actual production hours, buffer health,
claim corrections, benchmark reproducibility, KVScope/capstone readiness, and reader questions. The
remaining outline is allowed to change at each checkpoint; preserving the learning objective matters
more than preserving a named tool or product selected months earlier.

**Success metrics:** subscriber growth by source, archive return traffic, replies and corrections from
maintainers, successful independent reruns or code usage, and the share of quantitative claims backed
by committed artifacts. Track open rate as directional only because privacy features distort it; do
not use a fixed 45% target as the primary quality signal. The operational success metric is that the
cadence, evidence bar, and buffer all survive each checkpoint.
