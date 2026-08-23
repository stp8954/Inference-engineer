# LLM Inference Blog Series — Master Plan

**Audience:** ML engineers who deploy and serve models — comfortable with PyTorch, CUDA-adjacent concepts, and systems thinking, but who want a rigorous ground-up treatment of inference.

**Platform:** Substack (email-first). Two publications-in-one via Substack sections: a **Deep Dive** section and a **This Week in Inference** section, so subscribers can opt into either or both.

**Cadence:** Two posts per week.

- **Deep Dive** (educational track) — publishes a fixed day, e.g., Tuesday. ~1,500–2,500 words, code + diagrams. Evergreen.
- **This Week in Inference** (news track) — publishes Friday or Monday. ~600–1,000 words. Timely, skimmable, link-heavy.

The two tracks reinforce each other: the digest cites your own deep dives ("we covered paged attention in Week 5"), which drives archive traffic; the deep dives stay relevant because readers see the concepts appear in the news.

---

## Track A: The Deep Dive Curriculum (Beginner → SOTA)

A 30-week arc in six phases. Each post ends with a "what's next" teaser and links back to prerequisites, so the archive works as a self-serve course.

### Phase 1 — Foundations (Weeks 1–4)

**Week 1: What actually happens when an LLM generates a token.**
The forward pass viewed as a serving problem. Autoregressive decoding, logits → sampling, temperature/top-p. Build a naive `generate()` loop in ~50 lines of PyTorch. Establish the series' framing: inference is a *systems* problem, not a modeling problem.

**Week 2: Prefill vs. decode — the two-phase workload.**
Why processing the prompt and generating tokens are completely different computational regimes: compute-bound vs. memory-bandwidth-bound. Arithmetic intensity, the roofline model. This single distinction explains half of everything that follows in the series. First KVScope cameo: show real prefill/decode timing splits from the profiler as the post's evidence ("see this yourself" — full tool introduction comes in Week 4).

**Week 3: The KV cache — why it exists and what it costs.**
Derive the KV cache from the attention equation. Compute its memory footprint by hand for Llama-class and MoE models. Show the quadratic → linear compute tradeoff and why memory, not FLOPs, is the binding constraint of decode.

**Week 4: Measuring inference — the metrics that matter (KVScope debut).**
TTFT, TPOT/ITL, throughput vs. goodput, latency percentiles, tokens/s/GPU, cost per million tokens. How benchmarks lie (batch size games, input/output length mixes). Official introduction of **KVScope** ([github.com/stp8954/KVScope](https://github.com/stp8954/KVScope)) as the series' companion profiler: prefill/decode timing separation, KV cache growth tracking, arithmetic-intensity estimation, and JSON reports. Every benchmark in the rest of the series runs through it. (Tease it in Week 2 — "here's how to see prefill vs. decode yourself" — but keep Week 1 tool-free; the naive loop must feel primitive.)

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
Architecture tour of the major open engines: where the scheduler, model runner, and kernels live; CUDA graphs; when to pick which. Includes a hands-on KVScope benchmark across all three engines — also the milestone for KVScope's planned vLLM backend.

### Phase 3 — Making the Model Cheaper (Weeks 10–14)

**Week 10: Quantization I — the ideas.**
Number formats (FP16/BF16 → FP8 → INT4/NVFP4/MXFP4), weight-only vs. weight+activation, calibration, outliers. Why quantization is nearly free for decode but tricky for prefill quality.

**Week 11: Quantization II — the methods in practice.**
GPTQ, AWQ, FP8 (per-tensor vs. per-block scaling), KV cache quantization, QAT vs. PTQ. What actually ships in production and how to evaluate quality degradation properly.

**Week 12: Attention kernels — FlashAttention and friends.**
Why attention was memory-bound, tiling and online softmax, FlashAttention 1→4 lineage, FlashInfer, decode-specific kernels (flash-decoding, split-KV). Light-touch kernel reading, not kernel writing.

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
The architecture shift in large-scale serving: separate prefill and decode fleets, KV cache transfer (NIXL/UCX-style), and orchestration layers like NVIDIA Dynamo and llm-d. Why hyperscalers converged on this.

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
Blackwell/Blackwell Ultra-class GPUs and rack-scale systems (NVL72), AMD MI-series, custom silicon (TPU, Trainium, Groq/Cerebras-style dataflow), memory bandwidth as destiny, and reading MLPerf/InferenceMAX benchmarks critically.

**Week 24: The economics of inference.**
Cost per million tokens end-to-end: utilization, batching, quantization, caching, and hardware choices composed into a full cost model. Sizing and configuring a serving deployment from scratch. Sets up the finale: "now that you understand every layer, let's build one."

### Phase 6 — Capstone: Build a Mini Inference Engine in Rust (Weeks 25–30)

The payoff arc: reimplement the core ideas of a vLLM-class engine in Rust, built on candle so GPU kernels come for free and the interesting code is ours — the scheduler, batching, and KV memory management from Phases 2–4 made real. The Week 1 naive loop is the baseline the engine must beat; vLLM is the yardstick it's measured against.

**Week 25: Skeleton + forward pass.** Project layout, candle model loading (Qwen/Llama-class), tokenizer, greedy decode loop in Rust. Feature parity with the Week 1 Python loop — and a first honest benchmark against it.

**Week 26: KV cache + sampling.** Implement the cache derived in Week 3; temperature/top-p sampling; streaming token output over an HTTP/SSE endpoint. The engine becomes a usable server.

**Week 27: Continuous batching.** The iteration-level scheduler from Week 5, in real code: request queue, in-flight batch assembly, finished-sequence eviction. Throughput curve vs. batch size.

**Week 28: Paged KV cache.** Block allocator, block tables, fragmentation measurement — PagedAttention's memory management (Week 6) minus the custom kernels, with honest notes on what the real kernels buy.

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

**Automation:** a scheduled task (set up separately) runs weekly, sweeps these sources, and drafts the digest for you to edit — target: you spend ≤1 hour editing, not 4 hours hunting.

---

## Production Workflow & Growth

**Weekly rhythm:** Digest draft lands automatically → edit Monday (1h) → publish. Deep dive: outline Tuesday of the *prior* week, draft over 2–3 sessions (~4–6h), diagrams + code repo Friday, publish Tuesday.

**Voice: learning in public, receipts over résumé.** The narrator is not a veteran of production serving — and the series never pretends otherwise. Authority comes from three things the reader can verify: derivations, runnable code, and KVScope measurements. Concretely: (a) first person and honest provenance — "I measured," "this surprised me," "I got this wrong last week," never "in my experience running large fleets"; (b) confidence calibrated to evidence — state verified math and reproduced measurements plainly, hedge only what's actually untested; (c) surprise as a narrative engine — the moment something didn't behave as predicted is the most engaging paragraph in any post; (d) a visible corrections policy — a changelog on each post, and public thanks when maintainers or readers correct something (a correction from a vLLM contributor is a trophy, not a failure); (e) questions posed before answers — walk the reader through the confusion you had, then resolve it. The 30-week arc reinforces this: the reader is watching someone actually make the journey, which is why the Rust engine finale has real stakes.

**Quality bar for deep dives:** every post has (a) at least one original diagram, (b) runnable code or a napkin-math calculation, (c) links to primary sources (papers, not other blogs). Receipts are the substitute for tenure — this is what makes the learner voice credible rather than merely modest.

**Visuals:** static SVG figures are the default (see `planning/visual-strategy.md` for the three-tier decision and the animation shortlist). Animation is reserved for 4–6 concepts across the whole series where motion is load-bearing — the KV cache and continuous batching first — and is not touched until the cadence is proven past Week 4.

**Companion repos:** two of them. (1) The series repo — a folder per week with code, benchmark scripts, and diagrams; the Phase 6 Rust engine lives here as its own crate. (2) **KVScope** — the standalone profiler, developed in public alongside the series. Every measurement published in the blog is a KVScope JSON report committed to the series repo, so results are reproducible; blog traffic drives stars, and original tool-generated measurements differentiate the blog from newsletters recycling vendor benchmarks. KVScope's own roadmap phases (profiler → dashboard → optimizer advisor) ship as series milestones — but posts never block on tool features; the Phase 1 profiler is the only hard dependency (needed by Week 4).

**Running example (decided 2026-08-23):** every number, figure, and derivation is anchored to a **Llama-3.1-8B-class model at BF16** — 16 GB of weights, ~210 tok/s batch-1 ceiling on an H100 SXM. A **70B is used only as a contrast**, where scale carries a point the 8B cannot: capacity that overflows a single GPU, KV-cache growth at long context, and the motivation for parallelism in Week 15. One caution when reaching for it — the batch-1 compute-to-bandwidth imbalance is *model-independent*. It equals the hardware's ops:byte ratio (~295 on H100), because bytes and FLOPs both scale with parameter count. A bigger model changes the absolute ceiling and what fits in VRAM, not the ratio. (An early Week 1 draft claimed a distinct "280× imbalance" for 70B; it is the same 295×.)

**Naming decision (open):** the KVScope README references "The Inference Engineer" as the series name; this plan uses "Inference from Scratch." Both work — "The Inference Engineer" names the reader (strong publication brand, Pragmatic-Engineer-style), "Inference from Scratch" names the promise (strong curriculum brand). Recommended: use **The Inference Engineer** as the Substack publication name and **Inference from Scratch** as the flagship series/course within it, with "This Week in Inference" as the digest. Decide before post #1 ships, then align the KVScope README to it.

**Substack setup:** two sections (Deep Dives / This Week in Inference) so readers can manage email volume; free everything for the first 3 months to build the list; consider paid tier later for the digest or "office hours" threads, keep the curriculum free forever.

**Distribution:** cross-post threads on X/LinkedIn summarizing each deep dive; submit standout posts to Hacker News and r/LocalLLaMA (genuine, not spammy); the digest is inherently shareable — end each with "forward this to someone on your infra team."

**Titles:** curriculum posts get stable numbering ("Inference from Scratch #6: PagedAttention") for course-like coherence; digests get dates ("This Week in Inference — Aug 24, 2026").

**First-month launch plan:** write Weeks 1–3 *before* launching so you have a buffer and new subscribers land on substance; launch with Week 1 + your first digest in the same week; buffer never drops below 2 weeks of deep dives.

**Success metrics:** subscriber count and open rate (digest should hit 45%+), archive traffic to early curriculum posts (signals the course structure works), and replies/corrections from engine maintainers (signals credibility with the target audience).
