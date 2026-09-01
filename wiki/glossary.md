# Glossary

Every term I had to go and look up before this field made sense, defined the way I wish someone had
defined it for me. Plain language first; precision second. Each entry names the installment that
actually teaches the idea — the definition here is a handhold, not a substitute.

**Maintenance:** this page is the term list for the pre-publish first-use audit (see the Jargon
discipline rules in `planning/series-plan.md`). When a post introduces a term, it lands here in the
same commit. This is a living page and is meant to be published as a standing reference that grows
with the series.

---

**Arithmetic intensity** — How much math you get per byte you fetch from memory, in FLOPs per byte. A
workload's *supply*. Divide the operations a piece of work performs by the bytes it has to move to
perform them. Batch-1 decode of a BF16 model is about 1: two bytes of weight arrive, two floating-point
operations happen, and the chip goes looking for the next weight. → Week 1, derived properly in Week 2.

**Autoregressive** — Generating one token at a time, where each new token is appended to the input and
the whole thing is fed back in to produce the next. There is no finished answer waiting somewhere; the
model commits one token at a time and cannot revise. → Week 1.

**Bandwidth-bound** — The work finishes when the memory system is done, not when the arithmetic is
done. Buying more compute does not help. Batch-1 decode is the canonical example. Opposite of
**compute-bound**, where the arithmetic units are the limit and faster memory would not help. → Week 1.

**Batch size** — How many independent requests the GPU processes together in one pass. Batch size 1
means one request and nothing else sharing the work, which is the worst case for efficiency and the
best case for understanding what is going on. → Week 1; batching itself is Week 5.

**BF16 / FP16 / FP8 / INT4** — Number formats, distinguished by how many bits each value uses. BF16
("brain float 16") and FP16 are 2 bytes per number; FP8 is 1; INT4 is half. The count is what matters
for inference: a model's size in bytes is roughly its parameter count times its bytes-per-parameter, and
bytes are what the memory system moves. → Week 1; the tradeoffs are Week 13.

**Chunked prefill** — Slicing one long prefill into pieces and interleaving them with other requests'
decode steps, so a single big prompt does not monopolize the GPU and spike everyone else's latency. A
scheduling fix, not a math fix — it changes no FLOPs and no bytes. → Week 7.

**Compute-bound** — See **bandwidth-bound**.

**Continuous batching** — Adding and removing requests from the running batch at every step, instead of
waiting for a whole batch to finish before starting the next. Also called iteration-level scheduling.
The single highest-leverage idea in serving. → Week 5.

**Decode** — The phase that generates output, one token per forward pass, each conditioned on
everything written so far. Irreducibly sequential, and normally bandwidth-bound at low batch. → Weeks 1–2.

**Disaggregated serving** — Running prefill and decode on separate pools of GPUs, because they stress
the hardware so differently that sharing a machine makes both worse. Requires shipping the KV cache
between them. → Week 17.

**Expert / Mixture-of-Experts (MoE)** — An architecture where each layer holds many parallel
sub-networks ("experts") and a small router picks a handful per token. A 284B-parameter MoE might
activate only 13B per token. Sparse per token, but the whole expert pool still has to live somewhere.
→ Week 14.

**FLOP** — One floating-point operation: a single multiply, or a single add. FLOPs count work done;
FLOP/s (with the "per second") measures a chip's rate. The rule of thumb for a dense model is 2 FLOPs
per parameter per token. → Week 1.

**FlashAttention** — An attention implementation that never writes the big intermediate score matrix to
main memory, keeping tiles on-chip instead. Identical math, far fewer bytes moved. The algorithm has
been frozen since 2022; each new version is pure hardware exploitation. → Week 12.

**HBM** — High Bandwidth Memory: the GPU's main memory, where model weights live. Fast compared to
system RAM, glacial compared to the chip's on-die SRAM. Every decode step reads essentially all of it.
→ Week 1.

**ITL (inter-token latency)** — The gap between consecutive streamed tokens. What "feels fast" or
"feels laggy" once generation has started. → Week 4.

**KV cache** — The saved intermediate attention values (keys and values) for tokens already processed,
kept so each step does not recompute the whole sequence from scratch. It is the optimization that makes
generation practical — and, by collapsing the arithmetic while leaving the memory traffic, the
optimization that made decode bandwidth-bound in the first place. → Week 3.

**Logits** — The raw, unnormalized scores the model produces over the vocabulary — one number per
possible next token, before any softmax or sampling. → Week 1.

**MHA / MQA / GQA / MLA** — Attention variants that differ in how many key/value heads they keep.
Fewer means a smaller KV cache, which mostly buys you room for a bigger batch rather than a directly
faster token. → Week 13.

**ops:byte ratio** — A property of a chip: its peak compute divided by its peak memory bandwidth. For
an H100, 989 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOPs per byte. Read it as a demand — to keep its arithmetic
units busy, the hardware needs that many operations performed on every byte it fetches. Also called the
**ridge point**. → Week 1.

**PagedAttention** — Storing the KV cache in fixed-size blocks scattered through memory, with a lookup
table, instead of one contiguous slab per request. Removes the fragmentation that forced engines to
over-reserve memory. Named after OS virtual memory, which it copies deliberately. → Week 6.

**Parameter** — One learned number in the model. Parameter counts (8B, 70B, 753B) are the headline, but
what the memory system moves is *bytes*, so always convert. → Week 1.

**Prefill** — The phase that processes your prompt, all tokens in one pass, before any output appears.
Normally compute-bound, and it is what TTFT measures. → Weeks 1–2.

**Prefix caching** — Reusing the KV cache across requests that share a prefix, so a repeated system
prompt is not re-processed for every user. The prefix ends at the first token that differs, which makes
prompt ordering a performance decision. → Week 7.

**Quantization** — Storing weights (and sometimes activations or the KV cache) in fewer bits. Primarily
a bandwidth optimization: half the bytes, half the time to read them. → Week 13.

**Ridge point** — See **ops:byte ratio**. The arithmetic intensity at which a workload stops being
bandwidth-bound and starts being compute-bound. → Weeks 1–2.

**Roofline model** — The chart you get by plotting what a workload supplies (arithmetic intensity)
against what the hardware demands (its ridge point). Two ceilings — one sloped for memory, one flat for
compute — and every workload sits under one of them. The standard way to reason about which resource is
actually the limit. → Week 2.

**Sampling / temperature / top-k / top-p** — Turning the logits into an actual choice. Temperature
rescales the distribution (lower is more deterministic), top-k keeps the k highest-scoring tokens, top-p
(nucleus) keeps the smallest set whose probabilities sum past p. Greedy decoding skips all of it and
takes the maximum. → Week 1.

**Speculative decoding** — Having a small fast model guess several tokens ahead, then verifying them all
in one pass of the big model. Provably does not change the output distribution — the only major
optimization with literally zero quality cost. → Week 16.

**SRAM** — The small, extremely fast memory on the GPU die itself, a few hundred KB per streaming
multiprocessor. Roughly 10× the bandwidth of HBM and roughly 1/100,000th the capacity. Most kernel
optimization is the art of getting your working set into it. → Week 12.

**Tensor / pipeline / expert parallelism (TP / PP / EP)** — Three ways to split a model that does not
fit on one GPU: TP splits individual layers across GPUs (chatty, must stay inside a node), PP assigns
whole layers to different GPUs, EP distributes MoE experts. → Week 15.

**Throughput vs latency** — Throughput is tokens per second across everyone using the machine; latency
is how long *your* request takes. Batching improves the first and worsens the second. Almost every
serving decision is a position on this tradeoff. → Week 4.

**Token / tokenizer** — Models do not read text; they read integer IDs from a fixed vocabulary. The
tokenizer does the conversion. A token is roughly a word-piece — sometimes a whole word, often a
fragment. → Week 1.

**TTFT (time to first token)** — How long from sending a request until the first token appears.
Dominated by prefill. The metric users experience as "did it hang?" → Week 4.

**VRAM / unified memory** — VRAM is a discrete GPU's dedicated memory; Apple silicon instead shares one
pool between CPU and GPU ("unified memory"). The tradeoff is capacity versus bandwidth: Apple gives you
much more room, a discrete GPU moves bytes much faster. → Weeks 1, 23.
