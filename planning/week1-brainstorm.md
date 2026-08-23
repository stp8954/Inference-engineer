# Week 1 Brainstorm — "What actually happens when an LLM generates a token"

*Inference from Scratch #1. The job of this post: hook ML engineers, establish the series' systems framing, and earn the subscribe. It's the most-read post you'll ever write in this series (every later post links back to it), so it should be opinionated and concrete, not a survey.*

---

## Angle options (pick one as the spine)

**Angle A — "Follow one token through the machine" (narrative walkthrough).**
Trace a single request — `"The capital of France is"` — from HTTP request to the word "Paris" appearing. Tokenizer → embedding lookup → 32 transformer layers → LM head → logits → sampling → detokenize → repeat. At each stop, note what's expensive and what's cheap. The narrative device gives the post momentum and gives you a map you can zoom into for the next 23 weeks.
*Pro:* extremely linkable ("the classic 'life of a token' post"). *Con:* it's been done — you win on depth and numbers, not novelty.

**Angle B — "Your GPU is 95% idle" (provocation-first).**
Open with the punchline usually saved for Week 2: during decode, an H100 doing single-request generation uses a tiny fraction of its compute — generation is a memory-bandwidth problem wearing a compute costume. Then rewind and explain the generation loop to justify the claim. Napkin math as the hero: params × 2 bytes ÷ memory bandwidth = a hard floor on tokens/sec, and it predicts real latency shockingly well.
*Pro:* strongest hook for ML engineers; immediately signals "this series does real numbers." *Con:* steals some of Week 2's thunder — you'd re-scope Week 2 toward the roofline model and prefill.

**Angle C — "Build the dumbest possible inference server" (code-first).**
Start from `model.generate()` being a black box. Write the ~50-line greedy loop in raw PyTorch, run it on a small model (Qwen 0.5B / Llama 1B class), measure tokens/sec, and end with "here's why this is 100x slower than vLLM — this series explains every one of those 100x." The gap between naive and production *is* the series thesis.
*Pro:* hands-on, repo starts week one, the benchmark becomes the series' recurring baseline. *Con:* code-heavy posts get skimmed in email; needs a strong prose layer on top.

**Recommendation: A as the spine, with B's napkin math as the centerpiece section and C's code as the companion repo.** The narrative carries email readers, the math delivers the "aha," the code serves the ones who want to go deeper. That composite is the draft outline below.

---

## Title candidates

- "What Actually Happens When an LLM Generates a Token" — clear, searchable, does the job.
- "The Life of a Token" — elegant, riffs on the classic "life of a packet" systems posts; subtitle carries the specifics.
- "Your LLM Is Just a Very Expensive Autocomplete Loop (Here's the Loop)"
- "Inference from Scratch #1: One Token at a Time"
- "It Takes a Whole Datacenter to Say 'Paris'"

Front-runners: #1 for SEO, #2 for brand. You can use #2 as the title and #1 as the subtitle.

## Hook options (first 3 sentences decide the open rate on post #2)

1. **The latency riddle:** "Type a prompt into ChatGPT and the first word appears in half a second. Each word after that also takes tens of milliseconds — even though the model 'read' your whole prompt at once. Why is reading 1,000 words fast, but writing the 1,001st slow? By the end of this post you'll know exactly why — and it isn't the reason most people think."
2. **The cost frame:** "Every token a frontier model emits costs someone real money — and in 2026, inference, not training, is where most AI compute is burned. This series is about that machine: what it does, why it's expensive, and how the best teams make it 10–100x cheaper."
3. **The confession:** "I used PyTorch for years before I could honestly answer: what happens, physically, on the GPU between two words of a streamed response? This is the post I wish I'd read."

## The napkin-math centerpiece (the section readers will screenshot)

Memory-bandwidth floor for batch-1 decode: every generated token must read every weight once.

- 8B model @ FP16 ≈ 16 GB of weights. H100 SXM ≈ 3.35 TB/s. → 3350/16 ≈ **~210 tokens/s ceiling**, and real engines get within striking distance of it.
- Same model on a MacBook (M-series, ~120–400 GB/s) → 8–25 tok/s, which is exactly what llama.cpp users see. One formula explains both.
- Flip side: prefill processes all prompt tokens in one pass — that's why 1,000 prompt tokens don't take 1,000× as long as one generated token. (Full treatment: Week 2.)

This section quietly plants the series' three big seeds: memory bandwidth is destiny (→ quantization, Week 10), reading weights once per token is wasteful if you're only serving one request (→ batching, Week 5), and recomputing attention over the past would be insane (→ KV cache, Week 3 — mention it exists, don't derive it yet).

## Draft outline (Angle A spine)

1. Hook (riddle) + what this series is and who it's for (2 short paragraphs).
2. The cast: tokenizer, embeddings, transformer stack, LM head, sampler — one tight paragraph each, one full-loop diagram.
3. The loop: autoregression made concrete — the model is a pure function `(tokens so far) → (distribution over next token)`; everything else is orchestration around calling it repeatedly.
4. Sampling: logits → temperature → top-p/top-k → chosen token. Runnable snippet; myth-bust that temperature=0 guarantees determinism (batching and floating-point nondeterminism say otherwise — great "engineers will nod" detail).
5. Napkin math centerpiece (above).
6. Run it: the 50-line loop from Angle C, measured tokens/sec, "this is our baseline for the whole series."
7. What we're deliberately ignoring (KV cache, batching, kernels) = the series roadmap, phase by phase. End with Week 2 teaser: "why reading is cheap and writing is expensive."

Target: ~2,000 words + 2 diagrams + 1 code block inline, rest in the repo.

## Diagram ideas

- **The loop:** circular flow — prompt tokens in → forward pass → logits → sample → token appended → back into the model. Make the arrow back into the model visually heavy; that arrow is the whole story.
- **The napkin:** a "weights must flow" visual — 16 GB of weights streaming through a 3.35 TB/s pipe per token, with the tok/s ceiling as the punchline annotation.

## Misconceptions worth explicitly killing (great for engagement/replies)

- "The model plans the whole sentence" — no, one token at a time, no lookahead (speculative decoding nuance comes Week 16).
- "Longer answers are slower because the model thinks harder" — no, it's the same forward pass every token; output length is a *cost* multiplier, not a difficulty signal.
- "Generation is slow because GPUs are slow at math" — the compute is nearly idle at batch 1; it's a memory problem.
- "temperature=0 means reproducible outputs" — not in production serving.

## Companion repo (week1/)

`naive_generate.py` (the 50-line loop), `napkin.py` (the bandwidth-floor calculator: model size, dtype, hardware → predicted ceiling vs. measured), README with results table on 2–3 hardware targets. The napkin calculator is a great share-bait mini-tool on its own.

## Open questions for you

- Small model choice for runnable examples: Qwen-0.5B-class (runs anywhere, incl. Colab free tier) vs. Llama-8B-class (numbers match the napkin math section)? Could do both: 0.5B to run, 8B for the math.
- Do you want a "Prerequisites" box (comfortable with PyTorch + transformer basics) to set the ML-engineer bar explicitly in post #1?
- Series name check: "Inference from Scratch" — locked, or brainstorm alternatives before you ship #1?
