# The Life of a Token

### Inference from Scratch #1 — What actually happens when an LLM generates a token

*This is the first post in **Inference from Scratch** — me learning LLM inference in public, from a naive PyTorch loop toward a small serving engine in Rust. I'm not an inference veteran; the deal is that every claim here is derived, reproduced, or traced to an authoritative source, and when I get something wrong I'll correct it in the open. A new deep dive arrives every other Tuesday; a separate weekly digest, This Week in Inference, covers what changed in the field each week.*

---

Type a prompt into a fast LLM service and a familiar pattern appears: the prompt is processed in one burst, then the answer streams out token by token. Why can reading a block of text feel fast while writing the continuation is irreducibly sequential?

I used PyTorch for years before I could honestly answer that question — I could train and fine-tune models, but what happened *physically* on the GPU between two words of a streamed response was a black box. This post is the one I wish I'd read back then. We'll build the dumbest possible inference loop in ~50 lines of PyTorch, run it, and derive a one-line formula that predicts LLM generation speed on hardware from a MacBook to an H100 — out of two spec-sheet numbers and a single division. Whether the prediction survives contact with real silicon is Week 4's job, and I'll publish the misses along with the hits.

That formula introduces the thesis of this whole series: **for conventional low-batch decode, moving data is often the binding constraint.** KV caches, paged attention, continuous batching, quantization, speculative decoding, and disaggregated serving attack different parts of the wider serving problem. But first, the machine itself.

## The cast of characters

Strip away the product surface and the token-generation path reduces to five conceptual components.

The **tokenizer** turns your string into a sequence of integer IDs from a fixed vocabulary. In this example it runs on the CPU before the timed model loop, and it is the last time the input is ordinary text.

The **embedding table** maps each token ID to a vector—one row lookup in a matrix of shape
`[vocab_size, hidden_dim]`. Your prompt is now a stack of model-width vectors.

The **transformer stack** — many layers of attention and MLP blocks — is where essentially all the parameters and FLOPs live. Each layer mixes information across positions (attention) and across features (MLP), then writes an updated representation.

The **LM head** projects the final hidden state of the *last* position back onto the vocabulary: one
`[hidden_dim, vocab_size]` matmul producing one raw score per vocabulary token—the *logits*. Note
what this means: however long your prompt, the model's entire output at each step is a single
distribution over "what token comes next."

The **sampler** turns that distribution into a choice — the only explicitly stochastic step in this loop. Temperature rescales the logits, top-k/top-p prune the tail, and a random draw picks the winner. The tokenizer decodes that token ID to a text fragment, which can then stream to your screen.

## The loop

Here's the part that surprised me when I first traced through it: there is no separate
sentence-level answer buffer and no standard decoder lookahead. At each step, the model acts like a
function:

> **(all tokens so far) → (probability distribution over the next token)**

Generation is just calling that function in a loop, appending each chosen token, and calling it again. A long answer requires one sequential decision per generated token, each conditioned on what was already written. Everything we call an "inference engine" — vLLM, SGLang, TensorRT-LLM — is orchestration wrapped around this loop.

![Autoregressive decoding as a cycle: the model's output token becomes part of its next input](figures/fig1-the-loop.png)

So let's write the loop with no orchestration at all.

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # swap in an 8B model if you have the VRAM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto"
).eval()

def generate(prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    messages = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = ids.shape[1]
    t0, n = time.perf_counter(), 0
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(ids, use_cache=False).logits[0, -1]   # full re-forward, every step
        if temperature <= 0:
            next_id = torch.argmax(logits).reshape(1)            # greedy decode
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p              # nucleus filter
            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
            next_id = torch.multinomial(probs / probs.sum(), 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
        n += 1
        if next_id.item() == tok.eos_token_id:
            break
    dt = time.perf_counter() - t0
    print(f"{n} tokens in {dt:.2f}s -> {n/dt:.1f} tok/s")
    return tok.decode(ids[0, prompt_len:], skip_special_tokens=True)

print(generate("The capital of France is"))
```

Run it. It works — this is a minimal autoregressive generator, with none of the request handling,
streaming, batching, stop controls, or observability that would make it a server. It is also
*catastrophically* slow, and slower per-token the longer the output gets. Note the `use_cache=False`:
every step re-processes the entire sequence from scratch, so step 500 redoes all the work of steps
1–499. Fixing exactly that wastefulness is Week 3 (the KV cache). This loop is our pedagogical
baseline. The printed timing is only a demonstration; reproducible comparisons use the repository's
benchmark script, which fixes the workload, warms up the device, synchronizes it, and reports raw
results.

The repository ships two reference configurations: **Qwen2.5-0.5B** as the accessible teaching
model and the pinned **Llama-3.1-8B** checkpoint as the numerical anchor. The generation path is the
same; the hardware requirements are not.

## The napkin math that explains everything

Now the payoff. Here's the question that cracked this whole topic open for me: when the model generates one token at batch size 1, what is the GPU actually doing? I assumed the answer was "a lot of matrix math." I was wrong in an interesting way.

For one conventional decode step through our dense 8B anchor, the GPU must use essentially every
dense-layer weight. Those weights live in HBM, the GPU's main memory, and are far too large to remain
in its on-chip caches. At batch size 1 there is very little work with which to amortize reading them.
This statement is deliberately scoped: embedding lookups touch selected rows, MoE models activate
selected experts, and later techniques such as batching and speculative decoding change how much
useful work one weight load can produce.

An 8B-parameter model in BF16 is roughly 16 GB of weights. An H100 SXM's advertised peak HBM
bandwidth is 3.35 TB/s. For standard one-token-at-a-time, batch-1 decode—without quantization or
speculation—the idealized weight-bandwidth ceiling is:

> **3,350 GB/s ÷ 16 GB ≈ 210 tokens/second** — a peak-rate upper bound, not a performance prediction.

Meanwhile, how much *math* does the dense part of that token take? Each weight participates in
roughly one multiply-add, so the familiar rule of thumb is 2 FLOPs per parameter, or about 16
GFLOPs. (That approximation is doing real work here; in the next installment we derive it and find
where it stops holding.) Against a dense BF16 peak of roughly 989 TFLOP/s, that gives a separate
compute-time lower bound of about **16 microseconds**. The weight-read lower bound is about **4.8
milliseconds**.

Those are not two phases of a measured timeline. Real kernels overlap memory movement with
arithmetic, sustain neither advertised peak, and batch-1 matrix-vector operations use the compute
units poorly. The useful comparison is the **roughly 295:1 gap between the hardware's peak compute
and bandwidth balance**. Conventional batch-1 decode offers too little arithmetic per byte to use
most of the H100's compute capacity.

![Peak-rate lower bounds for one decode step: roughly 4,800 µs for the weight read versus 16 µs for dense arithmetic, explicitly not a measured timeline](figures/fig2-time-budget.png)

Run the same idealized arithmetic for a dense 70B at BF16 and you get 140 GB of weights, a ~42 ms
weight-read lower bound, and a ~142 µs dense-compute lower bound. The imbalance is again **295×**.
It has to be: in this simplified derivation, bytes and FLOPs both scale with parameter count, so the
model size cancels. What's left is a property of the *chip*: 989 TFLOP/s ÷ 3.35 TB/s ≈ **295 FLOPs
per byte**, the H100's peak ops:byte ratio. An operation with arithmetic intensity well below that
ridge point is bandwidth-bound in the roofline model—and batch-1 dense decode is roughly one FLOP
per weight byte at BF16.

Within that simplified dense-model calculation, model size does not change the imbalance; it changes
the capacity requirement and the absolute ceiling. A 70B model's roughly 140 GB of BF16 weights does
not fit on one 80 GB H100, and one H100's worth of peak bandwidth would imply only about 24 tok/s.
That is part of the motivation for splitting large models across GPUs (Week 15). The rest of the
series examines how real systems shrink, avoid, amortize, or relocate different kinds of data
movement—and where compute or communication becomes the next bottleneck.

This one formula — **weights ÷ bandwidth** — travels remarkably well:

![Idealized weight-bandwidth ceilings for an 8B BF16 model using advertised peak bandwidth across two MacBook Pro configurations, an RTX 4090, and an H100](figures/fig3-bandwidth-ceilings.png)

- **Two current MacBook Pro examples:** Apple lists 153 GB/s for the base M5 and 614 GB/s for the
  highest-bandwidth M5 Max configuration. The formula gives idealized BF16 ceilings of roughly 10 and
  38 tok/s respectively, provided the selected unified-memory capacity can hold the weights plus
  runtime and KV state. A 4-bit representation shrinks the weight term substantially, but real speedup
  also depends on the quantized kernels and dequantization overhead. These are predictions from
  [Apple's specifications](https://www.apple.com/macbook-pro/specs/), not my measurements.
- **H100 serving one conventional batch-1 stream:** the same calculation gives an idealized
  ~210 tok/s ceiling from [NVIDIA's 3.35 TB/s specification](https://www.nvidia.com/en-us/data-center/h100/).
- **The riddle from the top:** prefill processes the prompt tokens together, which creates much more
  opportunity to reuse weights and raise arithmetic intensity. Whether a particular prefill is
  actually compute-bound depends on sequence length, batch, model, kernels, and hardware. Decode must
  perform another sequential step for every output token and, at low batch, commonly remains
  bandwidth-bound. That conditional asymmetry is the next installment.

And notice the opportunity hiding in the arithmetic: a batch can reuse each loaded weight across
multiple sequences. The weight-read term is amortized over more useful tokens, while activation and
KV traffic, arithmetic, and scheduling work still grow with the batch. Step time is not fixed, but
throughput can rise dramatically until compute or another resource becomes the bottleneck. Weeks
5–8 cover the machinery that makes this dynamic batching practical.

## Four things I had wrong about this loop

**"The decoder has a finished answer waiting."** Standard autoregressive decoding does not keep a
sentence-level output buffer or revise future tokens: it commits one token at a time. That is an
implementation claim, not a claim about what internal representations may encode. (The interesting
wrinkle, speculative decoding, *guesses* ahead and verifies; Week 16.)

**"Longer answers are slower only because the questions are harder."** Every additional token adds
another sequential decode step, regardless of whether it contains a breakthrough or boilerplate.
The per-token cost is not perfectly constant—attention and KV traffic grow with context, and batching
state changes—but output length remains a first-order cost driver. That is why long reasoning traces
changed inference economics (Week 20).

**"Generation is slow because the GPU lacks arithmetic throughput."** The roofline comparison points
the other way: conventional batch-1 decode exposes far less arithmetic per byte than an H100 needs to
use its peak compute capacity. The exact utilization is a measurement, not something this napkin math
can supply. This was the single biggest update to my mental model—and I took it seriously only after
two independent derivations reached the same resource imbalance.

**"Removing sampling randomness guarantees bitwise reproducibility."** Greedy decoding removes the
explicit random draw, but a production stack can still use nondeterministic kernels or change
floating-point reduction order with batching and kernel selection. I have not yet reproduced the
batch-companion effect; that needs the serving setup from Week 5, so treat it as a sourced warning
rather than one of my measurements.

## Where this series goes

We're deliberately ignoring three enormous things today: the re-computation waste in our loop (**KV cache**, Week 3), the fact that we served exactly one request (**batching**, Week 5 — the single highest-leverage idea in serving), and everything about kernels, quantization, parallelism, and scale (Phases 3–5). We'll also build the measurement harness this series will use for every benchmark (Week 4).

And it all converges somewhere concrete: the final arc is aimed at a small inference engine in Rust,
benchmarked against both today's naive loop and vLLM. Before publishing that detailed build promise,
I'm proving the model-loading, cached-decoding, variable-length batching, and memory-layout path in a
feasibility spike. The outcome of that work—not a line-count slogan—will determine the exact scope.

**Next installment:** *Prefill vs. decode — why reading and writing stress the hardware differently.*
The two-phase workload hiding inside every request, the roofline model, and the variables that move
each phase between bandwidth- and compute-bound regimes.

**Where the numbers came from.** The peak-rate lower bounds use official hardware specifications
(H100 SXM: 3.35 TB/s HBM and roughly 989 TFLOP/s dense BF16; current MacBook Pro bandwidths linked
above) and are cross-checked against two 2026 books that reach the same roofline conclusion by
different routes. Philip Kiely's *Inference Engineering* (Baseten) works through an H100 ops:byte
ridge point of roughly 295; Vizuara's *Inference Engineering: The Definitive Workshop Guide* derives
arithmetic intensity of roughly one for dense batch-1 decode. The ratio compares hardware resource
ceilings. It is not a measured utilization percentage.

Three caveats matter. First, advertised bandwidth and compute are ceilings; achieved rates are lower.
Second, kernels overlap memory movement and arithmetic, so the two lower bounds cannot be pasted
together as a literal timeline. Third, KV reads, activations, sampling, framework overhead, and—in a
distributed deployment—communication add work that the weights-only formula omits. That is exactly
why the formula is useful as a falsifiable ceiling rather than a promised speed.

Where I've measured something myself I say so; where I'm standing on someone else's arithmetic, this is the someone else.

**Changelog.** Nothing yet. Corrections land here with credit — if you spot an error, you'll be named.

*Code for this post—including `napkin.py` and the synchronized, warm-started cached benchmark used to
test the ceiling—is in the
[companion repo](https://github.com/stp8954/inference-from-scratch/tree/main/week-01-life-of-a-token).
Run `bench_decode.py --cache` on your hardware and tell me how far below the ceiling it lands; I'll
publish a reader-submitted table with the exact configurations. If you spot an error, say so—corrections
get credited in the changelog. And if you want to learn this alongside me, subscribe: one reproducible
deep dive every other Tuesday, from scratch to the state of the art.*
