# The Life of a Token

### Inference from first principles #1 — What actually happens when an LLM generates a token

*This is the first post in a series that starts from a naive generation loop and ends in a small serving engine. I am learning this in public. Every claim is derived, reproduced, or traced to a source; when I get something wrong I will correct it here.*

**Series contract (locked for this curriculum)**

| | Pin |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` at BF16 (~8B dense parameters, ~16 GB of weights) |
| Lab GPU | one cloud **H100 SXM 80GB** (advertised 3.35 TB/s HBM, ~989 TFLOP/s dense BF16) |
| What “I measured” means | that box, that checkpoint, a pinned image, a checked-in harness |
| What this post is | first principles + spec-sheet **ceilings**. Measured tok/s come later |

A 0.5B Qwen checkpoint appears in the code block so you can run the loop on a smaller GPU. It is a teaching toy. Every ceiling, figure, and later benchmark in the series is the 8B on the H100 unless a post says otherwise.

---

Type a prompt into a fast LLM service and a familiar pattern appears: the prompt is processed in one burst — the **prefill** — then the answer streams out one token at a time, each one a **decode** step. Why can reading a block of text feel fast while writing the continuation is usually sequential?

I used PyTorch for years before I could honestly answer that. I could train and fine-tune models. What happened *physically* on the GPU between two words of a streamed response was a black box. This post is the one I wish I had read then.

We will do two things only:

1. Write the dumbest possible generation loop — the thing every inference engine is wrapping.
2. Derive a one-line **ceiling**: advertised HBM bandwidth divided by the bytes of weights. On our H100 pin that is about 210 tok/s for batch-1 BF16 decode. That is an upper bound from a spec sheet, not a speed I have measured.

That ceiling is the thesis of the series: **for conventional low-batch decode, moving weights is often the binding constraint.** KV caches, paged attention, continuous batching, quantization, speculative decoding, disaggregated serving — if those words mean nothing yet, good. Each one is an attack on some part of the same problem. The last arc of the series is a small Rust engine that makes those attacks visible on the same H100, next to vLLM. A feasibility spike (load, cached decode, ragged batch, layout) decides the exact engine scope. Not a line-count slogan.

## The cast of characters

Strip away the product surface and the token-generation path reduces to five parts.

The **tokenizer** turns your string into integer IDs from a fixed vocabulary. In the loop below it runs on the CPU before the timed model call. After that the input is not text.

The **embedding table** maps each token ID to a vector — one row lookup in a matrix of shape `[vocab_size, hidden_dim]`. The prompt is now a stack of model-width vectors.

The **transformer stack** — many layers of attention and MLP blocks — is where almost all the parameters and FLOPs live. Each layer mixes information across positions (attention) and across features (MLP), then writes an updated representation.

The **LM head** projects the final hidden state of the *last* position onto the vocabulary: one `[hidden_dim, vocab_size]` matmul, one raw score per vocab token — the *logits*. However long the prompt, the necessary output of a decode step is a single distribution over “what token comes next.”

The **sampler** turns that distribution into a choice — the only explicitly random step in this loop. Temperature rescales the logits, top-k / top-p prune the tail, a draw picks the winner. The tokenizer turns that ID back into a text fragment.

## The loop

There is no separate sentence-level answer buffer and no standard lookahead. At each step the model is a function:

> **(all tokens so far) → (probability distribution over the next token)**

Generation is calling that function, appending the chosen token, and calling it again. A long answer is one sequential decision per token. vLLM, SGLang, TensorRT-LLM, and the engine this series will build are orchestration around this loop.

![Autoregressive decoding as a cycle: the model's output token becomes part of its next input](figures/fig1-the-loop.png)

Here is the loop with no orchestration. The 0.5B checkpoint is so you can run it without 16 GB of VRAM. Swap the `MODEL` line for `meta-llama/Llama-3.1-8B-Instruct` when you are on the series box.

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Teaching toy. Series pin is meta-llama/Llama-3.1-8B-Instruct at BF16 on an H100 SXM.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
            # Full-sequence forward. Only the last logit is used.
            logits = model(ids, use_cache=False).logits[0, -1]
        if temperature <= 0:
            next_id = torch.argmax(logits).reshape(1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
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

This is a generator, not a server. No request handling, streaming, batching, stop sets, or metrics.

It is also doing extra work the *story* of decode does not require. With `use_cache=False`, Hugging Face still runs every position in the sequence and produces logits at every index. We then take `[:, -1]`. Step 500 recomputes steps 1–499. That waste is the **KV cache**: keep the attention keys and values instead of rebuilding them. Later in the series. Until then, treat KV as “the results this loop throws away.”

The printed tok/s is a demo. It does not synchronize the GPU and it includes sampling and `torch.cat`. Real comparisons use a harness that pins the 8B checkpoint, warms up, synchronizes, and reports the raw log on the H100. That comes later.

## The napkin math that explains the constraint

When the 8B generates one token at batch size 1 on the H100 — one request, nothing else sharing the GPU — what is the chip actually doing? I assumed “a lot of matrix math.” That was wrong in a useful way.

For one conventional decode step through a dense 8B, the GPU must use essentially every dense-layer weight. Those weights live in HBM. They do not stay in on-chip cache. At batch 1 there is almost no extra arithmetic with which to amortize reading them.

Scope: embedding lookups touch selected rows. MoE models activate selected experts. Batching and speculative decoding change how much useful work one weight load produces. None of that is today.

BF16 stores each parameter in 2 bytes. An 8B-parameter dense model is about **16 GB** of weights (a little more on disk; the napkin uses \(8 \times 10^9 \times 2\)). The H100 SXM advertises 3.35 TB/s of HBM bandwidth. For one-token-at-a-time, batch-1 decode, no quant, no speculation, the idealized **weight-bandwidth ceiling** is:

> **3,350 GB/s ÷ 16 GB ≈ 210 tokens/second** — a peak-rate upper bound, not a measured speed.

How much math is that token? Rule of thumb: 2 FLOPs per parameter (one multiply and one add). About 16 GFLOPs for the 8B. Against ~989 TFLOP/s of dense BF16 on the H100, the compute-time lower bound is about **16 microseconds**. The weight-read lower bound is about **4.8 milliseconds**.

Those are not two phases of a timeline I measured. Real kernels overlap movement and math, hit neither advertised peak, and use the tensor cores poorly on batch-1 matrix-vector work. The useful comparison is the gap: **about 295:1 between the chip’s peak compute and its peak bandwidth balance**. Batch-1 dense decode does not supply enough math per byte to keep the H100 busy.

![Peak-rate lower bounds for one decode step: roughly 4,800 µs for the weight read versus 16 µs for dense arithmetic, explicitly not a measured timeline](figures/fig2-time-budget.png)

The same arithmetic on a dense 70B at BF16: ~140 GB of weights, ~42 ms to read them at peak, ~142 µs of dense math. The imbalance is still **295×**. Bytes and FLOPs both scale with parameter count, so model size cancels. What remains is a property of the chip:

**989 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOPs per byte.**

That is the H100’s ridge point — the ops:byte ratio it wants. Batch-1 dense decode supplies about **one** (two bytes of weight, two FLOPs). The FLOPs-per-byte a workload supplies is its **arithmetic intensity**. When intensity sits far below the ridge, the work is **bandwidth-bound**: it finishes when the memory system is done. Buying more FLOPs does not help. That plot is the **roofline**, which comes later.

Model size does not change the imbalance in this simplified picture. It changes capacity and the absolute ceiling. 140 GB of 70B BF16 weights do not fit on one 80 GB H100. One H100’s peak bandwidth would imply only about 24 tok/s even if they did. That is part of why large models get split across GPUs — later.

The same formula on other advertised bandwidths is just the formula. It is not the lab:

![Idealized weight-bandwidth ceilings for an 8B BF16 model using advertised peak bandwidth](figures/fig3-bandwidth-ceilings.png)

- **H100 SXM (the lab pin):** 3.35 TB/s → ~210 tok/s ceiling. [NVIDIA H100 spec](https://www.nvidia.com/en-us/data-center/h100/).
- **RTX 4090, for scale:** ~1.0 TB/s → ~63 tok/s ceiling on the same 16 GB of weights. Spec-sheet only.
- **A laptop, for scale:** Apple lists 153 GB/s (base M5) and 614 GB/s (top M5 Max). That is ~10 and ~38 tok/s *if* unified memory can hold ~16 GB of weights plus runtime and KV. A 16 GB SKU cannot. These are not measurements and this series will not use a Mac as a lab.

Prefill is the other half of the opening riddle. The prompt tokens can be processed together, so the same weights get reused across many positions and arithmetic intensity can rise. Whether a given prefill is compute-bound depends on length, batch, kernels, and the chip. Low-batch decode usually does not get that reuse. Later in the series.

A batch is the other lever: load the weights once, use them on several sequences. Throughput can rise until compute or something else becomes the limit. Later.

## Four things I had wrong about this loop

**“The decoder has a finished answer waiting.”** Standard autoregressive decoding commits one token at a time. It does not keep a sentence buffer or revise the future. Speculative decoding *guesses* ahead and checks; that is later.

**“Longer answers are slower only because the questions are harder.”** Every extra token is another sequential decode step. Per-token cost is not constant — KV traffic grows with filled context — but output length is a first-order cost. That is why long reasoning traces changed the economics.

**“Generation is slow because the GPU lacks FLOPs.”** The ridge comparison points the other way for batch-1 dense decode on an H100. Exact utilization is a measurement. I will not state one until I have run the harness.

**“Greedy decoding is bitwise reproducible.”** Greedy removes the explicit random draw. Production stacks can still change reduction order with kernels and batching. I have not reproduced that yet.

## Where this series goes

Today we ignore three things on purpose: the recompute waste (**KV cache**), serving more than one request (**batching**), and kernels / quant / parallelism. The harness on the H100 — same 8B, same image, warm-up, sync, raw log — comes later. After that, “I measured” has a pin.

The last arc is a small Rust engine on that same box, compared to this loop and to vLLM. The spike has to load the 8B, do cached decode, take a ragged batch, and print the same metrics the posts use. Until that runs, the engine is a destination, not a product announcement.

**Next:** *Prefill vs decode — why reading and writing stress the H100 differently.*

**Where these numbers came from.** H100 SXM 3.35 TB/s and ~989 TFLOP/s dense BF16 are vendor specs. The ~210 tok/s and ~295 ops:byte figures are arithmetic on those specs. Philip Kiely’s *Inference Engineering* (Baseten) works the same H100 ridge; Vizuara’s workshop guide derives intensity ≈ 1 for dense batch-1 decode. I have not yet run the 8B on an H100.

Three caveats. Advertised peaks are not achieved peaks. Kernels overlap, so 4.8 ms and 16 µs are not a stacked timeline. KV, activations, sampling, framework overhead, and later communication are missing from weights ÷ bandwidth. That is why it is a ceiling you can beat or miss later, not a promised tok/s.

**Changelog.** 2026-09-05 — Locked series model to Llama-3.1-8B-Instruct BF16 and lab GPU to cloud H100 SXM. Ceilings only; Mac/4090 left as formula examples. Naive loop now states that all positions are recomputed. Dropped week-number forward pointers; later posts stay unnamed. Dropped the weekly-digest and subscribe promises until those exist.

*Companion code: [inference-from-scratch / week-01](https://github.com/stp8954/inference-from-scratch/tree/main/week-01-life-of-a-token). If that path 404s, the listing in this repo is the source of truth until the companion is public.*
