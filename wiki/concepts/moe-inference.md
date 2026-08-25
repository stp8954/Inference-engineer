# moe-inference

## What it is
Serving mixture-of-experts models - expert parallelism, all-to-all, load balancing.

**The edge regime (FreeToken, 2026).** The batch-1 half of that asymmetry is not a curiosity — it is
a whole deployment category. On a personal machine there is exactly one user, so the sparsity holds
completely: only the routed experts are needed per token, and a 753B model does not require 753B
parameters of bandwidth per step. That is what makes frontier MoE models runnable on a gaming PC at
all, and it inverts the datacenter intuition, where cross-request expert activation makes nearly all
parameters hot.

What the edge regime changes is *which* bandwidth binds. The weights do not fit in VRAM, so a routed
expert that is not resident must cross PCIe from host memory — two orders of magnitude slower than
HBM. The Week 1 formula survives intact; you substitute a different bandwidth and a much smaller byte
count, because the bytes are only the *active, non-resident* experts rather than the whole checkpoint.
Every mechanism in FreeToken attacks one of those two terms: global LRU expert caching raises the
share of routed experts already in VRAM (shrinking the bytes), while double-buffered prefill streaming
and bandwidth-adaptive CPU–GPU co-execution hide the crossing behind compute (raising the effective
bandwidth). Expert-cache hit rate, not FLOPs, is the design variable. FreeToken also reallocates VRAM
between the expert cache and KV memory dynamically — the two consumers trade off directly, and the
right split depends on context length and routing locality rather than being fixed at startup.

## Key numbers
- Qwen3-235B-A22B: 22B of 235B params active per token; router picks 8 of 128 experts at each of 94 layers, per token (Kiely §2.2.4). [sourced]
- Key serving asymmetry: MoE sparsity is real at batch 1 (local inference), but in batched serving different requests activate different experts — expect nearly all params hot unless doing large-scale expert parallelism. [sourced]
- Edge MoE reach: 20+ MoE models served across hardware from 8 GB laptop GPUs to workstation GPUs; 35B-parameter models on laptops, 284B on a gaming desktop, 753B (GLM-5.2) on a single workstation GPU. [sourced] — FreeToken abstract. Recorded 2026-08-25.
- Sparsity, concretely: DeepSeek-V4-Flash activates **6 of 256 routed experts in each of 43 layers**, so only **13B of 284B** parameters participate in any single token. GLM-5.2 is 753B with 40B active (a 433 GB NVFP4 checkpoint). [sourced] — FreeToken §1, §5.1. Recorded 2026-08-25.
- Measured decode on an RTX 5090: **77–83 tok/s** on Qwen3.6-35B-A3B (BF16), **22–25 tok/s** on DeepSeek-V4-Flash (MXFP4) — 1.8–2.3× and 1.5–1.9× the strongest baseline. An 8 GB RTX 4060 laptop on PCIe ×8 serves a 35B model at **39.3 tok/s**. [sourced] — FreeToken §5.2, §5.3. Recorded 2026-08-25. Full table in [claims/technique-performance.md](../claims/technique-performance.md).
- **Expert-cache locality is the whole ballgame.** At the RTX 5090's serving capacity (37% of Qwen3.6's expert pool, 11% of DSV4-Flash's), decode-time expert miss rates are: global LRU **16% / 39%**, prefill-updated placement 41% / 59%, routing-blind static split 62% / 89%. Adjacent tokens route to overlapping experts, and a cache that simply follows the router beats any placement fixed ahead of time. [sourced] — FreeToken §5.3. Recorded 2026-08-25.

Expert Parallelism (Kiely §5.4.2): whole experts sharded per GPU (128 experts at EP8 → 16/GPU); tiny router replicated on every GPU; communication is token-passing between experts, not per-layer all-reduce, so EP scales across nodes on slower interconnects. EP buys throughput, not per-token latency; common to mix TP (attention) + EP (MoE layers). Multi-node MoE: EP16 for throughput vs TP8PP2 for per-user latency.

**The `q⋆` policy — the idea worth stealing.** Once a routed expert is missing from VRAM, every prior
system treats it as *data to move*: fetch it over PCIe. That caps single-stream decode at the PCIe rate
no matter how good the prefetch predictor gets, while the host CPU sits idle. FreeToken's observation is
that a missing expert is also *work that can execute where it already lives*. Since DMA transfer and CPU
execution both draw on host memory, a saturated PCIe link leaves residual bandwidth `B_H − B_P`;
balancing the two concurrent branches gives a closed form for how many of the `m` misses to fill versus
compute in place:

> `q⋆ ≈ m · B_P / B_H`

with `B_P` (PCIe transfer) and `B_H` (CPU expert-kernel) profiled on the actual machine at deployment.
It is cheap enough to stay device-resident inside a captured CUDA graph, which is why it survives
contact with a real runtime where per-layer host-side heuristics do not. As `B_H → B_P` it degenerates
gracefully to pure cache fill. Measured `B_H : B_P` ratios on real hardware run from ~1.1× to ~4×
(see [claims/hardware-specs.md](../claims/hardware-specs.md)) — that ratio is the whole design space.

## Open questions
- DeepSeek-V3 / expert load-balancing papers for Week 14 primary sources.
- How much of FreeToken's win is MoE sparsity versus the caching and scheduling machinery? The paper ablates prefill overlap and cache policy separately but does not isolate sparsity itself; a dense model of equal footprint would.
- The `q⋆` derivation assumes both branches are bandwidth-bound and that CPU execution scales with residual host bandwidth. Worth checking against our own numbers in Week 14 — it is a clean enough result to be worth re-deriving on the page rather than citing.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §2.2.4.
- [Yang et al., *FreeToken* (2026)](../../sources/2026-08-25-yang-freetoken.md) — edge-native MoE serving, expert residency, bandwidth-adaptive execution.

## Series mapping
- Week 14 (primary); Week 1 sidebar (the formula in the offload regime); Week 23 (consumer GPUs as a serving tier).
