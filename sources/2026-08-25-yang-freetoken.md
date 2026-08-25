# FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution

- **Type:** Paper + open-source system. arXiv:2608.16157. Code: https://github.com/FlashML-org/FreeToken. Project: flashml.ai
- **Authors:** Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu, Ion Stoica
- **Ingested:** 2026-08-25
- **Coverage:** COMPLETE for the paper (v1, 17 Aug 2026) — §1–7 including the §5 evaluation — plus the repository README. See the appended update of 2026-08-25 below.

## Summary

An edge-native MoE serving system from the Berkeley/MIT orbit that produced vLLM and PagedAttention. The framing is the contribution: a personal machine is treated "not as a small GPU, but as a unified, elastic inference platform" — GPU, CPU, host memory and the interconnects between them are one resource pool rather than a fast tier plus a fallback. Because agent workloads continuously change their execution pattern and edge hardware has machine-specific resource balance, FreeToken rejects fixed offloading policies and instead maps computation and model state to resources dynamically at runtime. The serving stack is co-designed end to end: model layout, expert residency, CPU–GPU co-execution, agent state reuse, and memory management. Reported reach is 20+ MoE models on hardware from 8 GB laptop GPUs up to workstation GPUs, spanning 35B-parameter models on laptops to 753B GLM-5.2 on a single workstation GPU.

Named mechanisms: bandwidth-adaptive CPU–GPU co-execution; double-buffered full-layer prefill streaming; global LRU expert caching; semantic anchor checkpoints that let agentic context edits (tool calls, thinking blocks) reuse state instead of re-prefilling; dynamic VRAM reallocation between the expert cache and KV memory without an engine restart. Quantization support spans MXFP4, NVFP4, FP8 and BF16.

## Why it matters to the series

It is the counterexample that generalizes Week 1's formula rather than breaking it. `ceiling = bandwidth ÷ bytes read per token` still holds; what changes off-GPU is which bandwidth and which bytes. The bandwidth term becomes the slowest tier a token must cross (host memory over PCIe) instead of HBM, and for an MoE the bytes term is only the *active, non-resident* experts rather than the whole checkpoint. Every mechanism in the paper is an attack on one of those two terms — expert caching shrinks the bytes, co-execution and streaming hide the crossing. That makes it a strong Week 1 sidebar and the anchor for Week 14.

## Wiki pages touched

- entities/freetoken.md (new)
- concepts/moe-inference.md (edge-serving regime, capacity claims, batch-1 sparsity payoff)
- concepts/hardware-landscape.md (consumer GPU as a real serving tier)
- concepts/prefix-caching.md (semantic anchor checkpoints for agentic context edits)
- claims/decode-bandwidth-ceilings.md (the offload regime of the same formula)

## Appended 2026-08-25 — full paper read, evaluation recorded

The three gaps flagged on first pass are closed.

**Throughput now available and recorded as `[sourced]`.** RTX 5090: 77–83 tok/s on Qwen3.6-35B-A3B
(BF16), 22–25 tok/s on DeepSeek-V4-Flash (MXFP4) — 1.8–2.3× and 1.5–1.9× the strongest baseline.
Decode stays within 12% of single-turn across three agent workloads while the most context-sensitive
baseline loses 31% by the second. Worst-case TTFT below 44 s everywhere; baselines reach 232 s
(llama.cpp), 179 s (Ollama), 946 s (KTransformers). 8 GB RTX 4060 laptop serves 35B at 39.3 tok/s.
Single RTX PRO 6000 serves GLM-5.2 (753B) at 14.9 tok/s vs llama.cpp's 7.3. Ablations: prefill
double-buffering worth 19/25/26% at 4K/8K/16K; expert-cache miss rates 16%/39% for global LRU against
41%/59% and 62%/89% for the two baseline placement policies at equal capacity.

**The PCIe gap is closed better than expected.** Table 1 gives *measured* transfer and host-kernel
bandwidths on six machines rather than platform specifications — now recorded in
`claims/hardware-specs.md`. The `q⋆ ≈ m · B_P / B_H` derivation (§3.2, eqs. 1–4) makes those two
numbers the governing parameters of the whole offload regime.

**284B confirmed** — the paper uses 284B for DeepSeek-V4-Flash (13B active) throughout; the README's
"290B+" is loose rounding. Use 284B.

**Also recorded:** the MoE prefill/decode sparsity inversion (§2.1–2.2) — decode is genuinely sparse,
but the union of routes across a long prompt turns the prefill working set effectively dense. Filed to
`concepts/prefill-decode.md` as Week 2 material.

## Remaining caveats

- Agent trajectories diverge across engines, so the paper deliberately does **not** compare cross-engine wall-clock totals — only per-request decode throughput and mean TTFT. Do not quote an overall "×faster" figure; it does not exist in the paper.
- Three of the six systems are rented dual-socket servers capped at 6 CPU threads and NUMA-pinned to emulate edge hosts. The authors validate the emulation against two real edge machines (bandwidths line up at 47.5–53.8 GB/s), but name the emulation if citing those rows.
- Baselines are single-version snapshots of fast-moving projects. Fine for a 2026 post; date the comparison.
