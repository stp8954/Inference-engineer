# FreeToken

## What it is

An edge-native MoE serving engine (FlashML, 2026) for running frontier-scale open-weight MoE models
on personal hardware — laptops, gaming desktops, single-GPU workstations — rather than datacenter
nodes. Python packaging with performance-critical kernels beneath; ships as a desktop application for
Windows and Linux and via `uv`/pip, and exposes OpenAI- and Anthropic-compatible APIs so local agents
can point at it unchanged.

Its thesis is that a personal machine is not a small GPU but a heterogeneous pool — GPU, CPU, host
RAM, PCIe — whose balance differs machine to machine, and that agent workloads shift their execution
pattern continuously. Both facts defeat a fixed offloading policy, so FreeToken maps computation and
model state to resources dynamically at runtime instead. The stack is co-designed around that:
model layout, expert residency, CPU–GPU co-execution, agent state reuse, memory management.

Mechanisms worth knowing by name: **bandwidth-adaptive CPU–GPU co-execution**, **double-buffered
full-layer prefill streaming**, **global LRU expert caching**, **semantic anchor checkpoints** (so
agentic context edits — tool calls, thinking blocks — reuse state instead of re-prefilling), and
**dynamic VRAM reallocation** between the expert cache and KV memory without restarting the engine.
Quantization support spans MXFP4, NVFP4, FP8 and BF16.

The author list matters for placing it: Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang,
Shanlin Sun, Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu, Ion Stoica. This is the same
research orbit that produced vLLM and PagedAttention, plus Song Han's quantization lineage — which
makes it a signal about where that group thinks serving goes next, not just another local-inference
project.

## Timeline

- 2026-08-17 — Paper (arXiv:2608.16157v1, cs.DC) and public release at flashml.ai; repository at
  `FlashML-org/FreeToken`. Reach: 20+ MoE models, 8 GB laptop GPUs through workstation GPUs, 35B on
  laptops to 753B GLM-5.2 on a single workstation GPU. Evaluated on Qwen3.6-35B-A3B (BF16),
  DeepSeek-V4-Flash (284B/13B active, MXFP4) and GLM-5.2 (753B/40B active, NVFP4), across six machines
  and four real agentic workloads, against llama.cpp, Ollama, KTransformers and MoE-Infinity.

## Headline results

RTX 5090 unless noted; full numbers in [claims/technique-performance.md](../claims/technique-performance.md).

- Decode **77–83 tok/s** (Qwen3.6-35B) and **22–25 tok/s** (DSV4-Flash) — 1.8–2.3× and 1.5–1.9× the best baseline.
- Decode rate stays within **12%** of single-turn across agent workloads; the most context-sensitive baseline loses 31% by the second workload.
- Worst-case TTFT under **44 s** everywhere; baselines hit 232 s / 179 s / 946 s. Tail TTFT is an availability boundary — clients time out before users do.
- 8 GB RTX 4060 laptop serves 35B at **39.3 tok/s**, 92% of the RTX 4090 rate and above Codex's 33 tok/s production median.
- Single RTX PRO 6000 serves **GLM-5.2 (753B) at 14.9 tok/s vs llama.cpp's 7.3**, bit-identical weights.

## The two ideas

1. **`q⋆ ≈ m · B_P / B_H`** — a missing expert is not only data to move but work that can execute where
   it lives. Prior systems serve every miss as a PCIe transfer and are therefore capped at the link
   rate while host compute idles. Splitting misses between transfer and in-place CPU execution by the
   ratio of two *measured* bandwidths lifts the ceiling from `B_P` toward `B_H`. Cheap enough to live
   inside a captured CUDA graph.
2. **Semantic anchor checkpoints** — agent harnesses edit context at special-token boundaries (thinking
   blocks, tool calls, tool outputs, turns), and those are exactly where a surviving prefix ends.
   Anchoring recurrent-state checkpoints there means an edit re-prefills only the genuinely new suffix.
   Named harness behaviors: OpenClaw strips thinking blocks from all but the latest assistant turn,
   OpenCode replaces old tool outputs with a placeholder, SWE-agent elides all but the last n
   observations.

## Relation to the series

- **Week 1** — the sidebar that generalizes the napkin formula. Off-GPU, `bandwidth ÷ bytes` still
  governs, but the bandwidth is PCIe rather than HBM and the bytes are only the active non-resident
  experts. Good place to show the mental model surviving a change of regime.
- **Week 14 (MoE)** — primary system reference for edge MoE serving and expert-residency policy.
- **Week 23 (hardware)** — evidence that consumer GPUs are a serving tier, not just a dev box.
- **Weeks 25–30 (Rust engine)** — *not* an architectural template. Python-plus-kernels, and the hard
  parts are offload scheduling, which is out of scope for the series' engine. Read it for the
  problem framing, not the code layout.

## Sources

- [Yang et al., *FreeToken* (2026)](../../sources/2026-08-25-yang-freetoken.md) — full paper including §5 evaluation, plus repository README.

## Open questions

- How much of the win is MoE sparsity versus the caching and scheduling machinery? §5.3 ablates prefill
  overlap and cache policy but does not isolate sparsity itself.
- Agent trajectories diverge across engines, so the paper deliberately does not compare cross-engine
  wall-clock totals — only per-request decode throughput and TTFT. Worth remembering before quoting any
  "×faster overall" figure that does not exist in the paper.
- The three rented server systems are dual-socket boxes capped at 6 CPU threads to emulate edge hosts.
  The authors validate the emulation against two real edge machines and the bandwidths line up, but the
  emulation is worth naming if we cite those rows.
