# attention-kernels

## What it is
FlashAttention lineage, FlashInfer, decode-specific kernels; tiling and online softmax.

Two optimization strategies (Kiely §2.5): implementation improvements (lossless, still O(N²) — FlashAttention eliminates excess HBM reads/writes of the intermediate S/P matrices via hand-fused per-GPU kernels; a handful of algorithm lines becomes tens of thousands of kernel lines, different code per GPU generation) vs. new algorithms that change complexity, usually with quality tradeoffs: sliding-window (O(Nw), w≈8–32K), gated, linear, compressed, and multi-latent attention — plus attention-free Mamba/SSM blocks in hybrids (e.g., NVIDIA Nemotron 3 Nano). Training with the same approximation preserves quality at inference.

Kernel generations are architecture-specific: FlashAttention 2 is the common default; FA3 targets Hopper, FA4 targets Blackwell; TensorRT-LLM's XQA kernel accelerates attention for encoder-ish workloads (embeddings). GEMM kernel families to benchmark per-model: CuTe, CUTLASS, DeepGEMM. SageAttention = 8-bit attention kernel (blockwise scaling + selective precision) — attention quantization matters most for compute-bound diffusion workloads where attention is 70–80% of compute.

**How FlashAttention actually works (Vizuara Ch 10).** Standard attention runs three separate kernels, so the N×N score and weight matrices pass through HBM twice — at N=4096 that's 32 MB per layer per pass, ~1 GB across 32 layers, for a kernel with very few FLOPs. FlashAttention never materializes them: Q is tiled into row-blocks (typically B_r=128), K/V into column-blocks (B_c=64) sized to fit SRAM, and each tile is loaded, consumed, and discarded on-chip.

The enabling trick is **online softmax**, which is exact, not an approximation. Each query row carries three running values — max `m`, sum `l`, partial output `o` — and merging a new tile rescales to a common reference: `m_new = max(m, m')`, `l_new = exp(m−m_new)·l + exp(m'−m_new)·l'`, same for `o`; final output is `o/l`. The rescaling factors cancel in the ratio, so results are bit-identical to standard attention.

**Version lineage:** FA-1 (2022) introduced tiling + online softmax with the outer loop over K/V (~40% of H100 peak). FA-2 (2023) swapped the loop order to put Q outside, which parallelizes across warps (~65%, ~2× faster). FA-3 (2024) is Hopper-specific — TMA async copies overlapping compute, producer/consumer warp specialization, FP8 matmul with FP16 accumulation (~85% of peak, 1.3× over FA-2). The algorithm is frozen; each release is pure hardware exploitation.

## Key numbers
- Video/diffusion: attention ≈ 70–80% of compute time; attention-result caching gives 30–40% speedups (quality varies). [sourced] — Kiely §6.6.
- Attention HBM traffic: standard ≈ 8N² + 8N·d bytes (the 8N² dominates); FlashAttention ≈ N·d + N²d²/M (M = SRAM size). Crossover ≈ N=1000 on H100; **~10× less attention traffic at N=32K**. FLOPs are identical — the speedup is purely bytes not moved. [sourced] — Vizuara §10.6.
- End-to-end gains are much smaller than kernel gains, because projections and FFN dominate decode: 7–13B @ 4K, attention is 15–25% of decode → **5–10% e2e**; 70B+ @ 32K+, attention is 30–50% → **15–25% e2e**. Prefill and training gains are far larger. [sourced] — Vizuara §10.8. **Good honesty beat for Week 12 — most posts quote the kernel number.**
- FA utilization by version on H100: FA-1 40%, FA-2 65%, FA-3 85% of tensor-core peak. Not using FlashAttention at all leaves ~2–3× on the table. [sourced] — Vizuara §10.7.
- At N=131,072 a single head's score matrix is 32 GB, written and read twice per pass — across 64 heads × 80 layers, FlashAttention avoids several TB of HBM traffic per forward pass. [sourced] — Vizuara §10.5.1.
- Kernel efficiency ladder: naive PyTorch matmul ≈ 30–50% of rated peak; hand-tuned cuBLAS ≈ 85–92% (realistic FP16 ceiling ~800 TFLOPs on H100's rated 989). Kernel-launch overhead (100s of µs) dominates at batch 1 → CUDA graphs amortize it; H100 TMA overlaps loads with compute (FA3 exploits it). [sourced] — Vizuara Ch 3 §3.12.

Kernel ecosystem (Kiely §4.1): cuBLAS/cuDNN are the stock primitives (GEMM everywhere); CUTLASS and CuTe are template libraries for authoring fast kernels (FA3 is built on CUTLASS); FlashInfer packages LLM-specific kernels (attention + fused sampling); DeepGEMM (DeepSeek) is a plugin-able FP8 GEMM. Kernels are generation-specific — an H100 kernel underuses a B200 — so selection (mostly automatic, occasionally manual plugins for hot paths) is the practical lever; writing kernels is not. Fusion kills memory round-trips between back-to-back kernels (matmul+bias+activation is the classic), which matters precisely because decode is bandwidth-bound; torch.compile auto-fuses simple chains but can't fuse handwritten plugin kernels.

Why tiling works (Vizuara Ch 6): a 64×64 FP16 tile is 8 KB, so it fits in an SM's 228 KB SRAM — FlashAttention keeps the N×N score matrix off HBM entirely, and FA3 reaches ~85% of H100 peak FP16 (the missing 15% is warp stalls, SRAM bank conflicts, compute/memory imbalance). One block per Q tile, grid = N/64 blocks, each tensor-core op one warp on a 16×16×16 MMA.

**What writing the decode kernel by hand actually costs (Chan, YALM 2026).** The literature above
describes finished kernels; this is the only source we have on the path to one, with a measurement at
each step. Three findings transfer directly to the Rust arc:

1. **Parallelization granularity dominates everything else.** The naive GPU port ran at 2.9 tok/s —
   *slower than the optimized CPU version* — because it launched one block per output element. Giving
   each warp a strided slice of the matmul instead took it to 51.7 tok/s in one change. Every
   subsequent optimization combined was worth about 23%.
2. **Coalescing writes, not just reads.** The original attention kernel partitioned poorly and lost
   memory throughput on scattered writes. The redesign splits the sequence into chunks across multiple
   blocks so that contiguous elements of the value matrix load together, and accumulates with
   `atomicAdd` into *shared* memory rather than global — which also avoids subnormal-value loss.
   Worth 56.1 → 63.7 tok/s.
3. **Half-precision loads can silently disable compiler loop unrolling.** Switching to an FP16 KV cache
   should have been a straight bandwidth win but was not: the compiler unrolled far less aggressively
   over half-precision loads, costing ~75 µs. The fix was manual — unroll in 16-iteration batches,
   prefetch into registers, and extract with a switch. A reminder that a "pure bandwidth" change can be
   undone by codegen, and that you must measure rather than reason about it.

## Open questions
- FlashAttention paper (Dao et al.) primary-source ingest for Week 12.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §2.5.
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 6 (SRAM tiling budget), Ch 3 §3.12 (kernel efficiency ladder).
- [Chan, *YALM* (2026)](../../sources/2026-08-24-chan-yalm.md) — hand-written decode kernels on an RTX 4090; measured step-by-step. See [claims/technique-performance.md](../claims/technique-performance.md) for the full ladder.

## Series mapping
- Week 12
