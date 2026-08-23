# Hardware specs and rules of thumb

Source key: Kiely = [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md)

- H100 SXM: 989 teraFLOPS dense FP16 Tensor Core compute; 3.35 TB/s HBM bandwidth; ops:byte ≈ 295. [sourced] — Kiely §2.4.1, matches NVIDIA spec sheet. Recorded 2026-08-22.
- H100 cache hierarchy: 256 KB L1 per SM; 50 MB L2 total. [sourced] — Kiely §3.1.2. Recorded 2026-08-22.
- Tensor Core FLOPS ≈ double per precision halving (FP16→FP8→FP4); "sparse" spec numbers assume 2:4 structured sparsity (50% zeros) and are ~2× dense — inference is dense by default, compare dense numbers. [sourced] — Kiely §3.1.1. Recorded 2026-08-22.
- VRAM sizing rule of thumb: weights + ≥50% headroom for KV cache; more for long context, large batches, or video generation. [sourced] — Kiely §3.1.2. Recorded 2026-08-22.
- Bottleneck map: LLM prefill compute-bound; LLM decode memory-bound (low-to-medium batch); image/video generation compute-bound. [sourced] — Kiely §2.4. Recorded 2026-08-22.
- FP8 sizing rule: ~1 GB VRAM per 1B params (weights only); KV cache often consumes 80%+ of post-weights VRAM in production. [sourced] — Kiely §5.4. Recorded 2026-08-22.
- B200 = 180 GB VRAM; instance ladder 180/360/720/1,440 GB (1–8×B200). [sourced] — Kiely §5.3.2, §5.4. Recorded 2026-08-22.
- GPU spec table (FP8 dense TFLOPS / VRAM / bandwidth): H100 1,979/80GB/3.35TB·s; H200 1,979/141GB/4.8TB·s; B200 ~5,000/192GB/8TB·s; B300 ~5,000/288GB/8TB·s; L4 242/24GB/300GB·s; L40 362/48GB/864GB·s. [sourced] — Kiely §3.2. Recorded 2026-08-22.
- Interconnects: NVLink 900 GB/s (Hopper) / 1,800 GB/s (Blackwell) per GPU pair; NVSwitch all-to-all intra-node; InfiniBand ≤400 Gb/s per NIC inter-node; Ethernet ~100 Gb/s; Grace NVLink C2C 900 GB/s CPU↔GPU. [sourced] — Kiely §3.3.1, §3.2.5. Recorded 2026-08-22.
- MIG: A100/H100/H200/B200 partition into ≤7 slices (H100: 8 memory / 7 compute slices; 132 SMs on SXM). [sourced] — Kiely §3.3.2. Recorded 2026-08-22.
- NVIDIA architecture cadence: Volta '17, Turing '18, Ampere '20, Lovelace '22, Hopper '22, Blackwell '24, Rubin '26 (HBM4 + CPX prefill chip), Feynman '28. [sourced] — Kiely §3.2. Recorded 2026-08-22.
- Local inference tradeoff: RTX 5090 = 32 GB @ 1,792 GB/s (~$5K system) vs Apple M3 Ultra = 512 GB @ 819 GB/s (~$10K) — bandwidth vs capacity. [sourced] — Kiely §3.5.1. Recorded 2026-08-22.
- SXM vs PCIe form factor: ~5% higher memory bandwidth on SXM (A100 example); SXM dominates inference. [sourced] — Kiely §3.3. Recorded 2026-08-22.
- H100 compute ceilings by precision (dense): FP32 67 / FP16 989 / FP8 1,979 / INT4 3,958 TFLOPS; SRAM 228 KB/SM. [sourced] — Vizuara Ch 3 (FP16/FP8 match Kiely §2.4.1). Recorded 2026-08-22.
- NVLink ~900 GB/s vs InfiniBand ~50 GB/s ≈ 18:1 — the ratio dictating TP-intra-node / PP-or-EP-inter-node. [sourced] — Vizuara Ch 1 (consistent with Kiely §3.3.1). Recorded 2026-08-22.
- H100 internals: 132 SMs × (128 FP32 CUDA cores + 4 tensor cores) ≈ 18,432 arithmetic lanes, ~110,000 threads; 528 tensor cores total (~2.5 TFLOPS FP16 each); 65,536 registers + 228 KB SRAM per SM; HBM latency ~600 cycles. B200 = 208 SMs, 8 TB/s, 1,800 GB/s NVLink, same 228 KB SRAM/SM. [sourced] — Vizuara Ch 6. Recorded 2026-08-22.
- Memory hierarchy bandwidths: registers (fastest) → SRAM ~20 TB/s → L2 ~5 TB/s (50 MB on H100) → HBM 3.35 TB/s (80 GB). ~10× bandwidth per tier, inversely ~10× capacity. [sourced] — Vizuara Ch 6. Recorded 2026-08-22.
- CPU vs GPU on a 4096² FP16 matmul: ~1 TFLOPS vs ~989 TFLOPS ≈ 1000×; 7B via llama.cpp on CPU ≈ 5 tok/s vs 200–500 tok/s per user on GPU. Transformers are ~95% matmul. [sourced] — Vizuara §6.2. Recorded 2026-08-22.
- FA3 reaches ~85% of H100 peak FP16; naive PyTorch ~30–50%; hand-tuned cuBLAS 85–92%. [sourced] — Vizuara Ch 3/Ch 6. Recorded 2026-08-22.
- Ridge point holds across generations: B200 raises bandwidth 2.4× and compute ~2.3× together, so the memory-bound regime is not fixed by new silicon. [sourced] — Vizuara §6.10. Recorded 2026-08-22.
