# Decode bandwidth ceilings (batch = 1)

Formula: ceiling tok/s ≈ memory bandwidth (GB/s) ÷ weight bytes (GB). Every parameter streams from HBM once per generated token.

- 8B model @ BF16 (~16 GB) on H100 SXM (~3.35 TB/s) → ~210 tok/s ceiling. [sourced] — vendor bandwidth spec; needs [verified] via KVScope run before Week 1 ships. Recorded 2026-08-22.
- 8B @ BF16 on Apple M-series (~150–400 GB/s) → ~9–25 tok/s. [hearsay] — consistent with community llama.cpp reports; verify on own MacBook. Recorded 2026-08-22.
- Per-token decode compute ≈ 2 FLOPs/param → ~16 GFLOP for 8B ≈ 1–2% of H100 BF16 peak. [sourced] — standard estimate; derivation in Week 1 draft. Recorded 2026-08-22.
- H100 ops:byte ratio ≈ 295 (989 TFLOPS dense FP16 ÷ 3.35 TB/s): any op with arithmetic intensity below ~295 is memory-bound on H100; standard attention at decode (d=128, N=4096, FP16) has AI ≈ 62. [sourced] — Kiely, Inference Engineering (2026) §2.4 (see ../../sources/2026-08-22-kiely-inference-engineering.md). Recorded 2026-08-22.
- 7B @ FP16 (14 GB) on H100: 14/3350 ≈ 4.18 ms/token ITL floor ≈ 239 tok/s ceiling — independent source, same formula as the 8B entry. [sourced] — Vizuara Ch 2. Recorded 2026-08-22.
- Batch-1 ITL floors on H100 for a 7B model by precision: FP16 (14 GB) 4.18 ms; FP8 2.09 ms; INT4 1.05 ms. Compute utilization at batch 1 ≈ 0.003% (28 GFLOPs needed vs 989 TFLOPS available). [sourced] — Vizuara §6.6. Recorded 2026-08-22.
- Continuous batching at 32 users amortizes the same weight load to ~0.13 ms/token effective. [sourced] — Vizuara §6.6. Recorded 2026-08-22.
- Long-context reality check, Llama-3-70B @32K on H100 (2-way TP): 43 GB KV re-read (12.8 ms) + weights (~21 ms) ≈ 34 ms single-user ITL — KV traffic dominates past N≈4K. [sourced] — Vizuara §7.7. Recorded 2026-08-22.
