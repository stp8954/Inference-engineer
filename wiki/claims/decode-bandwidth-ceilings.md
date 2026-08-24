# Decode bandwidth ceilings (batch = 1)

Idealized batch-1 weight-streaming ceiling: tok/s ≈ sustained memory bandwidth (GB/s) ÷ weight
bytes (GB). This isolates the cost of reading weights once per decode step; it is not a performance
prediction when KV traffic, runtime overhead, communication, or unused bandwidth are material.

- 8B model @ BF16 (~16 GB of weights) on H100 SXM (~3.35 TB/s peak bandwidth) → ~210 tok/s idealized weight-bandwidth ceiling. [sourced] — vendor bandwidth spec plus the formula above; needs a pinned checkpoint and [verified] KVScope run before being presented as measured performance. Recorded 2026-08-22; wording clarified 2026-08-24.
- 8B @ BF16 on current MacBook Pro configurations: base M5 at 153 GB/s → ~9.6 tok/s; highest-bandwidth M5 Max at 614 GB/s → ~38.4 tok/s idealized weight-bandwidth ceilings, provided the selected unified-memory capacity holds the checkpoint plus runtime/KV state. [sourced] — Apple MacBook Pro technical specifications (see ../../sources/2026-08-24-apple-macbook-pro-specs.md); these are not measured inference results. Recorded 2026-08-24.
- Per-token dense linear-layer work ≈ 2 FLOPs/parameter → ~16 GFLOP for 8B. Against 989 TFLOP/s peak, that is an idealized ~16 μs compute-time floor; at the separate ~210 tok/s weight-bandwidth ceiling, it would demand ~3.4 TFLOP/s, or ~0.34% of peak. [sourced] — standard matrix-multiply estimate plus the H100 vendor peak specification; derivation in the Week 1 draft. Recorded 2026-08-22; dimensional comparison corrected 2026-08-24.
- H100 ops:byte ratio ≈ 295 (989 TFLOPS dense FP16 ÷ 3.35 TB/s): any op with arithmetic intensity below ~295 is memory-bound on H100; standard attention at decode (d=128, N=4096, FP16) has AI ≈ 62. [sourced] — Kiely, Inference Engineering (2026) §2.4 (see ../../sources/2026-08-22-kiely-inference-engineering.md). Recorded 2026-08-22.
- 7B @ FP16 (14 GB) on H100: 14/3350 ≈ 4.18 ms/token ITL floor ≈ 239 tok/s ceiling — independent source, same formula as the 8B entry. [sourced] — Vizuara Ch 2. Recorded 2026-08-22.
- Batch-1 ITL floors on H100 for a 7B model by precision: FP16 (14 GB) 4.18 ms; FP8 2.09 ms; INT4 1.05 ms. Compute utilization at batch 1 ≈ 0.003% (28 GFLOPs needed vs 989 TFLOPS available). [sourced] — Vizuara §6.6. Recorded 2026-08-22.
- Continuous batching at 32 users amortizes the same weight load to ~0.13 ms/token effective. [sourced] — Vizuara §6.6. Recorded 2026-08-22.
- Long-context reality check, Llama-3-70B @32K on H100 (2-way TP): 43 GB KV re-read (12.8 ms) + weights (~21 ms) ≈ 34 ms single-user ITL — KV traffic dominates past N≈4K. [sourced] — Vizuara §7.7. Recorded 2026-08-22.
