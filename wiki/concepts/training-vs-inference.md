# Training vs. inference

## What it is
Same transformer, opposite regimes (Vizuara Ch 4). Training: teacher forcing (targets = inputs shifted by one) lets all N positions compute in one parallel pass — matrix-matrix work, compute-bound, batches of millions of tokens. Inference decode: the model conditions on its own outputs, forcing one-token-at-a-time matrix-vector work — memory-bound. This asymmetry is the root cause of everything the runtime layer fixes. At inference only "how you use the model" can change (LoRA/distillation are training tricks with inference-time benefits).

## Key numbers
- Training step ≈ 3× an inference forward pass (forward + 2× backward); training FLOPs ≈ 6 × params × tokens, inference ≈ 2 × params × tokens generated. [sourced] — Vizuara Ch 4.
- Throughput-regime gap: training ~2–3M tok/s/GPU vs decode ~3–6K tok/s/GPU on H100 ≈ 400×. [sourced] — Vizuara Ch 4.
- Lifetime inference compute exceeds training compute by 10–100× for deployed models. [sourced] — Vizuara Ch 4; corroborate for Week 24.
- GPT-3 pretraining ≈ 5.25×10²³ FLOPs ≈ ~26 H100-GPU-years. [sourced] — Vizuara Ch 4.

## Open questions
- (none yet)

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 4.

## Series mapping
- Week 1 (why generation is sequential), Week 2 (regime asymmetry), Weeks 20/24 (inference as dominant cost).
