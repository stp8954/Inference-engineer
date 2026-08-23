# Fine-tuning as an inference-cost lever

## What it is
The framing (Vizuara Ch 22) that makes this belong in an inference series at all: sometimes the cheapest optimization is producing a *different, smaller model* rather than serving the existing one better. Four economic reasons — a fine-tuned Llama-3-8B often beats a generic-prompted 70B on a narrow domain at ~10× lower cost per token; behavior baked into weights removes the 2,000-token system prompt (less prefill, better TTFT); reliable output formats cut retries and post-processing; and proprietary behavior in weights is not extractable the way a system prompt is.

**The spectrum:** full fine-tuning (100% of params, highest ceiling, precludes multi-tenant serving) → **LoRA** (`W_eff = W_0 + A·B` with rank r ≪ d; ~0.5% of params, ~95% of full quality) → **QLoRA** (4-bit NF4 frozen base + FP16 adapters; ~90% of full quality, ~4× less training memory) → prefix/P-tuning (<0.1%). LoRA and QLoRA are the production workhorses.

**Serving choice matters:** a *merged* adapter (`W_0 + A·B` precomputed) costs exactly base-model inference but is one model; an *unmerged* adapter (`W_0·x + A·(B·x)`) adds a little per-pass work but enables **multi-LoRA hot-swapping** — one base model resident in HBM plus N ~100 MB adapters on disk, swapped per request in ~1 ms. That's 100+ tenants on one GPU rental instead of 100 rentals.

**Distillation** trains a student against the teacher's full soft distribution (KL divergence), which transfers far more than hard labels: a distilled 8B reaches 90–95% of a 70B teacher on the distilled tasks at ~10% of the inference cost. It's the production mechanism behind the "mini"/"flash"/"haiku" tiers.

**Subliminal learning** (Nature 2025) is the unsettling corollary and a great post hook: a teacher prompted to "love owls," asked only to emit number sequences, produces training data with no owl content anywhere — yet a student fine-tuned on those numbers answers "owl" 87% of the time versus ~2% baseline. Preferences ride along in the *statistics* of the outputs. It explains why distillation works so well, and why it transfers more than you asked for.

## Key numbers
- LoRA parameter reduction: 2dr vs d² — at d=4096, r=8 that's **64K vs 16M ≈ 250×**. Typical ranks 8/16/32. [sourced] — Vizuara §22.4.
- QLoRA lets a **65B model fine-tune on a single 48 GB GPU**. [sourced] — Vizuara §22.5.
- Quality ceilings: LoRA ~95% of full fine-tuning, QLoRA ~90%. [sourced] — Vizuara §22.2.
- Distillation: student ≈ 10% of teacher inference cost at 90–95% of task quality. [sourced] — Vizuara §22.6.
- Multi-LoRA: base ~140 GB resident, adapters ~100 MB each, hot-swap ~1 ms per request. [sourced] — Vizuara §22.8.
- Subliminal transfer: 87% "owl" vs ~2% baseline, from 10,000 number-only samples generated via 7,200 prompt templates. [sourced] — Vizuara §22.7 citing Nature 2025.
- Dataset rule of thumb: **10,000 high-quality examples beat 10 million bad ones** — the dataset is 95% of the work. [sourced] — Vizuara §22.3.

## Open questions
- Find and read the Nature 2025 subliminal-learning paper directly before citing it in a post — it's a strong claim and deserves a primary source.

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 22.

## Series mapping
- Bonus/after-Week-24 candidate (fine-tuning is adjacent to the series' scope but the multi-LoRA serving mechanics belong in Week 22 agents/serving); subliminal learning is a strong standalone digest item.
