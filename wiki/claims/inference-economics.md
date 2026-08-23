# Inference economics claims

Source key: Vizuara = [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md)

- $/M tokens ≈ (GPU $/hr ÷ tok/s) × 278 (278 = 1e6/3600). [verified] — arithmetic. Recorded 2026-08-22.
- Cost spectrum (~300×, early-2026 list prices per Vizuara Fig 2.6): GPT-5 API ~$30/M; Claude Sonnet 4.6 ~$15/M; Gemini 3 Pro ~$10/M; self-hosted Llama-70B (8×H100) ~$2/M; Llama-8B (1×H100) ~$0.40/M; Llama-8B (RTX 4090) ~$0.10/M. [hearsay] — verify current prices before citing in a post. Recorded 2026-08-22.
- Vizz AI case study: voice tutor ~$1.00/session on Gemini Live (~$300K/mo at 10K DAU) vs ~$0.05/session self-hosted (~$15K/mo) ≈ 20× — "the difference is not the model, it is the serving stack." [sourced] — Vizuara Ch 0. Recorded 2026-08-22.
- DynaRoute case study: intent-classifier routing (1B/8B/70B, 70/20/10 mix) → $215 vs $1,500 per 1M queries ≈ 86% savings; the ~20 ms router is itself an inference system. [sourced] — Vizuara Ch 0. Recorded 2026-08-22.
- API-vs-self-host threshold: below ~1M tokens/day, APIs win on cost (Vizuara and Kiely agree directionally). [sourced] — Vizuara Ch 2; cf. Kiely §1.1. Recorded 2026-08-22.
- Blended-cost adders: ~70% utilization target → ×1.43 on naive cost; autoscale overhead 5–15%; observability 2–5%. [sourced] — Vizuara Ch 2. Recorded 2026-08-22.
- Open-vs-closed quality gap: ~20 benchmark pts (2023) → ~3 (2026); Chinese-origin models ≈ 41% of HF downloads (Spring 2026). [hearsay] — Vizuara Ch 0 citing "State of Open Source on HF"; verify. Recorded 2026-08-22.
