# speculative-decoding

## What it is
Draft-and-verify generation (draft models, Medusa/EAGLE-style self-speculation) to break the one-token-per-forward-pass bound.

Uses decode's idle compute: speculator drafts, target validates; N accepted drafts + 1 generated = N+1 tokens per forward pass. Improves TPS/ITL only — never TTFT. Gains depend on draft cost, draft length, and acceptance rate (drops deeper into the sequence; one rejection discards the rest → short high-acceptance drafts win). Higher temperature hurts acceptance. Works best at low batch sizes; production systems dynamically disable speculation when batches saturate compute. Methods: draft-target (separate ≥10×-smaller same-family model; easiest, most overhead), Medusa (2–4 grafted decoder heads; historical, inspired EAGLE), EAGLE (purpose-built <1B drafter trained on the target's hidden states from three layers; up to ~8 draft tokens, high acceptance, runs in the same module as the target — the production go-to), and n-gram/lookahead (no draft model; dictionary of observed suffixes; 10+ token drafts that dominate for code completion/revision where output mirrors input).

**The accept/reject rule (Leviathan, Kalai & Matias 2022/23, Theorem 1).** For each drafted token: if `p_target(x) ≥ p_draft(x)` accept with probability 1; otherwise accept with probability `p_target/p_draft`. On rejection, resample from the residual `max(0, p_target − p_draft)/Z`. The accepted-token distribution then equals the target distribution **exactly** — this is the rare optimization with *zero* quality cost, unlike quantization or MLA.

Why it's exact: acceptance contributes `min(p_draft, p_target)` and the rejection path contributes precisely the missing mass `max(0, p_target − p_draft)`; they sum to `p_target`. Worth walking through in Week 16 — it's the most elegant proof in the whole serving stack.

**Expected speedup** is a geometric series, because accepting token k requires accepting 1…k−1 first:

`speedup ≈ (1 + α + α² + … + α^K) / (1 + draft_cost_fraction)`

**Draft sources:** n-gram matching (no model at all — a rolling hash over context, proposing the earlier continuation; shockingly good on code/JSON/SQL), EAGLE (a tiny 1-layer head on the target's own last-layer hidden state, run autoregressively — current SOTA), Medusa (multiple parallel heads, each predicting one future position, non-autoregressive so cheaper per step but blind to each other).

## Key numbers
- Draft model rule of thumb: ≥10× smaller than target, same family. [sourced] — Kiely §5.2.1.
- Acceptance rates and resulting speedups (K=4, draft cost 5%): n-gram α≈0.30 → **1.4×**; Medusa α≈0.55 → **2.0×**; EAGLE α≈0.75 → **2.9×**. [sourced] — Vizuara §15.5.
- **Rule of thumb: every +10 percentage points of acceptance ≈ +30% speedup** — acceptance rate is *the* metric to optimize, so invest in the drafter. [sourced] — Vizuara §15.5.
- Acceptance by drafter type: n-gram 30% conversational (much higher on structured output), Medusa 50–70%, EAGLE 60–80%. [sourced] — Vizuara §15.4.
- Workload sensitivity: long predictable output (code/JSON/SQL) → α>70%, 2.5–4×; short chat → 1.5–2×; high-temperature creative → 1.0–1.1×; single-token classification → no win. **The sharper the target's next-token distribution, the bigger the win.** [sourced] — Vizuara §15.7.
- Roofline: verification makes one weight-load produce K token-checks, so arithmetic intensity rises ~K×. **Batching amortizes across users, speculation across tokens — they multiply:** batch 32 × 3-token chains = 96 tokens per forward pass. [sourced] — Vizuara §15.8.
- The asymmetry that motivates it: Llama-3-7B decode on H100 = ~4 ms moving 14 GB of weights vs ~14 µs of arithmetic — the GPU is ~99.7% idle. [sourced] — Vizuara §15.1.
- EAGLE: <1B params, up to 8 draft tokens (~2× Medusa's effective depth). [sourced] — Kiely §5.2.3.
- N-gram speculation: 10+ token drafts viable when output resembles input (code editing). [sourced] — Kiely §5.2.4.

## Open questions
- Week 16 implementation: basic speculative sampling from scratch + measure acceptance-rate vs temperature with KVScope.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — §5.2.
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 15 (accept/reject proof, worked trace, acceptance-vs-speedup table, drafter comparison).

## Series mapping
- Week 16
