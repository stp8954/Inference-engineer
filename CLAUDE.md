# Working on this repo

Read this first. It is the handoff note for any new session.

## What this is

The knowledge base and writing home for a blog series on LLM inference, written by Sanket
(@stp8954). Two tracks:

- **Deep dives** — a 30-week curriculum, beginner → state of the art, ending by building a
  vLLM-style inference engine in Rust. Full plan: `planning/series-plan.md`.
- **This Week in Inference** — a weekly news digest. Format and source list are in the plan.

Related repos: **KVScope** (`stp8954/KVScope`) is the companion profiler and the series' measurement
tool. `inference-from-scratch` will hold the per-week code and the Rust engine — **not yet created**.

## Voice — read this before drafting anything

**Learning in public, receipts over résumé.** Sanket is not an inference veteran and the series never
pretends otherwise. Authority comes from derivations, runnable code, and measurements the reader can
reproduce — not from claimed experience. Concretely: first person with honest provenance ("I
measured", "this surprised me", "I had this wrong"), confidence calibrated to evidence (state
verified math plainly; hedge only what is genuinely untested), surprise used as the narrative
engine, a visible corrections changelog, and questions posed before answers. Full guidance in
`planning/series-plan.md` under **Voice**.

This matters mechanically: **never state a number plainly unless it traces to a `[verified]` or
`[sourced]` entry in `wiki/claims/`.** Hedge `[hearsay]`.

## Current state (as of 2026-08-22)

**Done**
- 30-week curriculum planned, including the Phase 6 Rust-engine arc (Weeks 25–30)
- Distribution strategy: X, Substack Notes, HN, plus Reddit/Discords/aggregators (`planning/distribution-strategy.md`)
- Wiki bootstrapped and populated: 20 concept pages, 7 claims pages, 5 entity pages
- **Both source books fully ingested** — Kiely, *Inference Engineering* (Baseten) and Vizuara's
  *Definitive Workshop Guide*. Only Kiely's appendices remain.
- Week 1 drafted and revised against the books: `drafts/week-01-life-of-a-token.md`
- Three Week 1 figures + the series anchor figure: `drafts/figures/` (all generated from code)
- Visual strategy decided: `planning/visual-strategy.md`
- A weekly scheduled task drafts the digest every Sunday 7am Pacific (account-level; survives sessions)

**Open decisions**
1. **Naming.** KVScope's README says "The Inference Engineer"; the plan says "Inference from
   Scratch". Recommendation on the table: *The Inference Engineer* as the publication, *Inference
   from Scratch* as the flagship series inside it. Not yet decided.
2. Substack publication not yet created; `inference-from-scratch` repo not yet created.

**Settled decisions**
- **Running example model (decided 2026-08-23): 8B at BF16 is the anchor; 70B appears only as a
  contrast.** Every figure, ceiling, and derivation defaults to Llama-3.1-8B-class at BF16 — it is
  also the largest model that can actually be measured on a Mac and a single cloud GPU, which the
  "I measured" voice requires. 70B is used only where the scale contrast carries a point that the
  8B cannot. When reaching for it, note that the batch-1 compute/bandwidth imbalance is
  **model-independent** (it equals the hardware's ops:byte ratio, ~295 on H100) — what 70B changes
  is capacity and absolute ceiling, not the ratio. Getting this wrong produced a real error in the
  Week 1 draft; see the changelog note in `wiki/log.md`.

**Next actions**
- Run the Week 1 code on real hardware (a Mac and a cloud GPU) and replace predicted tok/s with
  measured. The voice requires this before publishing — "I measured" must be true. **The Week 1
  draft has been rewritten to promise these measurements in Week 4 rather than claim them.** If the
  runs happen before publishing, the hedges in the intro, the MacBook bullet, and the Qwen/Llama
  config note can be upgraded to plain statements — and `wiki/claims/decode-bandwidth-ceilings.md`
  can move the 8B/H100 entry from `[sourced]` to `[verified]`.
- Draft Week 2 (prefill vs decode). The derivation plan is already recorded in
  `wiki/concepts/prefill-decode.md` — the `24Nd² + 4N²d` formula, the fact that Week 1's
  "2 FLOPs per parameter" is its N=1 case, and the `N = 6d` crossover (~24,600 tokens for an 8B)
  where attention FLOPs finally overtake the linear layers.
- Housekeeping: a fine-grained GitHub PAT was shared in an earlier chat session and is unusable from
  Cowork (the git proxy blocks it). **It should be revoked** if that hasn't happened.

## How to work here

**Ingesting a source** (paper, book, release notes, blog post): follow `WIKI_SCHEMA.md` exactly —
create the `sources/` stub, update every relevant concept/entity/claims page, update `wiki/index.md`,
append to `wiki/log.md`, commit as `ingest: <source title>`. Large PDFs: split them and read in
20-page chunks; fanning subagents across chunks works well and keeps the main context clean.

**Drafting a post**: start from the relevant `wiki/concepts/` page (several are already effectively
post skeletons — `inference-metrics.md` for Week 4, `inference-engines.md` for Week 9,
`vllm-internals.md` for the Rust arc). Cite claims, not memory.

**Figures**: `drafts/figures/gen.py` and `anchor.py` generate everything; `shot.py` rasterizes to 2×
PNGs. Upload PNGs to Substack — it does not handle SVG reliably. Regenerate rather than redraw.

**Committing**: small, descriptive commits. `wiki/log.md` is append-only and is the operation record.
