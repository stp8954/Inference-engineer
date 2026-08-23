# Wiki Schema

How the `wiki/` knowledge base is structured and how an LLM (or human) should maintain it. This file is the contract; follow it in every session.

## Layers

1. **Raw sources — immutable.** PDFs and long documents live in the Google Drive inbox folder (`inference-engineer-inbox`); web sources are referenced by URL. Each ingested source gets a stub in `sources/` named `YYYY-MM-DD-short-slug.md` recording: title, authors, URL/Drive filename, date ingested, one-paragraph summary, and the list of wiki pages it touched. Never edit a source stub after creation except to append.
2. **The wiki — LLM-owned.** Markdown pages under `wiki/`, freely rewritten as understanding improves.
3. **This schema — the rules.** Update it deliberately and log the change in `wiki/log.md`.

## Page types

- `wiki/concepts/<slug>.md` — one page per technique or idea (kv-cache, speculative-decoding, …). Sections: **What it is** (2–4 paragraphs), **Key numbers** (with provenance links), **Open questions**, **Sources**, **Series mapping** (which week(s) cover it, link to draft/post when it exists).
- `wiki/entities/<slug>.md` — one page per project, company, or hardware platform (vllm, sglang, kvscope, …). Sections: **What it is**, **Timeline** (dated notable releases/events, append-only), **Relation to the series**, **Sources**.
- `wiki/claims/<slug>.md` — quantitative claims worth citing in posts, one file per tight topic (e.g. `decode-bandwidth-ceilings.md`). Each claim is one bullet: the claim, the number, the exact source (paper + table/section, or benchmark run), date recorded, and a confidence tag: `[verified]` (we reproduced it), `[sourced]` (primary source, not reproduced), `[hearsay]` (secondary source — needs upgrade before use in a post).

## Navigation files

- `wiki/index.md` — every wiki page by category with a one-line summary. Update in the same commit as any page add/remove.
- `wiki/log.md` — append-only. One line per operation: `YYYY-MM-DD [INGEST|QUERY|LINT|SCHEMA] description (pages touched)`.

## Operations

- **Ingest**: read the source → create its `sources/` stub → update every relevant concept/entity/claims page (a typical paper touches 3–10 pages) → update `index.md` → append to `log.md` → commit with message `ingest: <source title>`.
- **Query**: when answering from the wiki, cite wiki pages; if the synthesis is durable (e.g., "everything we know about MLA for Week 13"), file it as a new page rather than losing it to chat history.
- **Lint**: periodically check for contradictions, stale claims (esp. `[hearsay]` older than a month), orphan pages, and missing series mappings. Log as `LINT`.

## Conventions

- Relative links between pages (`../entities/vllm.md`) so the wiki browses on GitHub and in Obsidian.
- Every number that could appear in a blog post must trace to a claims entry. Posts cite claims; claims cite sources. This provenance chain is what licenses the series' plainly-stated confidence (see `planning/series-plan.md`, Voice section).
- Blog workflow: `drafts/week-NN-slug.md` → published → move final text to `posts/week-NN-slug.md` with the Substack URL in front-matter; add a changelog section at the bottom of the post file for post-publication corrections.
- Commits from cloud sessions push directly; commits from home sessions go through the local clone. Always `git pull --rebase` before starting an ingest.
