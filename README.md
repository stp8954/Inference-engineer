# inference-engineer

The knowledge base and writing home for **Inference from Scratch** — a blog series learning LLM inference in public, from a naive PyTorch loop to a vLLM-style engine in Rust — and its weekly companion digest, **This Week in Inference**.

Code for the series lives in a separate repo: [`inference-from-scratch`](https://github.com/stp8954/inference-from-scratch). The KVScope profiler, the series' companion tool, lives at [`KVScope`](https://github.com/stp8954/KVScope).

> **Working on this repo with Claude?** Start with [`CLAUDE.md`](CLAUDE.md) — current state, open
> decisions, next actions, and the conventions.

## Layout

| Path | What it holds |
|---|---|
| `planning/` | Series plan, distribution strategy, per-post brainstorms |
| `drafts/` | Posts in progress (`week-NN-slug.md`) |
| `posts/` | Final published versions, as shipped to Substack |
| `wiki/` | The LLM-maintained knowledge base (see `WIKI_SCHEMA.md`) |
| `sources/` | One metadata stub per ingested source; raw PDFs live in Google Drive |

## The wiki

`wiki/` follows the [LLM wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): an LLM ingests raw sources (papers, release notes, blog posts) and compiles them into interlinked markdown pages — concept pages, entity pages, and provenance-tracked claims — that stay current as new sources arrive. `wiki/index.md` catalogs every page; `wiki/log.md` is the append-only operation log. Conventions live in `WIKI_SCHEMA.md`.

Humans are welcome to read and correct anything; the LLM does the bookkeeping.
