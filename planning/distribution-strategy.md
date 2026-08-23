# Distribution Strategy — X, Substack Notes, and Hacker News

*Goal: turn the series into subscribers without LinkedIn. Primary channels: X, Substack Notes, HN (spike events). Secondary: Reddit, community Discords, aggregator newsletters, GitHub, SEO — see "Beyond the big three." The unifying principle: you're not "promoting a blog," you're being a useful inference engineer in public — the blog is where the depth lives.*

---

## The asset advantage (use it)

Most newsletter authors promote with "new post 🧵👇". You have three assets they don't:

1. **Numbers nobody else publishes** — every deep dive produces original measurements (KVScope reports, napkin-math-vs-measured tables). Charts with *your own data* are the single most shareable object in technical X.
2. **A tool** — KVScope gives you build-in-public content between posts: a new metric shipped, a surprising profile, a bug that taught you something.
3. **A visible arc** — "we're building a vLLM-class engine in Rust by Week 30" is a serialized story people follow like a show. Every post is an episode, not an article.

Everything below is about converting these three into engagement.

## X strategy

**The core loop (per deep dive):** don't post a link — post the *insight*, then let the link ride along.

- **Tuesday (post day):** a 6–10 tweet thread that delivers the post's one "screenshot moment" in full — e.g., the weights÷bandwidth formula predicting a MacBook's and an H100's tok/s with the same equation, with the chart. The thread must be valuable *without clicking*. Last tweet: "full derivation + code: [link]". Threads that are complete in themselves get shared; threads that are teasers get ignored.
- **Thursday:** repackage one atom from the same post in a different format — the misconception tweet ("temperature=0 does not make your LLM deterministic in production — here's why"), a single chart, or a 30-second napkin calculation. One post = 3–4 X assets across the week.
- **Weekend:** one build-in-public note — KVScope progress, a surprising profile result, next week's question as a teaser ("why is reading 1,000 tokens cheaper than writing 1?").

**The reply strategy (this is where followers actually come from at zero audience):** for the first 2–3 months, spend more time replying than posting. Turn on notifications for the people your readers already follow — vLLM/SGLang maintainers, inference engineers at labs and inference providers (Together, Fireworks, Baseten, Modal), SemiAnalysis, GPU-poor/llama.cpp community figures. When they post about serving, reply within the hour with something substantive: a measurement, a correction, a "here's the napkin math on that." A good reply under a 500K-view post outperforms your own tweet by 100x at your current size. Your bio does the conversion: "Building an inference engine in public. Deep dive every Tuesday → [link]".

**Misconception-bait, used honestly:** each post's "things engineers get wrong" section is your engagement engine. Frame as confident, falsifiable claims — technical X cannot resist correcting or confirming. Never rage-bait; always resolve with the real explanation.

**Cadence target:** 1 thread + 3–5 standalone tweets + 10–15 substantive replies per week. Consistency beats volume; don't tweet filler on quiet days.

**What to skip:** engagement-farming formats ("agree?", polls-for-polls'-sake), AI-generated-looking image cards, and threads about threads. Your differentiation is that you're the person with real numbers.

## Substack Notes strategy

Notes is a different animal: smaller reach per hit than X, but every impression is someone *already in the newsletter-reading, subscribe-button-pressing mindset* — conversion per view is far higher, and the algorithm favors conversation over follower count, so a zero-follower account can travel. Treat it as its own channel, not an X mirror.

- **Daily-ish short notes (5 min/day):** one idea per note, written as a thought rather than an announcement. The formats that work: the counterintuitive fact ("An H100 generating one response at a time uses ~1% of its compute"), the napkin calculation, the "what I learned building KVScope this week", the honest process note ("wrote 2,000 words on KV caches today and deleted half"). Notes readers reward voice and process more than X does.
- **Restack every post with a one-line take** — restacks are Notes' native share unit and put you in your subscribers' followers' feeds. Also restack *other* inference/systems writers with substantive comments; the writers themselves are your most valuable early audience, and Notes is small enough that the big accounts actually see you.
- **Engage in the ML/engineering writer cluster:** comment on posts from adjacent Substacks (systems, MLOps, AI infra). On Substack, the author you thoughtfully comment on today recommends your publication next month — and **recommendations are the #1 organic growth lever on the platform**, worth more than any individual viral note. Target: earn 3–5 recommendations from adjacent technical Substacks in the first 3 months by being their best commenter and recommending them first.
- **Use Notes for the digest, too:** each Friday digest yields 2–3 quotable one-liners ("vLLM shipped X this week — here's why it matters in one paragraph"). News travels well on Notes.

## Hacker News strategy

HN is not a weekly channel — it's a spike machine, and you get a limited number of credible shots. Save them for posts with genuine "front page shape":

- **Best candidates:** Week 1 (life of a token — classic HN explainer shape), the napkin-math post if split out, Week 4 (Show HN: KVScope), Week 6 (PagedAttention explainer), Week 25+ ("We're building a vLLM-style engine in Rust" and the Week 30 showdown benchmark). A "Show HN" for KVScope is its own separate shot from the blog posts.
- **Mechanics:** submit yourself (weekday morning US time), plain title without numbers-hype, and *stay in the thread all day* replying to every technical question — HN converts on author presence in comments. Never ask friends to upvote (ring detection buries you).
- **Expectation-setting:** most submissions go nowhere; one front page = hundreds to low thousands of subscribers. Submit the strong candidates, ignore the misses, resubmit reworked material months later.

## The weekly rhythm (all channels, ~4–5 hrs/wk)

Tuesday: deep dive publishes → X thread + restack with take on Notes + submit to two aggregator newsletters. Wednesday: reply day on X. Thursday: second X asset (misconception/chart) + 1–2 Notes. Friday/Monday: digest publishes → 2 quotable notes + digest one-liner on X. Weekend: build-in-public KVScope note. Reddit: 1–2 native posts/month when a finding has the right shape. HN: only when a post has front-page shape.

## Launch sequence (first 3 weeks)

You launch with the buffer (Weeks 1–3 written), so: publish Week 1 → X thread same morning → HN submission next day (it's your strongest HN candidate; day-2 submission avoids splitting attention). Notes daily from one week *before* launch so the account isn't empty when traffic arrives. Pin a tweet/note that states the series premise and arc ("from a 50-line loop to a Rust inference engine in 30 weeks") — the arc, not the topic, is what makes people subscribe rather than just read.

## Measurement (check monthly, not daily)

Track only: subscribers by source (Substack shows this), X profile-visit→follow rate on thread weeks, Notes-driven subscriptions, and recommendations earned. If X threads get views but no subs, the threads are too complete (rebalance toward the arc); if Notes converts better per hour spent (it usually does early), shift time there. Ignore likes entirely.

## Beyond the big three — secondary channels

**Reddit (start immediately).** r/LocalLLaMA is the densest audience of tokens-per-second obsessives anywhere; r/MachineLearning for the paper-adjacent posts. Post natively — the chart, the finding, the full explanation in the Reddit post itself, newsletter linked once at the bottom. Bare link-drops die; "I measured decode ceilings across 6 GPUs, here's the data" threads thrive. Strong candidates: napkin-math results, KVScope findings, every Phase 6 Rust milestone (the Week 30 showdown especially). Cadence: 1–2 posts/month, only the ones with genuinely native shape.

**Community Discords/Slacks (start now, share later).** vLLM Discord, GPU MODE (CUDA/kernels community), EleutherAI, MLOps Community Slack. Same posture as the X reply strategy: weeks of being helpful before any self-linking, then share only into directly relevant discussions. One respected member re-sharing your post beats a hundred impressions elsewhere. This is also where future collaborators and technical reviewers come from.

**Aggregator newsletters (standing weekly item).** TLDR AI, AlphaSignal, Ben's Bites, Latent Space link roundups, Hackernewsletter, Pointer, daily.dev. Submit each Tuesday post to two of them — costs one email, and a single placement can be worth hundreds of subscribers. Track which ones ever bite and double down there.

**GitHub as a channel.** PR the repos into the lists readers actually browse (Awesome-LLM-Inference, the KV-cache optimization survey lists). Per-concept README links back to the relevant deep dives. Ship Week 1's code as an open-in-Colab notebook — notebooks travel on their own. KVScope's own README is the top of a funnel: every star is a potential subscriber.

**SEO (free, compounding).** "What is a KV cache," "prefill vs decode," "PagedAttention explained" are durable queries with weak competition, and curriculum posts are shaped like the answers. Costs nothing beyond descriptive titles/subheads and stable URLs. If cross-posting to Medium/Dev.to for reach, always set canonical back to Substack.

**Seed now, harvest later.** Conference/meetup CFPs (MLSys-adjacent meetups, PyTorch Conference, Ray Summit, vLLM meetups): submit around Week 8–10 so slots land when the series is mature. Podcasts (Latent Space, Changelog, infra shows): pitch at ~1K subscribers when the Rust arc gives you a story. LinkedIn: same traction gate, per plan. Bluesky: cheap mirror at most — the AI-infra crowd remains on X.

**Sequencing:** Reddit + aggregator submissions into the weekly rhythm now; inhabit vLLM and GPU MODE Discords from week one (quietly); awesome-list PRs once the repo has ~3 weeks of content; CFP submissions calendared for ~2 months in.

## What compounds vs. what spikes

Compounding: reply-strategy relationships, Substack recommendations, the KVScope build-in-public narrative, the archive's course structure. Spikes: HN front pages, viral threads. Play for the compounding assets and treat spikes as found money — a subscriber from a recommendation retains far better than one from a viral hit.
