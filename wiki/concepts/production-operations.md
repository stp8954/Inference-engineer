# Production operations

## What it is
The infrastructure layer around the engine (Kiely Ch 7). Highlights relevant to the series: cold starts decompose into GPU procurement → image load → weight load → engine startup (TensorRT-LLM/compiled-PyTorch engine builds take minutes; cache engines per exact GPU/CUDA combo; never bake weights into images — load from a same-datacenter cache at GB/s). Autoscaling should combine utilization signals (lagging) with traffic signals (leading); concurrency targets should match batch size. Routers are request-level, load balancers system-level; smart routing is KV-cache-aware and LoRA-aware (Dynamo routes on sequence length, prefix, LoRA). Deploys: canary beats blue-green for GPU fleets. Cost: compare total API token cost vs GPU-hours over ≥1 week, count engineering time (TCO).

**Replication and routing (Vizuara Ch 18).** Replication is linear in both throughput and cost — 1 replica at 20 users / 3,000 tok/s becomes 8 replicas at 160 users / 24,000 tok/s for ~8× the money. The interesting part is routing: round-robin ignores load and breaks under heterogeneous requests; least-busy needs sub-second telemetry; **sticky / KV-cache-aware routing** sends a returning user back to the replica holding their prefix cache, cutting average TTFT 2–3× for multi-turn chat. Production norm is sticky-with-least-busy-fallback.

**Cold start is dominated by weight loading:** VM allocation ~30 s → image download ~60 s → **model weights ~140 s** (140 GB at ~1 GB/s) → CUDA graph capture ~10 s ≈ **4 minutes**, which is far too slow to answer a traffic spike. Hence warm pools: idle replicas activated in seconds, costing ~20–30% extra GPUs — the price of holding P99 during bursts.

**Fleet operations (Vizuara Ch 18).** Autoscaling rule: **scale up fast, scale down slow** — cold starts are expensive, brief over-provisioning is cheap. Triggers: up when P99 TTFT breaches SLO for 30–60 s or utilization exceeds 70–80%; down when utilization stays under 30% for several minutes. The router is the fleet's control plane — stateless and replicated, with state living in observability — dispatching by user tier, model, and SLO to distinct pools. The full stack is six layers (client → edge/CDN → global LB → regional router → replica pool → GPU), each adding 2–40 ms in each direction.

**Reliability:** health checks every 2–5 s detect a crash in ~5 s, but replacement takes a ~4-minute cold start, during which survivors absorb load. N+1 provisioning (~20% over peak) absorbs a single failure without breaching SLO; stricter domains want N+2 or active-active multi-region.

**Spot capacity** is 60–70% cheaper but reclaimed on ~30 s notice — use for batch only, never interactive; on-demand core plus spot burst saves 20–30% of the GPU bill.

## Key numbers
- Reliability base rate: Llama 3 training — 16,000 GPUs, 54 days, 419 unexpected interruptions ≈ 1 failure per 50,000 GPU-hours (Grattafiori et al. 2024, via Kiely §7.3.3). [sourced]
- Geo latency rule of thumb: ~5 ms per time zone crossed. [sourced] — Kiely §7.3.2.
- Cost breakdown per million tokens: **GPU rental 55%, idle/headroom capacity 17%, cold-start overhead 12%, network egress 8%, orchestration/routing 5%, observability 3%**. Idle capacity is the hidden line item — real fleets run 60–70% average utilization to hold spike headroom. [sourced] — Vizuara §18.7.
- Multi-region: Singapore→Virginia ≈ 180 ms at fiber speed (~200 ms with routing) *before* inference starts; P99 TTFT 200 ms in-region vs 400–900 ms cross-region. Four regions ≈ 4× fixed cost — start with one, add when latency (not capacity) is the limiter. [sourced] — Vizuara §18.5.
- Cost-vs-concurrency curve: cost per token bottoms out at **70–80% average utilization**; over-provision by ~20% for P99 safety and no more. [sourced] — Vizuara §25.3.
- Worked cost example: 1,000M input @ $1.25/M + 500M output @ $10/M = $6,250 vs 1,600 GPU-hrs @ $3.50 = $5,600. [sourced] — Kiely §7.4.2.
- Pipeline locality: intra-cluster ~10 ms vs inter-cluster ~50 ms per hop — 40 ms × 5-step pipeline = 20% of a 1 s SLA. [sourced] — Kiely §7.2.5.

## Open questions
- How much of Ch 7 belongs in the series vs a "further reading" pointer — candidate for a bonus post after Week 24.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — Ch 7.

## Series mapping
- Week 24 (economics), background for Weeks 17–18.
