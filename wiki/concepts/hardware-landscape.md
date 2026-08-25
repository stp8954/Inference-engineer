# Hardware landscape

## What it is
What the series' Week 23 covers, seeded from Kiely Ch 3. NVIDIA generations (letter = architecture, number = model): Volta '17, Turing '18, Ampere '20, Lovelace '22, Hopper '22, Blackwell '24, Rubin '26, Feynman '28; work within the 3–5 newest. Hopper (H100/H200) = most widely used, first FP8; Blackwell (B200/B300) = current standard, adds FP4 + microscaling, FA4-era async pipelines; Lovelace (L4/L40) lacks NVLink — L4 fine for small models, L40 usually loses to H100 MIG slices. Rubin brings HBM4 and CPX (a separate prefill-oriented compute chip — hardware-level disaggregation). Grace/Vera ARM CPUs pair via NVLink C2C (900 GB/s) — enables fast CPU-side KV/LoRA offload. Nodes = 8 GPUs (NVLink/NVSwitch intra-node; InfiniBand ≤400 Gb/s inter-node — always slower than NVLink, and not every cloud provides it). Rack-scale: GB200 NVL72 (72 GPUs + 36 Grace). MIG partitions one GPU into ≤7 slices for small models (Whisper, TTS). Non-NVIDIA bets: bandwidth (Cerebras WSE-3, Groq LPU), efficiency (Furiosa, Qualcomm), platform (TPU, Inferentia/Trainium), plus AMD MI350, Etched Sohu, SambaNova — all fighting the CUDA moat. Local inference: Apple unified memory (capacity) vs discrete GPUs (bandwidth); phones cap out ~1–2B params; hybrid local+cloud is the direction.

**Inside the machine (Vizuara Ch 6).** An H100 is 132 SMs; each SM is 4 processing blocks holding 32 FP32 CUDA cores (128/SM) and one tensor core (4/SM, 528 total), 65,536 registers, and 228 KB of shared SRAM. Warp switches are free (state lives in the register file), which is how the GPU hides HBM's ~600-cycle latency. Tensor cores are scarce and large — using them well is the difference between a 10%-of-peak and an 80%-of-peak kernel; without them the FP32 cores alone give only 67 TFLOPS.

**Memory hierarchy** — bandwidth rises ~10× and capacity falls ~10× per tier: registers (256 KB/SM, fastest) → SRAM/shared (228 KB/SM, ~20 TB/s) → L2 (50 MB, ~5 TB/s) → HBM (80 GB, 3.35 TB/s, ~600-cycle latency). Optimization means moving the working set *up* the hierarchy — which is exactly what FlashAttention does (a 64×64 FP16 tile is 8 KB, so the score matrix lives in SRAM and never touches HBM). Long context is therefore a hardware problem: a 100K-token KV cache fits nowhere but HBM and must stream every decode step.

**Spec-sheet literacy (§6.10)** — read any new GPU with six numbers: HBM bandwidth (roofline slope + ITL floor), per-precision tensor TFLOPS (the ceilings), SRAM/SM (tiling budget), SM count (parallelism and minimum batch to saturate), NVLink bandwidth (intra-node parallelism), and inter-node bandwidth (cross-node parallelism). Note that B200 raises bandwidth 2.4× and compute ~2.3× together, so **the ridge point stays roughly put across generations** — the memory-bound problem does not get solved by new silicon.

**Consumer GPUs are a serving tier now, not just a dev box (FreeToken, 2026).** The local-inference
line above — Apple unified memory for capacity, discrete GPUs for bandwidth — understates what
changed once MoE models became the frontier default. On a single user's machine the sparsity holds
completely, so the binding constraint moves from "does the checkpoint fit in VRAM" to "how often does
a routed expert have to cross PCIe." That reframes an RTX-class card plus a large pool of host RAM as
one heterogeneous platform rather than a small GPU with a fallback, and it puts 35B-to-753B models on
hardware that cost a few thousand dollars. The consequence for Week 23: the interesting spec on a
consumer box is no longer VRAM alone but the *ratio* of VRAM to host bandwidth, and PCIe generation
becomes a first-order number rather than a footnote.

## Key numbers
- See [claims/hardware-specs.md](../claims/hardware-specs.md) for the spec table.
- Edge serving reach: 8 GB laptop GPUs through single workstation GPUs; 35B on laptops to 753B on one workstation GPU (NVIDIA RTX 30/40/50 series). [sourced] — FreeToken abstract (see ../entities/freetoken.md). Recorded 2026-08-25.

## Open questions
- InferenceMAX/MLPerf cross-vendor numbers for Week 23 (independent of vendor claims).
- `hardware-specs.md` carries no PCIe Gen4/Gen5 x16 bandwidth figures. Needed before the offload-regime arithmetic can be stated numerically anywhere.

## Sources
- [Kiely, *Inference Engineering* (2026)](../../sources/2026-08-22-kiely-inference-engineering.md) — Ch 3 (§3.2–3.5).
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 6 (SM internals, memory hierarchy, warps/blocks/grids, NVLink vs IB, spec-sheet checklist).
- [Yang et al., *FreeToken* (2026)](../../sources/2026-08-25-yang-freetoken.md) — consumer GPUs as an edge MoE serving tier.

## Series mapping
- Week 23 (primary); MIG relevant to Week 24 economics.
