# Multimodal and embodied inference

## What it is
The frontier chapter (Vizuara Ch 23), organized around one idea: **every modality becomes tokens, so the only variable that changes is token rate per second of real-world signal** — and that rate sets both the KV budget and the latency budget. Text typing is ~2 tok/s; voice ~50 tok/s; audio ~75 tok/s; video ~7,680 tok/s. Three orders of magnitude between text and video, on the same forward pass.

**Voice** is a latency-budget problem: humans notice above ~300 ms and it feels live below 200 ms. A cascade (VAD → ASR → LLM → TTS) spends ~50 + 80 + 120 + 50 ms — leaving ~50 ms of slack across the entire pipeline. Native voice-token models skip the text round trip at the cost of debuggability.

**Video** is a KV-capacity problem: a 224×224 frame is ~256 tokens, so a one-minute 30 fps clip is ~460K tokens ≈ 59 GB of KV for a 7B model — an H100 is exhausted at roughly 80 seconds of video. MLA, quantization, and paging push the crossing right but don't beat linear growth; production systems subsample frames or pool spatially.

**Two roofline points per request.** Encoder/tokenizer prefill sits at AI ≈ 200–500 (compute-bound) while LLM decode stays at AI ≈ 1 (memory-bound). Every multimodal request switches regimes mid-flight, which is an even stronger argument for disaggregated prefill/decode than text-only serving: encoders want FLOP-tuned GPUs, decoders want bandwidth and KV capacity.

**Embodied** inference is where latency becomes physical. A robot control loop at 30 Hz must fit perception + policy + actuation into 33 ms — the 300 ms human perceptual budget is generous by comparison. VLA models (RT-2, OpenVLA, RDT-1B; 3B–55B params) emit action tokens from image + language + proprioception tokens, and the same toolkit applies: KV-cache the invariant instruction across control steps, quantize for the edge, speculate for faster loops. World models are video decoders with the same autoregressive drift problem (rollouts diverge after ~30 frames), so 2026 production splits hybrid: a fast on-device reflex policy plus ~1 Hz calls to a big cloud planner.

## Key numbers
- Token rates per second of signal: text ~2 (typing) / ~10 (reading); voice ~50; audio ~75; video ~7,680. [sourced] — Vizuara §23.1.
- Video KV: 256 tok/frame × 30 fps × 60 s = **460,800 tokens ≈ 59 GB** for a 7B (128 KB KV/token); crosses an 80 GB H100 at **~80 seconds** of clip. [sourced] — Vizuara §23.4.
- Voice 300 ms budget: VAD 50 + ASR 80 + LLM TTFT 120 + TTS 50 ms. [sourced] — Vizuara §23.2.
- Audio tokenization compresses ~200:1 (1 s of 16 kHz → 50–75 tokens) via HuBERT/EnCodec/wav2vec-class encoders against a 1,024–4,096-entry codebook. [sourced] — Vizuara §23.3.
- Robot control at 30 Hz: 33 ms total — perception 10–15 ms, policy 15–20 ms, the rest for dispatch. Edge caps around 4B params at FP8 in a 15 W envelope (Jetson Orin ~200 TFLOPS FP16, 0.2 TB/s); cloud round trips are 50–200 ms. [sourced] — Vizuara §23.8, §23.10–11.
- Per-frame world-model cost: AR naive 2,000 ms → optimized 40 ms; diffusion naive 3,000 → optimized 80 ms; real-time floor is 33 ms. Latent-space rollouts cut per-step cost 50–200×. [sourced] — Vizuara §23.7.

## Open questions
- Where does this fit the series? Candidate: a Week 23-adjacent bonus post, or fold the voice-latency budget into Week 22 (serving agents) since it's the most concrete "latency is a product feature" example in either book.

## Sources
- [Vizuara, *Workshop Guide* (2026)](../../sources/2026-08-22-vizuara-workshop-guide.md) — Ch 23.

## Series mapping
- Bonus / frontier post; voice budget material also supports Week 22.
