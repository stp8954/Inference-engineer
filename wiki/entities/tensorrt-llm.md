# tensorrt-llm

## What it is
NVIDIA's inference engine/library, kernel-focused. GitHub - NVIDIA/TensorRT-LLM. Best peak performance of the big three via handwritten (partly closed-source) fused kernels; best Hopper/Blackwell and NVFP4 support; NVIDIA-only; steepest learning curve; no image/video models. Versioning trap: V0 (0.x) was a TensorRT plugin; V1 (1.x, summer 2025) is standalone PyTorch-based with no TensorRT dependency.

## Timeline
- Per Kiely: "in-flight batching" (their name for continuous batching); hardware-targeted engine compilation takes minutes (cache engines; only valid on exact GPU/CUDA combo — slowest cold starts of the big three); XQA attention kernel (strong for embedding/encoder workloads); sibling tool ModelOpt is the leading PTQ toolkit (outputs run on vLLM/SGLang/TRT-LLM too); orchestrates draft-target speculation; the optimization backbone for Whisper/TTS-class models.

## Relation to the series
- Week 9

## Sources
- (pending first ingests)
