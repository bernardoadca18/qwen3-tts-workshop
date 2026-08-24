# Qwen3-TTS Voice Service agent guide

This repository owns provider-neutral voice synthesis, voice/generation profiles,
durable narration jobs, deterministic chunking, atomic WAV output and validation.
It does not own pronunciation rewriting, final video-level mastering, narration
alignment, editorial decisions, composition or rendering.

Runtime code is under runtime/voice_service, tests under tests, named profiles
under config and explicit benchmark/smoke tools under scripts. Unit tests must
use a fake provider and must never load Qwen, CUDA or the network.

Keep model loading in FastAPI lifespan, one bounded queue and one inference
worker. Preserve submitted text exactly, persist resolved seeds and profile/model
identity, preserve native sample rate, never crossfade independent utterances,
and mark completed only after validated atomic publication. Do not add hidden
provider retries, brokers, multiple GPU workers, raw Qwen controls to the stable
API, or final LUFS processing.
