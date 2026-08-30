# Production Voice Baseline

Status: frozen for production workflow validation  
Date: 2026-08-29

## Canonical configuration

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Revision: `fd4b254389122332181a7c3db7f27e918eec64e3`
- Runtime: `cuda:0`, `float16`, `sdpa`
- Voice profile: `bernardo_xvec_v2`
- Voice identity: `c4e74a0da51bd5fbc18c4d9b0bc5623d2e4ea91ff8d873374b6c1e41632b802b`
- Reference: `ref_full_processed_raw.wav`
- Reference SHA-256: `8856924e05f76d1a3a5b8554218f80b5e1a898aebfe211ec1b65b60eabdbc70b`
- Clone mode: `x_vector_only` (no reference transcript)
- Generation profile: `narration_v2`
- Generation identity: `e5433e1cb8b20f91913ad37763b891f3b327a50be2f5c7d09e9a5fdaa97f05d7`
- Production base seed: `303` (owned by the Orchestrator)
- Sampling: `do_sample=true`, `temperature=0.95`, `top_p=1.0`, `top_k=50`,
  `repetition_penalty=1.05`, `subtalker_dosample=true`,
  `subtalker_temperature=0.90`, `subtalker_top_p=1.0`, `subtalker_top_k=50`,
  `max_new_tokens=2048`.

The old voice and generation profiles remain registered and immutable for historical
reproduction. Profile names, versions, reference bytes, model revision and controls produce new
canonical identities. Startup fails closed if the frozen reference no longer matches its declared
digest.

## Decision record

Blind reference listening selected `ref_full_processed_raw.wav`; the runner-up was
`enhanced_ref_full_processed_raw.wav`. X-vector-only cloning was preferred over ICL for rhythm,
prosody, naturalness and overall pronunciation. Seed `303` won the final blind seed sweep. With
that seed, `temperature=0.95` / `subtalker_temperature=0.90` was the best overall naturalness and
stability compromise; main `1.00` was less consistent and changing the subtalker did not provide a
global benefit.

The selected samples showed no catastrophic speaker drift, runaway or serious content loss. This
is not a claim of model perfection. Deterministic drift can still occur; seeds `1337`, `8675309`
and `202` remain useful benchmark canaries and are intentionally not production defaults.

## Boundaries and known limitations

- Existing duration, artifact, ASR/content and retry guards remain authoritative and unchanged.
- Speaker drift remains possible; ASR is not a speaker verifier.
- Pronunciation lexicon work remains a separate, reviewed Orchestrator workflow.
- Post-processing is not part of this freeze. The existing native WAV, deterministic assembly and
  transparent mastering path is unchanged.
