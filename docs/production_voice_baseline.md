# Production Voice Baseline

Status: frozen for production workflow validation  
Date: 2026-09-13

## Canonical configuration

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Revision: `fd4b254389122332181a7c3db7f27e918eec64e3`
- Runtime: `cuda:0`, `float16`, `sdpa`
- Voice profile: `bernardo_xvec_v3`
- Voice identity: `bf047b087b09e7a2566a1282a8a9eeb4a043ef44051b2030859ca94ae46dde4f`
- Reference: `ref_full_processed_raw_normalized.wav`
- Reference SHA-256: `11b7b30c41c3d365c4122a1549cd505d16876219f75fa213f87fa15e341be546`
- Clone mode: `x_vector_only` (no reference transcript)
- Generation profile: `narration_v2`
- Generation identity: `e5433e1cb8b20f91913ad37763b891f3b327a50be2f5c7d09e9a5fdaa97f05d7`
- Production base seed: `303` (owned by the Orchestrator)
- Sampling: `do_sample=true`, `temperature=0.95`, `top_p=1.0`, `top_k=50`,
  `repetition_penalty=1.05`, `subtalker_dosample=true`,
  `subtalker_temperature=0.90`, `subtalker_top_p=1.0`, `subtalker_top_k=50`,
  `max_new_tokens=2048`.

The old voice and generation profiles, including `bernardo_xvec_v2`, remain registered and
immutable for historical
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
- Speaker drift remains possible; ASR is not a speaker verifier. The runtime now records
  warning-only, chunk-level `qwen_ecapa_warning_v1` similarity diagnostics against the frozen
  reference. The production voice/generation identities and acceptance policy remain unchanged.
- Pronunciation lexicon work remains a separate, reviewed Orchestrator workflow.
- Post-processing is not part of this freeze. The existing native WAV, deterministic assembly and
  transparent mastering path is unchanged.

## 2026-09-09 controlled candidates

- `narration_nonstreaming_v1` identity
  `0de37fd2129c17ebcc21628436b2a5d464e804d8c4aad13b12da86ebdfd079ab` adds only
  `non_streaming_mode=true`. Two of five comparison samples fell below the warning threshold,
  versus zero of five for `narration_v2`; it was not promoted.
- `bernardo_icl_candidate_v1` identity
  `f6462cfb31e37e75406336f84fbc722463018f8307bdec7116f3d96356a09739` uses the same frozen
  reference plus its ASR-validated transcript. Pace changed and one of five samples warned; blind
  review is still required, so it was not promoted.
- Existing temperature 0.80/0.70 candidates showed only a small single-text similarity/pace
  trade-off. This is insufficient to replace the listening-selected production temperatures.
- Warmup added about three seconds of explicit GPU work and did not reduce the measured first-call
  wall time; `WARMUP_ENABLED=false` remains the operational default.

See `upstream_audit_2026-09-09.md` for upstream release/PR status and benchmark decisions.
