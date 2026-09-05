# NeoVids Qwen3-TTS Voice Service

Production-oriented, single-GPU voice-cloning service for NeoVids. Qwen3-TTS is
an internal provider implementation; callers use versioned voice and generation
profiles plus durable narration jobs. Kokoro, Seed-VC, pronunciation rewriting,
final video mastering and Orchestrator integration are intentionally absent.

## Architecture

The FastAPI lifespan validates typed settings, opens SQLite, validates configured
voice references, resolves one pinned Hugging Face snapshot, loads one Qwen model,
builds each configured ICL or speaker-embedding-only prompt once, optionally warms up, recovers durable jobs and
starts one GPU worker. Importing voice_service.main does not import qwen-tts or
torch and does not initialize CUDA.

Requests create durable queued jobs and return HTTP 202. A bounded in-process
queue wakes the single worker; SQLite remains the source of truth. The worker
chunks text deterministically, derives stable per-chunk seeds, synthesizes one
utterance at a time, joins utterances with explicit silence, validates a temporary
PCM16 WAV, atomically renames it and only then marks the job completed.

The deployment scope is deliberately one process, one NVIDIA GPU and one worker.
There is no broker, distributed lease or multi-GPU scheduler.

## Configure

Copy .env.example to .env and adjust only operational settings. Important values
include model/revision, device/dtype/attention, default profiles, queue and text
limits, SQLite/output locations, retention, warmup and pause durations.

MODEL_REVISION defaults to the exact currently validated snapshot
fd4b254389122332181a7c3db7f27e918eec64e3. qwen-tts 0.1.1 forwards kwargs to
the model but not the processor, so the provider first resolves that revision
with huggingface_hub.snapshot_download and loads the resulting local path. This
pins model and processor files together.

CPU with float16, unknown devices/dtypes/attention backends and inconsistent
chunk limits fail configuration validation. FlashAttention is selectable only
when installed and validated by the operator; sdpa is the conservative default.

## Voice profiles

config/voices.yaml maps trusted names to reference audio/transcript files. The
`bernardo_v1` profile uses ICL cloning (x_vector_only_mode=False) and points to
the operator's current `/data/references/ref.wav` master. It is versioned as
`2`; changing the bytes or transcript changes its identity. At
startup the service checks file presence, non-empty transcript, readable and
finite non-empty audio, then records profile/version, hashes, duration, format
and model/revision-bound identity.

`bernardo_xvec_v1` is a separate historical opt-in profile using `x_vector_only_mode=True` and the same raw
reference audio. It deliberately has no transcript: Qwen receives `ref_text=None`, discards
reference speech codes from the prompt item and conditions on the speaker embedding only. Clone
mode and transcript semantics participate in profile identity, so ICL and x-vector prompts cannot
share cache entries. `bernardo_v1` and `bernardo_xvec_v1` remain available for reproduction.
The production default is the frozen `bernardo_xvec_v2` profile; its configured reference digest
is verified at startup and its transcript-free x-vector semantics cannot silently fall back to ICL.

The Qwen prompt is cached by this identity and is not recomputed per narration. Registered profile
prompts are pinned for the service lifetime. Benchmark-only ephemeral prompts use a bounded LRU
(`MAX_EPHEMERAL_PROMPT_CACHE_ENTRIES`, default 8); an active prompt is never evicted. Readiness
exposes entry/hit/miss/eviction counters without reference paths or tensor contents.
A restart remains the explicit invalidation mechanism for registered prompts. GET /v1/voices
returns safe metadata and never host paths.

## Generation profiles

config/generation_profiles.yaml contains internal Qwen controls. The stable API
selects the frozen `narration_v2`; `narration_v1` remains available for historical reproduction.
The API does not expose Qwen sampling vocabulary. Changing a
profile version or controls changes its identity and job provenance.
GET /v1/generation-profiles exposes only name, version and canonical identity
so callers can invalidate derived audio without learning provider controls.

Pronunciation preprocessing and the operator Pronunciation Lab belong upstream
in the Orchestrator. This service accepts already-resolved synthesis text,
preserves the submitted text exactly in job storage and never loads an editorial
lexicon or performs pronunciation rewriting.

## Run with Docker

~~~bash
docker compose config
docker compose build
docker compose up -d
curl http://localhost:8889/health/live
curl http://localhost:8889/health/ready
~~~

`/health/live` responds as soon as the event loop is serving, including while
the model initializes. `/health/ready` returns HTTP 503 with a safe
`initializing`/failure reason until provider, profiles and worker are usable.
Successful `GET /v1/narrations/<job-id>` status polls are filtered from the INFO
access log so model/chunk/guard/completion events remain visible; polling errors,
artifact requests and health requests are still logged normally.

Compose retains the Hugging Face cache, SQLite database, final outputs and
references across rebuilds. References/config are read-only mounts; jobs and
outputs are separate persistent directories. The service binds to localhost,
keeps NVIDIA reservation and uses readiness for its healthcheck.

## Async narration API

Create a job:

~~~bash
curl -i -X POST http://localhost:8889/v1/narrations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: narration-scene-42' \
  -d '{
    "text": "Geralt aprende o sinal Aard.",
    "language": "Portuguese",
    "voice_profile": "bernardo_xvec_v2",
    "generation_profile": "narration_v2",
    "seed": 303
  }'
~~~

seed=0 securely resolves once. The persisted response includes the resolved seed,
job state and status/audio URLs. Poll and download:

~~~bash
curl http://localhost:8889/v1/narrations/JOB_ID
curl -o narration.wav http://localhost:8889/v1/narrations/JOB_ID/audio
~~~

States are exactly queued, running, completed, failed. Status exposes safe
progress, model/profile provenance, result format/checksum/metrics, and a bounded
public error category/message. It never returns source text, stack traces or host
paths. Audio is available only after completion.

The deprecated synchronous POST /generate remains for current laboratory calls
and uses the same durable job/worker pipeline. Its historical default raw values
are accepted; non-default raw Qwen overrides are rejected. New consumers must
use /v1/narrations.

## Idempotency and determinism

An Idempotency-Key identifies the canonical input fingerprint: exact text,
language, profile identities, requested seed, model identity/revision and the
versioned effective chunking identity. Replay
returns the original job and original resolved seed. Reuse with different inputs
returns HTTP 409. Requests without a key create independent jobs.

Python, NumPy, torch CPU and all CUDA RNGs are seeded for each provider call.
Chunk seed i is derived from SHA-256 of
qwen3-tts-chunk-v1:<base-seed>:<i>. Retry/recovery therefore reconstructs the
same seed sequence. CUDA kernels may still prevent bit-for-bit equivalence; the
service promises traceable reproducibility metadata, not impossible guarantees.
There are no hidden synthesis retries.

### Physical chunks and immutable regeneration

Completed narration jobs expose their original physical chunks through
`GET /v1/narrations/{job_id}/chunks` and
`GET /v1/narrations/{job_id}/chunks/{chunk_index}/audio`. The list includes the frozen character
span, boundary, chunking identity, profile identities, synthesis text, seed and WAV identity. It
also returns `pause_after_frames`, the exact resolved silence used after that chunk (zero for the
final chunk). Older completed jobs recover it from their immutable assembly timeline rather than
consulting current pause defaults.

`POST /v1/narrations/{job_id}/chunks/{chunk_index}/regenerate` requires an explicit seed and an
`Idempotency-Key`. It synthesizes the stored `synthesis_text` through the same bounded inference
worker and publishes a new UUID-addressed WAV. It never writes the original `chunk_NNNN.wav`.
The response's `candidate_id` is retrieved exactly through
`GET /v1/narrations/{job_id}/chunk-candidates/{candidate_id}/audio`; there is no query-parameter
take alias. Replaying the same key and inputs returns the same candidate, while changed inputs
return HTTP 409. Qwen owns only artifact generation and integrity—promotion, rejection and the
current-take pointer remain Orchestrator decisions.

## Long text and audio assembly

The character-bounded heuristic preserves every original character and order. Production defaults
are now target 250 / maximum 400 characters (previously 500 / 800), reducing the amount of
autoregressive context in one inference call while retaining sentence-aware grouping. The
`qwen3-tts-chunking-v2` identity includes both effective limits, so changing either cannot reuse a
job created under different waveform semantics. It
prefers paragraph boundaries, then sentence boundaries, then whitespace fallback,
with hard cuts only as a last resort. Decimal points such as 3.14 are not treated
as sentence endings. Character limits are safeguards, not token-count claims.

Chunks are independent utterances and are never crossfaded. Assembly inserts a
configured deterministic silence based on paragraph, sentence, fallback or hard
split boundary. Qwen's native sample rate/channels must remain stable across
chunks. No legacy 22050 Hz resampling exists.

Every provider result is validated before assembly. The explainable v1 guard requires finite,
non-silent in-range audio at a valid sample rate and rejects a chunk above either the absolute
90-second ceiling or the conservative `10 + 0.35 * text_length_chars` ceiling. These thresholds
are deliberately loose enough for slow Portuguese narration; they target obvious runaways, not
stylistic speaking-rate differences. Rejection fails the job as `chunk_quality_guard`, publishes
no WAV and records safe chunk index/length/duration/seed/reason diagnostics. Qwen never retries a
rejected chunk internally; the Orchestrator owns explicit retry policy.

After each chunk passes the existing guard, it is appended to a temporary native-rate PCM16 WAV
together with its deterministic pause. Only the current quality-bounded chunk is resident; the
complete narration is never concatenated in memory. The temporary WAV is validated and atomically
renamed after the final chunk, and failures remove only the temporary file. Validation
checks shape, non-empty/finite samples, sample rate, PCM range, duration, reopened
WAV format, strict RIFF/data sizes, frame count and obvious clipping. SHA-256, size, duration,
channels, subtype, frame count and sample peak are persisted. Atomic rename is followed by another
reopen before completed state. HTTP publication reconciles those values, emits a SHA ETag, supports
HEAD/byte ranges and refuses replaced or truncated output. Partial WAVs are never served. Job
metadata attributes detected silence inside each generated chunk separately from configured
chunk-boundary pauses and includes lightweight transient candidates.

This is validation, not final mastering. The service does not apply LUFS gain,
limiting, compression, EQ, denoising or enhancement. Final 48 kHz conversion and
video-level integrated loudness mastering are intentionally deferred to the
Orchestrator migration stage, after complete narration assembly.

The pinned qwen-tts 0.1.1 implementation forwards `max_new_tokens` to the talker generation loop,
where each generated step contributes one codec frame until the codec EOS token or the configured
ceiling. The pinned model declares the 12 Hz tokenizer and 24 kHz output. Thus a generation near
2048 steps is consistent with roughly the observed multi-minute ceiling, but the production audio
does not prove why EOS was missed. `narration_v1` and its 2048-token baseline remain unchanged for
reproducibility; `narration_v2` changes only the frozen sampling controls documented in
[`docs/production_voice_baseline.md`](docs/production_voice_baseline.md). The duration guard is the
mandatory safety fuse at the waveform boundary.

## Recovery and retention

At startup, queued jobs are re-enqueued. A running job with a valid atomically
published final WAV is reconciled to completed. Otherwise its invalid/missing
output is removed, it returns to queued and reruns with the same persisted model,
profiles, base seed and chunk-seed algorithm. Terminal jobs older than
JOB_RETENTION_HOURS and their outputs are removed at startup.

The queue is memory-bounded and rejects overload with HTTP 503. SQLite persists
all accepted work; the queue is not the durable authority. Graceful shutdown
drains already accepted jobs before stopping the worker.

## Tests

Normal tests use FakeTTSProvider; they never import Qwen weights, initialize
CUDA, contact Hugging Face or require internet.

~~~bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/ruff check runtime tests scripts
.venv/bin/mypy
python -m compileall -q runtime tests scripts
~~~

## Benchmarks and GPU validation

With the service running:

~~~bash
PYTHONPATH=runtime python scripts/benchmark_generation.py
PYTHONPATH=runtime python scripts/benchmark_generation.py \
  --generations narration_v1 --seeds 101 202 303 --run
PYTHONPATH=runtime python scripts/benchmark_generation.py \
  --languages Portuguese Auto --seeds 101 202 303 --run
PYTHONPATH=runtime python scripts/benchmark_references.py \
  --voices bernardo_v1 bernardo_xvec_v1 --seeds 101 202 303 \
  --text 'Texto fixo para comparar referências.' --run
PYTHONPATH=runtime python scripts/soak_test.py --count 50 --run
~~~

Every generation/reference/soak command is a strict plan unless `--run` is present; plan mode makes
no HTTP request and writes no output. Generation/reference manifests record input identity, profiles, model/revision,
seed, language, latency, duration, RTF, sample rate, output/checksum, chunking identity,
words-per-minute, characters-per-second, per-chunk duration/timing/guard status and basic metrics. The built-in corpus includes a synthetic
long Portuguese regression similar in size and structure to the observed failure. The
soak run records sequential success, latency/RTF and process memory.
The complete seven-category ICL/x-vector comparison, including Orchestrator pronunciation,
Faster-Whisper coverage, pause/transient diagnostics and an empty human-review rubric, is the
explicit opt-in `backend/scripts/benchmark_clone_modes.py` command documented in the
Orchestrator's `backend/docs/NARRATION_HARDENING.md`. Whisper remains outside this provider's
runtime dependency set and no benchmark chooses a winning voice automatically.

The direct real-model smoke is opt-in and never part of pytest:

~~~bash
RUN_REAL_QWEN_SMOKE=1 PYTHONPATH=runtime python scripts/smoke_real_qwen.py
~~~

It requires the real pinned model, CUDA and configured references, writes a small
Portuguese WAV, validates it and reports latency, duration, RTF, sample rate,
seed, model/revision, peak and checksum.

## Security and limitations

This is a trusted-network service. Profile names come only from configured
registries; text and queue sizes are bounded; output IDs cannot traverse paths;
public errors hide internals. Before public exposure it still requires TLS,
authentication and infrastructure rate limiting.

Compose publishes port 8889 on the host so the separately deployed Orchestrator
can reach it through `host.docker.internal`. Restrict that port with the host
firewall; the service does not provide public-network authentication.

Known limitations are single-instance/single-GPU operation, no distributed lock,
no cancellation API, startup-only profile reload, no perceptual quality metric,
and no final mastering (the Orchestrator owns whole-narration mastering).
