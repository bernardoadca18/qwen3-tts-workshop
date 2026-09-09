from __future__ import annotations

from typing import Annotated, cast

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from voice_service.api.schemas import (
    BenchmarkNarrationCreate,
    ChunkArtifactOut,
    ChunkRegenerateOut,
    ChunkRegenerateRequest,
    GenerationProfileOut,
    LegacyGenerateRequest,
    NarrationCreate,
    NarrationOut,
    SpeakerEmbeddingOut,
    VoiceOut,
)
from voice_service.domain import JobStatus
from voice_service.services.audio import AudioValidationError
from voice_service.services.narration import (
    ChunkRegenerationFailedError,
    IdempotencyConflictError,
    NarrationClientError,
    NarrationQueueFullError,
    NarrationService,
)

router = APIRouter()


def _service(request: Request) -> NarrationService:
    try:
        return cast(NarrationService, request.app.state.runtime.require_service())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="voice service is not ready") from exc


@router.post(
    "/v1/narrations",
    response_model=NarrationOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["narrations"],
)
async def create_narration(
    payload: NarrationCreate,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> NarrationOut:
    if idempotency_key is not None and not (1 <= len(idempotency_key) <= 200):
        raise HTTPException(status_code=400, detail="invalid Idempotency-Key")
    try:
        job, created = await _service(request).create(
            text=payload.text,
            language=payload.language,
            voice_profile_name=payload.voice_profile,
            generation_profile_name=payload.generation_profile,
            requested_seed=payload.seed,
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NarrationClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NarrationQueueFullError as exc:
        raise HTTPException(status_code=503, detail="narration queue is full") from exc
    return NarrationOut.from_job(job, created=created)


@router.post(
    "/v1/benchmark/narrations",
    response_model=NarrationOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["benchmark"],
)
async def create_benchmark_narration(
    payload: BenchmarkNarrationCreate,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> NarrationOut:
    if idempotency_key is not None and not (1 <= len(idempotency_key) <= 200):
        raise HTTPException(status_code=400, detail="invalid Idempotency-Key")
    try:
        job, created = await _service(request).create_benchmark(
            text=payload.text,
            language=payload.language,
            reference_id=payload.reference_id,
            generation_profile_name=payload.generation_profile,
            requested_seed=payload.seed,
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NarrationClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NarrationQueueFullError as exc:
        raise HTTPException(status_code=503, detail="narration queue is full") from exc
    return NarrationOut.from_job(job, created=created)


@router.get("/v1/narrations/{job_id}", response_model=NarrationOut, tags=["narrations"])
async def get_narration(job_id: str, request: Request) -> NarrationOut:
    try:
        return NarrationOut.from_job(_service(request).get(job_id))
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="narration job not found") from exc


@router.api_route("/v1/narrations/{job_id}/audio", methods=["GET", "HEAD"], tags=["narrations"])
async def get_narration_audio(job_id: str, request: Request) -> FileResponse:
    service = _service(request)
    try:
        job = service.get(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="narration job not found") from exc
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="narration audio is not available")
    try:
        path, metrics = service.verified_audio(job.id)
    except (AudioValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="narration audio failed integrity checks"
        ) from exc
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{job.id}.wav",
        content_disposition_type="inline",
        stat_result=path.stat(),
        headers={
            "ETag": f'"{metrics.sha256}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Audio-SHA256": metrics.sha256,
            "X-Audio-Duration-Seconds": f"{metrics.duration_seconds:.9f}",
            "X-Audio-Frame-Count": str(metrics.frame_count),
        },
    )


@router.get("/v1/voices", response_model=list[VoiceOut], tags=["voices"])
async def list_voices(request: Request) -> list[VoiceOut]:
    runtime = request.app.state.runtime
    _service(request)
    assert runtime.voices is not None
    return [VoiceOut.from_profile(profile) for profile in runtime.voices.all()]


@router.get(
    "/v1/generation-profiles",
    response_model=list[GenerationProfileOut],
    tags=["generation-profiles"],
)
async def list_generation_profiles(request: Request) -> list[GenerationProfileOut]:
    runtime = request.app.state.runtime
    _service(request)
    assert runtime.generations is not None
    return [GenerationProfileOut.from_profile(profile) for profile in runtime.generations.all()]


@router.post("/generate", deprecated=True, tags=["legacy"])
async def legacy_generate(payload: LegacyGenerateRequest, request: Request) -> FileResponse:
    service = _service(request)
    profile = service.generations.get(service.settings.default_generation_profile)
    legacy_values = {
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "top_k": payload.top_k,
        "repetition_penalty": payload.repetition_penalty,
        "max_new_tokens": payload.max_new_tokens,
    }
    if any(profile.parameters[name] != value for name, value in legacy_values.items()):
        raise HTTPException(
            status_code=422,
            detail="raw Qwen overrides are disabled; use a named generation profile",
        )
    try:
        job, _ = await service.create(
            text=payload.text,
            language=payload.language,
            voice_profile_name=None,
            generation_profile_name=None,
            requested_seed=payload.seed,
            idempotency_key=None,
        )
        completed = await service.wait_terminal(
            job.id, timeout=service.settings.legacy_generate_timeout_seconds
        )
    except NarrationClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NarrationQueueFullError as exc:
        raise HTTPException(status_code=503, detail="narration queue is full") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="legacy generation timed out") from exc
    if completed.status is not JobStatus.COMPLETED:
        raise HTTPException(status_code=503, detail="narration generation failed")
    path = service.publisher.path_for(completed.id)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{completed.id}.wav",
        headers={
            "X-Qwen-Seed": str(completed.resolved_seed),
            "Deprecation": "true",
            "Link": '</v1/narrations>; rel="successor-version"',
        },
    )


@router.post(
    "/v1/speaker-embeddings",
    response_model=SpeakerEmbeddingOut,
    status_code=status.HTTP_200_OK,
    tags=["speaker-embeddings"],
)
async def extract_speaker_embedding(
    request: Request,
    audio: Annotated[
        UploadFile,
        File(description="Audio file (WAV) to extract speaker embedding from"),
    ],
) -> SpeakerEmbeddingOut:
    """Extract speaker embedding from uploaded audio using Qwen's ECAPA-TDNN encoder.

    The audio is automatically resampled to 24kHz if needed.
    Returns a 2048-dimensional speaker embedding vector.
    """
    service = _service(request)

    # Read uploaded audio
    audio_bytes = await audio.read()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        # Load audio
        wav, sr = sf.read(tmp.name)

        # Convert to mono if stereo
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)

        # Ensure float32
        wav = wav.astype(np.float32)

    # Extract embedding via provider
    try:
        embedding = service.provider.extract_speaker_embedding(wav, sr)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"speaker embedding extraction failed: {exc}",
        ) from exc

    return SpeakerEmbeddingOut.from_numpy(embedding, sr)


@router.get(
    "/v1/narrations/{job_id}/chunks",
    response_model=list[ChunkArtifactOut],
    tags=["narrations"],
)
async def list_job_chunks(
    job_id: str,
    request: Request,
) -> list[ChunkArtifactOut]:
    """List all persisted chunk artifacts for a narration job."""
    service = _service(request)
    try:
        job = service.get(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="job not completed")

    chunks_metadata = job.metadata.get("chunks", [])
    if not chunks_metadata:
        return []

    chunks_out = []
    assembly_timeline = job.metadata.get("assembly_timeline", [])
    for chunk_meta in chunks_metadata:
        chunk_index = chunk_meta["index"]
        chunk_id = f"{job_id}_{chunk_index:04d}"
        try:
            _, metrics = service.verified_chunk(job_id, chunk_index)
        except AudioValidationError:
            continue

        pause_after_frames = chunk_meta.get("pause_after_frames")
        if not isinstance(pause_after_frames, int) or pause_after_frames < 0:
            # Jobs completed before pause_after_frames was persisted still carry
            # the immutable assembly timeline. Recover the exact historical frame
            # count from that timeline instead of consulting today's defaults.
            pause_after_frames = 0
            for item in assembly_timeline if isinstance(assembly_timeline, list) else []:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") != "inserted_pause" or item.get("chunk_index") != chunk_index:
                    continue
                try:
                    start = float(item["start_seconds"])
                    end = float(item["end_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                pause_after_frames = max(0, round((end - start) * metrics.sample_rate))
                break

        chunks_out.append(
            ChunkArtifactOut(
                chunk_id=chunk_id,
                job_id=job_id,
                chunk_index=chunk_index,
                text=chunk_meta.get("text", ""),
                synthesis_text=chunk_meta.get("synthesis_text", ""),
                seed=chunk_meta["seed"],
                duration_seconds=metrics.duration_seconds,
                sample_rate=metrics.sample_rate,
                byte_size=metrics.byte_size,
                sha256=metrics.sha256,
                frame_count=metrics.frame_count,
                start_char=chunk_meta["start_char"],
                end_char=chunk_meta["end_char"],
                boundary=chunk_meta["boundary"],
                pause_after_frames=pause_after_frames,
                chunking_identity=job.chunking_identity,
                chunking_strategy_version=job.metadata["chunking_strategy_version"],
                voice_identity=job.voice_identity,
                generation_identity=job.generation_identity,
                created_at=job.completed_at or job.created_at,
                audio_url=f"/v1/narrations/{job_id}/chunks/{chunk_index}/audio",
                speaker_similarity=(
                    chunk_meta.get("speaker_identity", {}).get("similarity")
                    if isinstance(chunk_meta.get("speaker_identity"), dict)
                    else None
                ),
                speaker_similarity_reference_identity=(
                    chunk_meta.get("speaker_identity", {}).get("reference_identity")
                    if isinstance(chunk_meta.get("speaker_identity"), dict)
                    else None
                ),
                speaker_similarity_policy_version=(
                    chunk_meta.get("speaker_identity", {}).get("policy_version")
                    if isinstance(chunk_meta.get("speaker_identity"), dict)
                    else None
                ),
                speaker_identity_status=(
                    chunk_meta.get("speaker_identity", {}).get("status")
                    if isinstance(chunk_meta.get("speaker_identity"), dict)
                    else None
                ),
                speaker_embedding_seconds=(
                    chunk_meta.get("speaker_identity", {}).get("embedding_seconds")
                    if isinstance(chunk_meta.get("speaker_identity"), dict)
                    else None
                ),
            )
        )

    return chunks_out


@router.get(
    "/v1/narrations/{job_id}/chunks/{chunk_index}/audio",
    response_class=FileResponse,
    tags=["narrations"],
)
async def get_chunk_audio(
    job_id: str,
    chunk_index: int,
    request: Request,
) -> FileResponse:
    """Download a specific chunk's WAV audio file."""
    service = _service(request)
    try:
        job = service.get(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="job not completed")

    try:
        chunk_path, metrics = service.verified_chunk(job_id, chunk_index)
    except NarrationClientError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=409, detail="chunk failed integrity checks") from exc

    return FileResponse(
        path=chunk_path,
        media_type="audio/wav",
        filename=f"{job_id}_chunk_{chunk_index:04d}.wav",
        content_disposition_type="inline",
        stat_result=chunk_path.stat(),
        headers={
            "ETag": f'"{metrics.sha256}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Audio-SHA256": metrics.sha256,
        },
    )


@router.api_route(
    "/v1/narrations/{job_id}/chunk-candidates/{candidate_id}/audio",
    methods=["GET", "HEAD"],
    tags=["narrations"],
)
async def get_chunk_candidate_audio(
    job_id: str,
    candidate_id: str,
    request: Request,
) -> FileResponse:
    """Download exactly one immutable regeneration artifact by stable identity."""
    service = _service(request)
    try:
        path, metrics = service.verified_chunk_candidate(job_id, candidate_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="chunk candidate not found") from exc
    except (AudioValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="chunk candidate failed integrity checks"
        ) from exc
    return FileResponse(
        path=path,
        media_type="audio/wav",
        filename=f"{candidate_id}.wav",
        content_disposition_type="inline",
        stat_result=path.stat(),
        headers={
            "ETag": f'"{metrics.sha256}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Audio-SHA256": metrics.sha256,
        },
    )


@router.post(
    "/v1/narrations/{job_id}/chunks/{chunk_index}/regenerate",
    response_model=ChunkRegenerateOut,
    status_code=status.HTTP_200_OK,
    tags=["narrations"],
)
async def regenerate_chunk(
    job_id: str,
    chunk_index: int,
    payload: ChunkRegenerateRequest,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ChunkRegenerateOut:
    """Regenerate a single chunk with a new seed.

    This endpoint creates a new candidate take for the specified chunk using the
    frozen synthesis text snapshot and the provided seed. The candidate is NOT
    automatically promoted - the caller must decide whether to use it.
    """
    service = _service(request)
    if idempotency_key is None or not (1 <= len(idempotency_key) <= 200):
        raise HTTPException(
            status_code=400,
            detail="valid Idempotency-Key is required for chunk regeneration",
        )
    try:
        candidate, created = await service.create_chunk_candidate(
            job_id=job_id,
            chunk_index=chunk_index,
            seed=payload.seed,
            idempotency_key=idempotency_key,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NarrationClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NarrationQueueFullError as exc:
        raise HTTPException(status_code=503, detail="narration queue is full") from exc
    except ChunkRegenerationFailedError as exc:
        raise HTTPException(
            status_code=500,
            detail=exc.candidate.error_message or "chunk regeneration failed",
        ) from exc
    return ChunkRegenerateOut.from_candidate(candidate, created=created)
