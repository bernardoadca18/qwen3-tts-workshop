from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import FileResponse

from voice_service.api.schemas import (
    GenerationProfileOut,
    LegacyGenerateRequest,
    NarrationCreate,
    NarrationOut,
    VoiceOut,
)
from voice_service.domain import JobStatus
from voice_service.services.narration import (
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


@router.get("/v1/narrations/{job_id}", response_model=NarrationOut, tags=["narrations"])
async def get_narration(job_id: str, request: Request) -> NarrationOut:
    try:
        return NarrationOut.from_job(_service(request).get(job_id))
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="narration job not found") from exc


@router.get("/v1/narrations/{job_id}/audio", tags=["narrations"])
async def get_narration_audio(job_id: str, request: Request) -> FileResponse:
    service = _service(request)
    try:
        job = service.get(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="narration job not found") from exc
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="narration audio is not available")
    try:
        path = service.publisher.path_for(job.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="narration audio is not available") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="narration audio is not available")
    return FileResponse(path, media_type="audio/wav", filename=f"{job.id}.wav")


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
