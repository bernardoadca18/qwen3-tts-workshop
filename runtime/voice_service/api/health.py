from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, object]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request) -> dict[str, object]:
    runtime = request.app.state.runtime
    if not runtime.ready or runtime.service is None or not runtime.service.worker_operational:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": runtime.initialization_error or "initializing",
            },
        )
    assert runtime.provider is not None
    assert runtime.voices is not None
    assert runtime.generations is not None
    metadata = runtime.provider.metadata()
    default_voice = runtime.voices.get(runtime.settings.default_voice_profile)
    default_generation = runtime.generations.get(runtime.settings.default_generation_profile)
    return {
        "status": "ready",
        "provider": metadata.provider,
        "model_id": metadata.model_id,
        "model_revision": metadata.model_revision,
        "device": metadata.device,
        "dtype": metadata.dtype,
        "attention_impl": metadata.attention_impl,
        "cuda_available": metadata.cuda_available,
        "queue_depth": runtime.service.queue_depth,
        "queue_capacity": runtime.settings.max_queue_size,
        "worker_operational": runtime.service.worker_operational,
        "startup_metrics": runtime.startup_metrics,
        "prompt_cache": metadata.prompt_cache,
        "chunking_strategy_version": runtime.service.chunker.strategy_version,
        "chunking_identity": runtime.service.chunker.identity,
        "chunking": {
            "target_chars": runtime.service.chunker.target_chars,
            "max_chars": runtime.service.chunker.max_chars,
        },
        "default_voice_profile": {
            "name": default_voice.name,
            "version": default_voice.version,
            "identity": default_voice.identity,
            "clone_mode": default_voice.clone_mode,
            "reference_id": default_voice.reference_id,
            "reference_sha256": default_voice.audio_sha256,
        },
        "default_generation_profile": {
            "name": default_generation.name,
            "version": default_generation.version,
            "identity": default_generation.identity,
        },
    }


@router.get("/health")
async def compatibility_health(request: Request) -> dict[str, object]:
    return await ready(request)
