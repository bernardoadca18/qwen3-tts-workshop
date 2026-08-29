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
    metadata = runtime.provider.metadata()
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
        "prompt_cache": metadata.prompt_cache,
        "chunking_strategy_version": runtime.service.chunker.strategy_version,
        "chunking_identity": runtime.service.chunker.identity,
        "chunking": {
            "target_chars": runtime.service.chunker.target_chars,
            "max_chars": runtime.service.chunker.max_chars,
        },
    }


@router.get("/health")
async def compatibility_health(request: Request) -> dict[str, object]:
    return await ready(request)
