from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from voice_service.api.health import router as health_router
from voice_service.api.narrations import router as narration_router
from voice_service.core.config import Settings
from voice_service.core.logging import configure_logging
from voice_service.core.runtime import ProviderFactory, RuntimeState
from voice_service.providers import QwenTTSProvider


def create_app(
    *,
    settings: Settings | None = None,
    provider_factory: ProviderFactory = QwenTTSProvider,
) -> FastAPI:
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    runtime = RuntimeState(resolved_settings, provider_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async def initialize_background() -> None:
            try:
                await runtime.initialize()
            except Exception:
                return

        initialization_task = asyncio.create_task(
            initialize_background(), name="qwen-tts-initialization"
        )
        await asyncio.sleep(0)
        try:
            yield
        finally:
            try:
                await initialization_task
            finally:
                await runtime.shutdown()

    application = FastAPI(
        title="NeoVids Voice Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.runtime = runtime
    application.include_router(health_router)
    application.include_router(narration_router)
    return application


app = create_app()
