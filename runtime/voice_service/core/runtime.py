from __future__ import annotations

import logging
import time
from collections.abc import Callable

from voice_service.core.config import Settings
from voice_service.providers import QwenTTSProvider, TTSProvider
from voice_service.repositories import NarrationJobRepository
from voice_service.services.audio import AtomicWavPublisher
from voice_service.services.narration import NarrationService
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    VoiceProfileRegistry,
)

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[Settings], TTSProvider]


class RuntimeState:
    def __init__(
        self,
        settings: Settings,
        provider_factory: ProviderFactory = QwenTTSProvider,
    ):
        self.settings = settings
        self.provider_factory = provider_factory
        self.provider: TTSProvider | None = None
        self.service: NarrationService | None = None
        self.voices: VoiceProfileRegistry | None = None
        self.generations: GenerationProfileRegistry | None = None
        self.ready = False
        self.initialization_error: str | None = None

    async def initialize(self) -> None:
        started = time.monotonic()
        try:
            repository = NarrationJobRepository(self.settings.job_database_path)
            repository.initialize()
            voices = VoiceProfileRegistry.load(
                self.settings.voice_profiles_file,
                model_id=self.settings.model_id,
                model_revision=self.settings.model_revision,
            )
            generations = GenerationProfileRegistry.load(self.settings.generation_profiles_file)
            voices.get(self.settings.default_voice_profile)
            generations.get(self.settings.default_generation_profile)
            provider = self.provider_factory(self.settings)
            self.provider = provider
            await provider.initialize()
            for profile in voices.all():
                await provider.prepare_voice(profile)
            if self.settings.warmup_enabled:
                await provider.generate(
                    text=self.settings.warmup_text,
                    language="Portuguese",
                    voice_profile=voices.get(self.settings.default_voice_profile),
                    generation_profile=generations.get(self.settings.default_generation_profile),
                    seed=1,
                )
            service = NarrationService(
                settings=self.settings,
                provider=provider,
                repository=repository,
                voices=voices,
                generations=generations,
                publisher=AtomicWavPublisher(
                    self.settings.output_directory,
                    max_duration_seconds=self.settings.max_result_duration_seconds,
                ),
            )
            self.service = service
            await service.start()
            self.voices = voices
            self.generations = generations
            self.ready = True
            logger.info(
                "voice service ready model=%s revision=%s device=%s dtype=%s "
                "attention=%s voices=%s initialization_seconds=%.3f",
                self.settings.model_id,
                self.settings.model_revision,
                self.settings.device,
                self.settings.dtype,
                self.settings.attention_impl,
                [profile.name for profile in voices.all()],
                time.monotonic() - started,
            )
        except Exception:
            self.initialization_error = "initialization_failed"
            logger.exception("voice service initialization failed")
            raise

    async def shutdown(self) -> None:
        self.ready = False
        if self.service is not None:
            await self.service.stop()
        if self.provider is not None:
            await self.provider.shutdown()

    def require_service(self) -> NarrationService:
        if not self.ready or self.service is None:
            raise RuntimeError("voice service is not ready")
        return self.service
