from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from voice_service.core.config import Settings
from voice_service.domain import AudioChunk, GenerationProfile, VoiceProfile
from voice_service.providers import ProviderError, ProviderMetadata


class FakeTTSProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.initialized = False
        self.prepared: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.maximum_active = 0
        self.delay = 0.0
        self.fail = False
        self.fail_initialize = False
        self.shutdown_called = False
        self.invalid_samples = False

    async def initialize(self) -> None:
        self.initialized = True
        if self.fail_initialize:
            raise ProviderError("fake initialization failure")

    async def prepare_voice(self, profile: VoiceProfile) -> None:
        if profile.identity not in self.prepared:
            self.prepared.append(profile.identity)

    async def generate(
        self,
        *,
        text: str,
        language: str,
        voice_profile: VoiceProfile,
        generation_profile: GenerationProfile,
        seed: int,
    ) -> AudioChunk:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise ProviderError("fake internal provider detail")
            self.calls.append(
                {
                    "text": text,
                    "language": language,
                    "voice": voice_profile.name,
                    "generation": generation_profile.name,
                    "seed": seed,
                }
            )
            if self.invalid_samples:
                samples = np.array([0.1, np.nan], dtype=np.float32)
            else:
                frames = max(240, len(text) * 30)
                phase = np.arange(frames, dtype=np.float32)
                samples = 0.2 * np.sin(2 * np.pi * 220 * phase / 24000)
            return AudioChunk(samples=samples, sample_rate=24000)
        finally:
            self.active -= 1

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider="fake",
            model_id=self.settings.model_id,
            model_revision=self.settings.model_revision,
            device=self.settings.device,
            dtype=self.settings.dtype,
            attention_impl=self.settings.attention_impl,
            initialized=self.initialized,
            cuda_available=False,
        )

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.initialized = False
