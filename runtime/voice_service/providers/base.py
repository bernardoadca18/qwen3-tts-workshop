from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from voice_service.domain import AudioChunk, GenerationProfile, VoiceProfile


class ProviderError(RuntimeError):
    pass


class ProviderOutOfMemoryError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    model_id: str
    model_revision: str
    device: str
    dtype: str
    attention_impl: str
    initialized: bool
    cuda_available: bool | None
    prompt_cache: dict[str, Any] | None = None


class TTSProvider(Protocol):
    async def initialize(self) -> None: ...

    async def prepare_voice(self, profile: VoiceProfile) -> None: ...

    async def generate(
        self,
        *,
        text: str,
        language: str,
        voice_profile: VoiceProfile,
        generation_profile: GenerationProfile,
        seed: int,
    ) -> AudioChunk: ...

    def metadata(self) -> ProviderMetadata: ...

    async def shutdown(self) -> None: ...
