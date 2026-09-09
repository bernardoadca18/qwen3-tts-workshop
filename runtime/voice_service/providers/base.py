from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

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

    def speaker_similarity(
        self, audio: AudioChunk, voice_profile: VoiceProfile
    ) -> float: ...

    def extract_speaker_embedding(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]: ...

    def metadata(self) -> ProviderMetadata: ...

    async def shutdown(self) -> None: ...
