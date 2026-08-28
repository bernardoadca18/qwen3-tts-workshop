from __future__ import annotations

import asyncio
import logging
import random
import threading
from typing import Any

import numpy as np

from voice_service.core.config import Settings
from voice_service.domain import AudioChunk, GenerationProfile, VoiceProfile
from voice_service.providers.base import (
    ProviderError,
    ProviderMetadata,
    ProviderOutOfMemoryError,
)

logger = logging.getLogger(__name__)


class QwenTTSProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._torch: Any | None = None
        self._voice_prompts: dict[str, Any] = {}
        self._inference_lock = threading.Lock()
        self._initialized = False
        self._cuda_available: bool | None = None
        self._snapshot_path: str | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        if self._initialized:
            return
        import torch
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        snapshot_path = snapshot_download(
            repo_id=self.settings.model_id,
            revision=self.settings.model_revision,
        )
        logger.info(
            "initializing Qwen provider model=%s revision=%s device=%s dtype=%s attention=%s",
            self.settings.model_id,
            self.settings.model_revision,
            self.settings.device,
            self.settings.dtype,
            self.settings.attention_impl,
        )
        self._model = Qwen3TTSModel.from_pretrained(
            snapshot_path,
            device_map=self.settings.device,
            dtype=dtype_map[self.settings.dtype],
            attn_implementation=self.settings.attention_impl,
        )
        self._torch = torch
        self._cuda_available = bool(torch.cuda.is_available())
        self._snapshot_path = snapshot_path
        self._initialized = True

    async def prepare_voice(self, profile: VoiceProfile) -> None:
        await asyncio.to_thread(self._prepare_voice_sync, profile)

    def _prepare_voice_sync(self, profile: VoiceProfile) -> None:
        self._require_initialized()
        if profile.identity in self._voice_prompts:
            return
        assert self._model is not None
        x_vector_only = profile.clone_mode == "x_vector_only"
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(profile.reference_audio),
            ref_text=None if x_vector_only else profile.transcript,
            x_vector_only_mode=x_vector_only,
        )
        self._voice_prompts[profile.identity] = prompt
        logger.info(
            "voice prompt cached profile=%s identity=%s clone_mode=%s audio_sha256=%s",
            profile.name,
            profile.identity,
            profile.clone_mode,
            profile.audio_sha256,
        )

    async def generate(
        self,
        *,
        text: str,
        language: str,
        voice_profile: VoiceProfile,
        generation_profile: GenerationProfile,
        seed: int,
    ) -> AudioChunk:
        return await asyncio.to_thread(
            self._generate_sync,
            text,
            language,
            voice_profile,
            generation_profile,
            seed,
        )

    def _generate_sync(
        self,
        text: str,
        language: str,
        voice_profile: VoiceProfile,
        generation_profile: GenerationProfile,
        seed: int,
    ) -> AudioChunk:
        self._require_initialized()
        prompt = self._voice_prompts.get(voice_profile.identity)
        if prompt is None:
            raise ProviderError("voice prompt has not been prepared")
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch
        with self._inference_lock:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            try:
                with torch.inference_mode():
                    waveforms, sample_rate = self._model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=prompt,
                        **generation_profile.parameters,
                    )
            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ProviderOutOfMemoryError("CUDA out of memory") from exc
            except Exception as exc:
                raise ProviderError("Qwen synthesis failed") from exc
        if not waveforms:
            raise ProviderError("Qwen returned no waveform")
        samples = np.asarray(waveforms[0], dtype=np.float32)
        return AudioChunk(samples=samples, sample_rate=int(sample_rate))

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider="qwen3-tts",
            model_id=self.settings.model_id,
            model_revision=self.settings.model_revision,
            device=self.settings.device,
            dtype=self.settings.dtype,
            attention_impl=self.settings.attention_impl,
            initialized=self._initialized,
            cuda_available=self._cuda_available,
        )

    async def shutdown(self) -> None:
        self._voice_prompts.clear()
        self._model = None
        self._torch = None
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._model is None:
            raise ProviderError("provider is not initialized")
