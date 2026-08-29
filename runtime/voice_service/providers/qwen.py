from __future__ import annotations

import asyncio
import logging
import random
import threading
from collections import OrderedDict
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
        # Registered profiles are deliberately pinned for the service lifetime.
        self._voice_prompts: dict[str, Any] = {}
        self._ephemeral_voice_prompts: OrderedDict[str, Any] = OrderedDict()
        self._active_prompt_references: dict[str, int] = {}
        self._prompt_cache_lock = threading.Lock()
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0
        self._prompt_cache_evictions = 0
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
        with self._prompt_cache_lock:
            if profile.identity in self._voice_prompts:
                self._prompt_cache_hits += 1
                return
            if profile.identity in self._ephemeral_voice_prompts:
                self._ephemeral_voice_prompts.move_to_end(profile.identity)
                self._prompt_cache_hits += 1
                return
            self._prompt_cache_misses += 1
            assert self._model is not None
            x_vector_only = profile.clone_mode == "x_vector_only"
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=str(profile.reference_audio),
                ref_text=None if x_vector_only else profile.transcript,
                x_vector_only_mode=x_vector_only,
            )
            if profile.ephemeral:
                self._ephemeral_voice_prompts[profile.identity] = prompt
                self._evict_ephemeral_prompts_locked(protected_identity=profile.identity)
            else:
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
        prompt = self._acquire_prompt(voice_profile.identity)
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch
        try:
            with self._inference_lock:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
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
        finally:
            self._release_prompt(voice_profile.identity)
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
            prompt_cache=self.prompt_cache_diagnostics(),
        )

    def prompt_cache_diagnostics(self) -> dict[str, object]:
        with self._prompt_cache_lock:
            return {
                "policy": "registered_pinned_ephemeral_lru_v1",
                "entries": len(self._voice_prompts) + len(self._ephemeral_voice_prompts),
                "pinned_entries": len(self._voice_prompts),
                "registered_entries": len(self._voice_prompts),
                "ephemeral_entries": len(self._ephemeral_voice_prompts),
                "maximum_ephemeral_entries": (
                    self.settings.max_ephemeral_prompt_cache_entries
                ),
                "ephemeral_limit": self.settings.max_ephemeral_prompt_cache_entries,
                "active_entries": sum(
                    1 for count in self._active_prompt_references.values() if count > 0
                ),
                "hits": self._prompt_cache_hits,
                "misses": self._prompt_cache_misses,
                "evictions": self._prompt_cache_evictions,
                "estimated_bytes": None,
                "memory_measurement": "unavailable_without_tensor_introspection",
            }

    def _acquire_prompt(self, identity: str) -> Any:
        with self._prompt_cache_lock:
            prompt = self._voice_prompts.get(identity)
            if prompt is None:
                prompt = self._ephemeral_voice_prompts.get(identity)
                if prompt is not None:
                    self._ephemeral_voice_prompts.move_to_end(identity)
            if prompt is None:
                raise ProviderError("voice prompt has not been prepared")
            self._active_prompt_references[identity] = (
                self._active_prompt_references.get(identity, 0) + 1
            )
            return prompt

    def _release_prompt(self, identity: str) -> None:
        with self._prompt_cache_lock:
            count = self._active_prompt_references.get(identity, 0)
            if count <= 1:
                self._active_prompt_references.pop(identity, None)
            else:
                self._active_prompt_references[identity] = count - 1
            self._evict_ephemeral_prompts_locked()

    def _evict_ephemeral_prompts_locked(self, *, protected_identity: str | None = None) -> None:
        limit = self.settings.max_ephemeral_prompt_cache_entries
        while len(self._ephemeral_voice_prompts) > limit:
            evictable = next(
                (
                    identity
                    for identity in self._ephemeral_voice_prompts
                    if identity != protected_identity
                    and self._active_prompt_references.get(identity, 0) == 0
                ),
                None,
            )
            if evictable is None:
                return
            del self._ephemeral_voice_prompts[evictable]
            self._prompt_cache_evictions += 1

    async def shutdown(self) -> None:
        with self._prompt_cache_lock:
            self._voice_prompts.clear()
            self._ephemeral_voice_prompts.clear()
            self._active_prompt_references.clear()
        self._model = None
        self._torch = None
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._model is None:
            raise ProviderError("provider is not initialized")
