#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from voice_service.core.config import Settings
from voice_service.providers import QwenTTSProvider
from voice_service.services.audio import AtomicWavPublisher
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    VoiceProfileRegistry,
)
from voice_service.services.text_chunker import derive_chunk_seed


async def run() -> None:
    if os.getenv("RUN_REAL_QWEN_SMOKE") != "1":
        raise SystemExit("REAL_MODEL_SMOKE: NOT RUN (set RUN_REAL_QWEN_SMOKE=1)")
    settings = Settings()
    voices = VoiceProfileRegistry.load(
        settings.voice_profiles_file,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
    )
    generations = GenerationProfileRegistry.load(settings.generation_profiles_file)
    voice = voices.get(settings.default_voice_profile)
    generation = generations.get(settings.default_generation_profile)
    provider = QwenTTSProvider(settings)
    await provider.initialize()
    await provider.prepare_voice(voice)
    seed = derive_chunk_seed(12345, 0)
    started = time.monotonic()
    audio = await provider.generate(
        text="Geralt aprende o sinal Aard pela primeira vez.",
        language="Portuguese",
        voice_profile=voice,
        generation_profile=generation,
        seed=seed,
    )
    elapsed = time.monotonic() - started
    publisher = AtomicWavPublisher(Path("benchmark-results"), max_duration_seconds=120)
    job_id = "f" * 32
    _, metrics = publisher.publish(job_id, audio)
    print(
        json.dumps(
            {
                "status": "PASS",
                "latency_seconds": elapsed,
                "duration_seconds": metrics.duration_seconds,
                "real_time_factor": elapsed / metrics.duration_seconds,
                "sample_rate": metrics.sample_rate,
                "seed": seed,
                "model_id": settings.model_id,
                "model_revision": settings.model_revision,
                "sample_peak_dbfs": metrics.sample_peak_dbfs,
                "sha256": metrics.sha256,
            },
            indent=2,
        )
    )
    await provider.shutdown()


if __name__ == "__main__":
    asyncio.run(run())
