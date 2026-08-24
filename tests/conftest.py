from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_service.core.config import Settings
from voice_service.main import create_app

from tests.fakes import FakeTTSProvider


@pytest.fixture
def configured_settings(tmp_path: Path) -> Settings:
    reference = tmp_path / "reference.wav"
    transcript = tmp_path / "reference.txt"
    samples = 0.1 * np.sin(2 * np.pi * 220 * np.arange(2400) / 24000)
    sf.write(reference, samples, 24000, subtype="PCM_16")
    transcript.write_text("Referencia de voz para testes.", encoding="utf-8")
    voices = tmp_path / "voices.yaml"
    voices.write_text(
        "voice_profiles:\n"
        "  test_voice:\n"
        '    version: "1"\n'
        f"    reference_audio: {reference}\n"
        f"    reference_text: {transcript}\n"
        "    clone_mode: icl\n",
        encoding="utf-8",
    )
    generations = tmp_path / "generations.yaml"
    generations.write_text(
        "generation_profiles:\n"
        "  test_generation:\n"
        '    version: "1"\n'
        "    parameters:\n"
        "      temperature: 0.9\n"
        "      top_p: 1.0\n"
        "      top_k: 50\n"
        "      repetition_penalty: 1.05\n"
        "      max_new_tokens: 2048\n",
        encoding="utf-8",
    )
    return Settings(
        device="cpu",
        dtype="float32",
        voice_profiles_file=voices,
        generation_profiles_file=generations,
        default_voice_profile="test_voice",
        default_generation_profile="test_generation",
        output_directory=tmp_path / "outputs",
        job_database_path=tmp_path / "jobs" / "jobs.sqlite3",
        target_chunk_chars=40,
        max_chunk_chars=64,
        max_text_chars=1000,
        max_queue_size=2,
        sentence_pause_ms=10,
        paragraph_pause_ms=20,
        fallback_pause_ms=5,
        hard_split_pause_ms=2,
    )


@pytest.fixture
def app_factory(
    configured_settings: Settings,
) -> Callable[..., tuple[object, FakeTTSProvider]]:
    def factory(**provider_options: object) -> tuple[object, FakeTTSProvider]:
        provider = FakeTTSProvider(configured_settings)
        for name, value in provider_options.items():
            setattr(provider, name, value)
        app = create_app(
            settings=configured_settings,
            provider_factory=lambda _: provider,
        )
        return app, provider

    return factory
