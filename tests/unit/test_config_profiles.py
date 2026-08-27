from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_service.core.config import Settings
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    ProfileConfigurationError,
    VoiceProfileRegistry,
)


def test_settings_validate_device_dtype_and_chunk_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float16 on CPU"):
        Settings(device="cpu", dtype="float16")
    with pytest.raises(ValueError, match="TARGET_CHUNK_CHARS"):
        Settings(target_chunk_chars=900, max_chunk_chars=800)
    settings = Settings(device="cpu", dtype="float32", job_database_path=tmp_path / "db")
    assert settings.model_revision


def test_production_chunk_and_quality_guard_defaults_are_conservative() -> None:
    settings = Settings(_env_file=None)
    assert settings.target_chunk_chars == 250
    assert settings.max_chunk_chars == 400
    assert settings.chunk_max_duration_seconds == 90
    with pytest.raises(ValueError, match="CHUNK_DURATION_GRACE_SECONDS"):
        Settings(
            _env_file=None,
            chunk_max_duration_seconds=10,
            chunk_duration_grace_seconds=10,
        )


def test_voice_profile_loads_metadata_and_hash_is_stable(configured_settings: Settings) -> None:
    first = VoiceProfileRegistry.load(
        configured_settings.voice_profiles_file,
        model_id=configured_settings.model_id,
        model_revision=configured_settings.model_revision,
    ).get("test_voice")
    second = VoiceProfileRegistry.load(
        configured_settings.voice_profiles_file,
        model_id=configured_settings.model_id,
        model_revision=configured_settings.model_revision,
    ).get("test_voice")
    assert first.identity == second.identity
    assert len(first.audio_sha256) == 64
    assert len(first.transcript_sha256) == 64
    assert first.sample_rate == 24000
    assert first.channels == 1
    assert first.duration_seconds > 0


def test_voice_profile_rejects_missing_file(configured_settings: Settings, tmp_path: Path) -> None:
    config = tmp_path / "missing.yaml"
    config.write_text(
        "voice_profiles:\n  broken:\n    version: '1'\n"
        "    reference_audio: /does/not/exist.wav\n"
        "    reference_text: /does/not/exist.txt\n"
        "    clone_mode: icl\n",
        encoding="utf-8",
    )
    with pytest.raises(ProfileConfigurationError, match="missing"):
        VoiceProfileRegistry.load(
            config,
            model_id=configured_settings.model_id,
            model_revision=configured_settings.model_revision,
        )


def test_voice_profile_rejects_empty_transcript(
    configured_settings: Settings, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.wav"
    transcript = tmp_path / "empty.txt"
    sf.write(reference, np.zeros(100, dtype=np.float32), 24000)
    transcript.write_text("  ", encoding="utf-8")
    config = tmp_path / "voices.yaml"
    config.write_text(
        "voice_profiles:\n  broken:\n    version: '1'\n"
        f"    reference_audio: {reference}\n    reference_text: {transcript}\n"
        "    clone_mode: icl\n",
        encoding="utf-8",
    )
    with pytest.raises(ProfileConfigurationError, match="empty"):
        VoiceProfileRegistry.load(
            config,
            model_id=configured_settings.model_id,
            model_revision=configured_settings.model_revision,
        )


def test_generation_profile_resolves_and_rejects_invalid(
    configured_settings: Settings, tmp_path: Path
) -> None:
    profile = GenerationProfileRegistry.load(configured_settings.generation_profiles_file).get(
        "test_generation"
    )
    assert profile.parameters["temperature"] == 0.9
    assert profile.parameters["subtalker_temperature"] == 0.9
    with pytest.raises(ProfileConfigurationError, match="unknown"):
        GenerationProfileRegistry.load(configured_settings.generation_profiles_file).get("missing")
    invalid = tmp_path / "invalid-generation.yaml"
    invalid.write_text(
        "generation_profiles:\n  broken:\n    version: '1'\n"
        "    parameters:\n      temperature: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        GenerationProfileRegistry.load(invalid)
