from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
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


def test_xvector_profile_requires_only_audio_and_has_distinct_identity(
    configured_settings: Settings, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.wav"
    transcript = tmp_path / "reference.txt"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    transcript.write_text("Texto de referência.", encoding="utf-8")
    config = tmp_path / "voices.yaml"
    config.write_text(
        "voice_profiles:\n"
        "  icl:\n"
        "    version: '1'\n"
        f"    reference_audio: {reference}\n"
        f"    reference_text: {transcript}\n"
        "    clone_mode: icl\n"
        "  xvec:\n"
        "    version: '1'\n"
        f"    reference_audio: {reference}\n"
        "    clone_mode: x_vector_only\n",
        encoding="utf-8",
    )
    registry = VoiceProfileRegistry.load(
        config,
        model_id=configured_settings.model_id,
        model_revision=configured_settings.model_revision,
    )
    icl = registry.get("icl")
    xvec = registry.get("xvec")
    assert icl.transcript == "Texto de referência."
    assert icl.transcript_sha256 is not None
    assert xvec.transcript is None
    assert xvec.transcript_sha256 is None
    assert xvec.clone_mode == "x_vector_only"
    assert xvec.identity != icl.identity


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


def test_frozen_production_profiles_have_expected_contract(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    raw_voices = yaml.safe_load((repository / "config" / "voices.yaml").read_text())
    for profile in raw_voices["voice_profiles"].values():
        profile["reference_audio"] = str(
            repository / "data" / "references" / Path(profile["reference_audio"]).name
        )
        if profile.get("reference_text"):
            profile["reference_text"] = str(
                repository / "data" / "references" / Path(profile["reference_text"]).name
            )
    local_voices = tmp_path / "voices.yaml"
    local_voices.write_text(yaml.safe_dump(raw_voices), encoding="utf-8")

    settings = Settings(_env_file=None)
    voices = VoiceProfileRegistry.load(
        local_voices,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
    )
    production_voice = voices.get(settings.default_voice_profile)
    assert production_voice.name == "bernardo_xvec_v3"
    assert production_voice.reference_id == "ref_full_processed_raw_normalized.wav"
    assert production_voice.audio_sha256 == (
        "11b7b30c41c3d365c4122a1549cd505d16876219f75fa213f87fa15e341be546"
    )
    assert production_voice.clone_mode == "x_vector_only"
    assert production_voice.transcript is None
    assert production_voice.transcript_sha256 is None
    assert production_voice.identity == (
        "bf047b087b09e7a2566a1282a8a9eeb4a043ef44051b2030859ca94ae46dde4f"
    )
    icl_candidate = voices.get("bernardo_icl_candidate_v1")
    assert icl_candidate.identity != production_voice.identity
    assert icl_candidate.clone_mode == "icl"
    assert icl_candidate.reference_id == "ref_full_processed_raw.wav"
    assert icl_candidate.audio_sha256 == (
        "8856924e05f76d1a3a5b8554218f80b5e1a898aebfe211ec1b65b60eabdbc70b"
    )
    assert icl_candidate.transcript

    generations = GenerationProfileRegistry.load(repository / "config/generation_profiles.yaml")
    historical = generations.get("narration_v1")
    production_generation = generations.get(settings.default_generation_profile)
    assert production_generation.name == "narration_v2"
    assert production_generation.identity != historical.identity
    assert production_generation.identity == (
        "e5433e1cb8b20f91913ad37763b891f3b327a50be2f5c7d09e9a5fdaa97f05d7"
    )
    assert production_generation.parameters == {
        "do_sample": True,
        "temperature": 0.95,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "subtalker_dosample": True,
        "subtalker_temperature": 0.9,
        "subtalker_top_p": 1.0,
        "subtalker_top_k": 50,
        "max_new_tokens": 2048,
    }
    nonstreaming = generations.get("narration_nonstreaming_v1")
    assert nonstreaming.identity != production_generation.identity
    assert nonstreaming.parameters == {
        **production_generation.parameters,
        "non_streaming_mode": True,
    }


def test_configured_reference_digest_mismatch_fails_closed(
    configured_settings: Settings, tmp_path: Path
) -> None:
    reference = configured_settings.references_directory / "reference.wav"
    config = tmp_path / "digest-mismatch.yaml"
    config.write_text(
        "voice_profiles:\n"
        "  frozen:\n"
        "    version: '1'\n"
        f"    reference_audio: {reference}\n"
        f"    reference_sha256: '{'0' * 64}'\n"
        "    clone_mode: x_vector_only\n",
        encoding="utf-8",
    )
    with pytest.raises(ProfileConfigurationError, match="SHA-256 mismatch"):
        VoiceProfileRegistry.load(
            config,
            model_id=configured_settings.model_id,
            model_revision=configured_settings.model_revision,
        )


def test_optional_english_assets_do_not_break_required_registry(configured_settings, tmp_path):
    # Fake unit-test assets only; no operator reference is created or synthesized.
    config = yaml.safe_load(configured_settings.voice_profiles_file.read_text())
    en_audio = tmp_path / "optional.wav"
    en_text = tmp_path / "optional.txt"
    config["voice_profiles"]["optional_en"] = {
        "version": "1",
        "required": False,
        "reference_audio": str(en_audio),
        "reference_text": str(en_text),
        "clone_mode": "icl",
    }
    path = tmp_path / "optional.yaml"
    path.write_text(yaml.safe_dump(config))

    def load():
        return VoiceProfileRegistry.load(
            path,
            model_id=configured_settings.model_id,
            model_revision=configured_settings.model_revision,
        )

    baseline = load().get("test_voice").identity
    assert [voice.name for voice in load().all()] == ["test_voice"]
    en_text.write_text("A matching test transcript.")
    assert load().get("test_voice").identity == baseline
    en_audio.write_bytes(b"invalid wav")
    with pytest.raises(ProfileConfigurationError):
        load()
    source = Path(config["voice_profiles"]["test_voice"]["reference_audio"])
    en_audio.write_bytes(source.read_bytes())
    assert load().get("optional_en").clone_mode == "icl"
    en_text.write_text("")
    with pytest.raises(ProfileConfigurationError, match="empty"):
        load()


def test_optional_xvec_candidates_need_only_wav(configured_settings, tmp_path):
    repository = Path(__file__).resolve().parents[2]
    configured = yaml.safe_load((repository / "config/voices.yaml").read_text())["voice_profiles"]
    assert configured["bernardo_xvec_v3"]["clone_mode"] == "x_vector_only"
    assert "required" not in configured["bernardo_xvec_v3"]
    raw = yaml.safe_load(configured_settings.voice_profiles_file.read_text())
    fake_audio = raw["voice_profiles"]["test_voice"]["reference_audio"]
    candidates = ("bernardo_pt_xvec_candidate_v1", "bernardo_en_xvec_v1")
    for name in candidates:
        assert configured[name]["clone_mode"] == "x_vector_only"
        assert configured[name]["required"] is False
        assert "reference_text" not in configured[name]
        raw["voice_profiles"][name] = {**configured[name], "reference_audio": str(tmp_path / (name + ".wav"))}
    config = tmp_path / "optional-xvec.yaml"

    def load():
        config.write_text(yaml.safe_dump(raw))
        return VoiceProfileRegistry.load(config, model_id=configured_settings.model_id, model_revision=configured_settings.model_revision)

    registry = load()
    assert [profile.name for profile in registry.all()] == ["test_voice"]
    identity = registry.get("test_voice").identity
    # Reuse the existing fake fixture path, never any operator reference.
    for name in candidates:
        raw["voice_profiles"][name]["reference_audio"] = fake_audio
    registry = load()
    assert registry.get("test_voice").identity == identity
    for name in candidates:
        assert registry.get(name).clone_mode == "x_vector_only"
        assert registry.get(name).transcript is None
