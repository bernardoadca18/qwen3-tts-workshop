"""Validation tests for experimental ICL voice profiles."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from voice_service.core.config import Settings
from voice_service.services.profiles import VoiceProfileRegistry


def test_icl_profiles_in_production_yaml_have_correct_structure() -> None:
    """Validate ICL profiles exist in voices.yaml with correct fields."""
    repository = Path(__file__).resolve().parents[2]
    raw_voices = yaml.safe_load((repository / "config" / "voices.yaml").read_text())

    # PT ICL profile
    pt_icl = raw_voices["voice_profiles"]["bernardo_pt_icl_v1"]
    assert pt_icl["clone_mode"] == "icl"
    assert pt_icl["required"] is False
    assert "ref_pt_br_xvec_v4.wav" in pt_icl["reference_audio"]
    assert "ref_pt_br_xvec_v4.txt" in pt_icl["reference_text"]
    assert pt_icl["reference_sha256"] == "6d2df3e475cf58c7c14a5e70f8b345837a965c159dc6d79d4e8204d1e535dde7"

    # EN ICL profile
    en_icl = raw_voices["voice_profiles"]["bernardo_en_icl_v1"]
    assert en_icl["clone_mode"] == "icl"
    assert en_icl["required"] is False
    assert "ref_en_us_xvec_v2.wav" in en_icl["reference_audio"]
    assert "ref_en_us_xvec_v2.txt" in en_icl["reference_text"]
    assert en_icl["reference_sha256"] == "a2b84a1596e64f69cc81bde39e6fa2efe7823318f655f4ca39ad1f1082d7b52a"


def test_existing_xvec_profiles_remain_x_vector_only() -> None:
    """Existing xvec profiles must remain x_vector_only."""
    repository = Path(__file__).resolve().parents[2]
    raw_voices = yaml.safe_load((repository / "config" / "voices.yaml").read_text())

    pt_xvec = raw_voices["voice_profiles"]["bernardo_xvec_v4"]
    assert pt_xvec["clone_mode"] == "x_vector_only"

    en_xvec = raw_voices["voice_profiles"]["bernardo_en_xvec_v2"]
    assert en_xvec["clone_mode"] == "x_vector_only"


def test_icl_profiles_load_and_validate_correctly(configured_settings: Settings, tmp_path: Path) -> None:
    """ICL profiles must load with correct clone_mode, transcript, and identity."""
    repository = Path(__file__).resolve().parents[2]
    raw_voices = yaml.safe_load((repository / "config" / "voices.yaml").read_text())

    # Create local copies of production reference files
    ref_pt_audio = repository / "data" / "references" / "ref_pt_br_xvec_v4.wav"
    ref_pt_text = repository / "data" / "references" / "ref_pt_br_xvec_v4.txt"
    ref_en_audio = repository / "data" / "references" / "ref_en_us_xvec_v2.wav"
    ref_en_text = repository / "data" / "references" / "ref_en_us_xvec_v2.txt"

    # Build minimal config with just the profiles we need to test
    local_voices = tmp_path / "voices.yaml"
    test_profiles = {
        "test_pt_icl": {
            "version": "1",
            "reference_audio": str(ref_pt_audio),
            "reference_text": str(ref_pt_text),
            "reference_sha256": raw_voices["voice_profiles"]["bernardo_pt_icl_v1"]["reference_sha256"],
            "clone_mode": "icl",
        },
        "test_en_icl": {
            "version": "1",
            "reference_audio": str(ref_en_audio),
            "reference_text": str(ref_en_text),
            "reference_sha256": raw_voices["voice_profiles"]["bernardo_en_icl_v1"]["reference_sha256"],
            "clone_mode": "icl",
        },
        "test_pt_xvec": {
            "version": "1",
            "reference_audio": str(ref_pt_audio),
            "reference_sha256": raw_voices["voice_profiles"]["bernardo_xvec_v4"]["reference_sha256"],
            "clone_mode": "x_vector_only",
        },
        "test_en_xvec": {
            "version": "1",
            "reference_audio": str(ref_en_audio),
            "reference_sha256": raw_voices["voice_profiles"]["bernardo_en_xvec_v2"]["reference_sha256"],
            "clone_mode": "x_vector_only",
        },
    }
    local_voices.write_text(yaml.safe_dump({"voice_profiles": test_profiles}), encoding="utf-8")

    settings = Settings(_env_file=None)
    registry = VoiceProfileRegistry.load(
        local_voices,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
    )

    # PT ICL validation
    pt_icl = registry.get("test_pt_icl")
    assert pt_icl.clone_mode == "icl"
    assert pt_icl.transcript is not None
    assert len(pt_icl.transcript.strip()) > 0
    assert pt_icl.transcript_sha256 is not None
    assert pt_icl.audio_sha256 == "6d2df3e475cf58c7c14a5e70f8b345837a965c159dc6d79d4e8204d1e535dde7"

    # EN ICL validation
    en_icl = registry.get("test_en_icl")
    assert en_icl.clone_mode == "icl"
    assert en_icl.transcript is not None
    assert len(en_icl.transcript.strip()) > 0
    assert en_icl.transcript_sha256 is not None
    assert en_icl.audio_sha256 == "a2b84a1596e64f69cc81bde39e6fa2efe7823318f655f4ca39ad1f1082d7b52a"

    # XVEC profiles still work
    pt_xvec = registry.get("test_pt_xvec")
    assert pt_xvec.clone_mode == "x_vector_only"
    assert pt_xvec.transcript is None

    en_xvec = registry.get("test_en_xvec")
    assert en_xvec.clone_mode == "x_vector_only"
    assert en_xvec.transcript is None

    # Identities differ (clone_mode + transcript affect identity)
    assert pt_icl.identity != pt_xvec.identity
    assert en_icl.identity != en_xvec.identity


def test_defaults_unchanged() -> None:
    """Production defaults must remain unchanged."""
    settings = Settings(_env_file=None)
    assert settings.default_voice_profile == "bernardo_xvec_v4"
    assert settings.default_generation_profile == "narration_v2"
