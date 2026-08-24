from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_service.domain import AudioChunk, TextChunk
from voice_service.services.audio import (
    AtomicWavPublisher,
    AudioAssembler,
    AudioValidationError,
    normalize_audio_chunk,
    validate_wav,
)


def text_chunk(index: int, boundary: str) -> TextChunk:
    return TextChunk(index, index, index + 1, "x", boundary, index + 1)


def test_assembly_preserves_order_and_inserts_silence_without_crossfade() -> None:
    assembler = AudioAssembler(
        sentence_pause_ms=100,
        paragraph_pause_ms=200,
        fallback_pause_ms=50,
        hard_split_pause_ms=20,
    )
    first = AudioChunk(np.full(10, 0.25, dtype=np.float32), 1000)
    second = AudioChunk(np.full(10, -0.25, dtype=np.float32), 1000)
    result = assembler.assemble([text_chunk(0, "sentence"), text_chunk(1, "end")], [first, second])
    assert result.samples.shape == (120, 1)
    np.testing.assert_allclose(result.samples[:10, 0], 0.25)
    np.testing.assert_allclose(result.samples[10:110, 0], 0.0)
    np.testing.assert_allclose(result.samples[110:, 0], -0.25)


def test_invalid_provider_audio_is_rejected() -> None:
    with pytest.raises(AudioValidationError, match="non-finite"):
        normalize_audio_chunk(AudioChunk(np.array([0.0, np.nan]), 24000))
    with pytest.raises(AudioValidationError, match="sample rate"):
        normalize_audio_chunk(AudioChunk(np.zeros(10), 0))
    with pytest.raises(AudioValidationError, match="PCM range"):
        normalize_audio_chunk(AudioChunk(np.array([1.2]), 24000))


def test_atomic_publication_reopens_validates_and_hashes(tmp_path: Path) -> None:
    publisher = AtomicWavPublisher(tmp_path, max_duration_seconds=10)
    job_id = "a" * 32
    uri, metrics = publisher.publish(
        job_id, AudioChunk(np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000)
    )
    assert uri == f"narrations/{job_id}/audio"
    assert metrics.sample_rate == 24000
    assert metrics.channels == 1
    assert metrics.subtype == "PCM_16"
    assert metrics.duration_seconds == pytest.approx(0.1)
    assert len(metrics.sha256) == 64
    assert publisher.path_for(job_id).is_file()
    assert not list(tmp_path.glob("*.tmp.wav"))
    assert validate_wav(publisher.path_for(job_id), max_duration_seconds=10) == metrics


def test_invalid_saved_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "clipped.wav"
    sf.write(path, np.ones(2400, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(AudioValidationError, match="clipping"):
        validate_wav(path, max_duration_seconds=10)


def test_storage_rejects_path_traversal(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(AudioValidationError, match="silent"):
        validate_wav(silent, max_duration_seconds=10)
    with pytest.raises(ValueError, match="job id"):
        AtomicWavPublisher(tmp_path, max_duration_seconds=10).path_for("../escape")
