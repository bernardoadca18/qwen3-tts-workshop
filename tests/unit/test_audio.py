from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_service.domain import AudioChunk, TextChunk
from voice_service.services.audio import (
    AtomicWavPublisher,
    AudioAssembler,
    AudioValidationError,
    ChunkQualityGuardError,
    normalize_audio_chunk,
    validate_audio_chunk_quality,
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
    timeline = assembler.assemble_with_timeline(
        [text_chunk(0, "sentence"), text_chunk(1, "end")], [first, second]
    )
    assert timeline.expected_frame_count == 120
    assert timeline.timeline[1] == {
        "kind": "inserted_pause",
        "chunk_index": 0,
        "start_seconds": 0.01,
        "end_seconds": 0.11,
        "duration_seconds": 0.1,
        "boundary_type": "sentence",
        "reason": "configured deterministic chunk-boundary pause",
    }


def test_invalid_provider_audio_is_rejected() -> None:
    with pytest.raises(AudioValidationError, match="non-finite"):
        normalize_audio_chunk(AudioChunk(np.array([0.0, np.nan]), 24000))
    with pytest.raises(AudioValidationError, match="sample rate"):
        normalize_audio_chunk(AudioChunk(np.zeros(10), 0))
    with pytest.raises(AudioValidationError, match="PCM range"):
        normalize_audio_chunk(AudioChunk(np.array([1.2]), 24000))


def test_chunk_quality_guard_accepts_plausible_duration() -> None:
    phase = np.arange(24000 * 4, dtype=np.float32)
    metrics = validate_audio_chunk_quality(
        AudioChunk(0.1 * np.sin(2 * np.pi * 220 * phase / 24000), 24000),
        text="Uma frase informativa com duracao perfeitamente plausivel.",
        max_duration_seconds=90,
        max_seconds_per_char=0.35,
        duration_grace_seconds=10,
        min_peak_amplitude=1e-6,
    )
    assert metrics.audio_duration_seconds == pytest.approx(4)


def test_chunk_quality_guard_rejects_obvious_runaway() -> None:
    phase = np.arange(24000 * 160, dtype=np.float32)
    with pytest.raises(ChunkQualityGuardError, match="absolute_duration_limit") as error:
        validate_audio_chunk_quality(
            AudioChunk(0.1 * np.sin(2 * np.pi * 220 * phase / 24000), 24000),
            text="Paragrafo normal com algumas centenas de caracteres.",
            max_duration_seconds=90,
            max_seconds_per_char=0.35,
            duration_grace_seconds=10,
            min_peak_amplitude=1e-6,
        )
    assert error.value.metrics.audio_duration_seconds == pytest.approx(160)


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
    assert metrics.frame_count == 2400
    assert metrics.riff_data_byte_size == 4800
    assert len(metrics.sha256) == 64
    assert publisher.path_for(job_id).is_file()
    assert not list(tmp_path.glob("*.tmp.wav"))
    assert validate_wav(publisher.path_for(job_id), max_duration_seconds=10) == metrics


def test_invalid_saved_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "clipped.wav"
    sf.write(path, np.ones(2400, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(AudioValidationError, match="clipping"):
        validate_wav(path, max_duration_seconds=10)


def test_truncated_riff_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    sf.write(
        path,
        np.linspace(-0.2, 0.2, 2400, dtype=np.float32),
        24000,
        subtype="PCM_16",
    )
    path.write_bytes(path.read_bytes()[:-10])

    with pytest.raises(AudioValidationError, match="RIFF size"):
        validate_wav(path, max_duration_seconds=10)


def test_storage_rejects_path_traversal(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(AudioValidationError, match="silent"):
        validate_wav(silent, max_duration_seconds=10)
    with pytest.raises(ValueError, match="job id"):
        AtomicWavPublisher(tmp_path, max_duration_seconds=10).path_for("../escape")
