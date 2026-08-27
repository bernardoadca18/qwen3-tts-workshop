from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from voice_service.domain import AudioChunk, AudioMetrics, ChunkQualityMetrics, TextChunk


class AudioValidationError(ValueError):
    pass


class ChunkQualityGuardError(AudioValidationError):
    def __init__(self, reason: str, metrics: ChunkQualityMetrics):
        super().__init__(f"generated chunk failed quality guard: {reason}")
        self.reason = reason
        self.metrics = metrics
        self.diagnostics: dict[str, object] = {}


def validate_audio_chunk_quality(
    chunk: AudioChunk,
    *,
    text: str,
    max_duration_seconds: float,
    max_seconds_per_char: float,
    duration_grace_seconds: float,
    min_peak_amplitude: float,
) -> ChunkQualityMetrics:
    samples = normalize_audio_chunk(chunk)
    text_length = len(text)
    duration = samples.shape[0] / chunk.sample_rate
    peak = float(np.max(np.abs(samples)))
    relative_limit = duration_grace_seconds + max_seconds_per_char * max(text_length, 1)
    allowed_duration = min(max_duration_seconds, relative_limit)
    metrics = ChunkQualityMetrics(
        text_length_chars=text_length,
        audio_duration_seconds=duration,
        sample_rate=chunk.sample_rate,
        sample_peak=peak,
        maximum_allowed_duration_seconds=allowed_duration,
    )
    if not math.isfinite(duration) or duration <= 0:
        raise ChunkQualityGuardError("invalid_duration", metrics)
    if peak <= min_peak_amplitude:
        raise ChunkQualityGuardError("silent_or_near_silent", metrics)
    if duration > max_duration_seconds:
        raise ChunkQualityGuardError("absolute_duration_limit", metrics)
    if duration > relative_limit:
        raise ChunkQualityGuardError("duration_to_text_ratio", metrics)
    return metrics


def normalize_audio_chunk(chunk: AudioChunk) -> NDArray[np.float32]:
    samples = np.asarray(chunk.samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise AudioValidationError("provider returned an invalid waveform shape")
    if chunk.sample_rate <= 0:
        raise AudioValidationError("provider returned an invalid sample rate")
    if not np.all(np.isfinite(samples)):
        raise AudioValidationError("provider returned non-finite samples")
    if float(np.max(np.abs(samples))) > 1.0:
        raise AudioValidationError("provider waveform exceeds the PCM range")
    return samples


class AudioAssembler:
    def __init__(
        self,
        *,
        sentence_pause_ms: int,
        paragraph_pause_ms: int,
        fallback_pause_ms: int,
        hard_split_pause_ms: int,
    ):
        self.pause_ms = {
            "sentence": sentence_pause_ms,
            "paragraph": paragraph_pause_ms,
            "fallback": fallback_pause_ms,
            "hard": hard_split_pause_ms,
            "end": 0,
        }

    def assemble(
        self,
        text_chunks: list[TextChunk],
        audio_chunks: list[AudioChunk],
    ) -> AudioChunk:
        if not text_chunks or len(text_chunks) != len(audio_chunks):
            raise AudioValidationError("audio chunk count does not match text chunks")
        normalized = [normalize_audio_chunk(chunk) for chunk in audio_chunks]
        sample_rate = audio_chunks[0].sample_rate
        channels = normalized[0].shape[1]
        parts: list[NDArray[np.float32]] = []
        for index, (text_chunk, samples) in enumerate(zip(text_chunks, normalized, strict=True)):
            if audio_chunks[index].sample_rate != sample_rate or samples.shape[1] != channels:
                raise AudioValidationError(
                    "provider changed sample rate or channels between chunks"
                )
            parts.append(samples)
            if index < len(normalized) - 1:
                pause_ms = self.pause_ms.get(text_chunk.boundary, self.pause_ms["fallback"])
                pause_frames = round(sample_rate * pause_ms / 1000)
                if pause_frames:
                    parts.append(np.zeros((pause_frames, channels), dtype=np.float32))
        return AudioChunk(samples=np.concatenate(parts, axis=0), sample_rate=sample_rate)


def validate_wav(path: Path, *, max_duration_seconds: float) -> AudioMetrics:
    try:
        info = sf.info(str(path))
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise AudioValidationError("published WAV is unreadable") from exc
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise AudioValidationError("result must be PCM16 WAV")
    if samples.size == 0 or sample_rate <= 0 or not np.all(np.isfinite(samples)):
        raise AudioValidationError("published WAV contains invalid audio")
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0 or duration > max_duration_seconds:
        raise AudioValidationError("published WAV duration is invalid")
    absolute = np.abs(samples)
    peak = float(np.max(absolute))
    if peak <= 1e-6:
        raise AudioValidationError("published WAV is silent")
    clipping_count = int(np.count_nonzero(absolute >= 0.999))
    if clipping_count / samples.size > 0.001:
        raise AudioValidationError("published WAV contains obvious digital clipping")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    peak_dbfs = 20 * math.log10(peak) if peak > 0 else float("-inf")
    return AudioMetrics(
        duration_seconds=duration,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=info.subtype,
        byte_size=path.stat().st_size,
        sha256=digest,
        sample_peak_dbfs=peak_dbfs,
        clipping_sample_count=clipping_count,
    )


class AtomicWavPublisher:
    def __init__(self, output_directory: Path, *, max_duration_seconds: float):
        self.output_directory = output_directory
        self.max_duration_seconds = max_duration_seconds

    def initialize(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("invalid job id")
        return self.output_directory / f"{job_id}.wav"

    def publish(self, job_id: str, audio: AudioChunk) -> tuple[str, AudioMetrics]:
        samples = normalize_audio_chunk(audio)
        self.initialize()
        final_path = self.path_for(job_id)
        temporary = self.output_directory / f".{job_id}.{uuid4().hex}.tmp.wav"
        try:
            sf.write(str(temporary), samples, audio.sample_rate, format="WAV", subtype="PCM_16")
            metrics = validate_wav(
                temporary,
                max_duration_seconds=self.max_duration_seconds,
            )
            os.replace(temporary, final_path)
            return f"narrations/{job_id}/audio", metrics
        finally:
            temporary.unlink(missing_ok=True)

    def reconcile(self, job_id: str) -> AudioMetrics | None:
        path = self.path_for(job_id)
        if not path.is_file():
            return None
        try:
            return validate_wav(path, max_duration_seconds=self.max_duration_seconds)
        except AudioValidationError:
            return None

    def delete(self, job_id: str) -> None:
        self.path_for(job_id).unlink(missing_ok=True)
