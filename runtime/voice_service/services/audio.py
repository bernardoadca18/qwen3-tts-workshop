from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
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


class SpeakerSimilarityGuardError(AudioValidationError):
    """Raised when generated audio fails speaker similarity validation."""

    def __init__(
        self,
        similarity: float,
        threshold: float,
        reference_identity: str,
    ):
        super().__init__(
            f"speaker similarity {similarity:.4f} below threshold {threshold:.4f} "
            f"for voice identity {reference_identity}"
        )
        self.similarity = similarity
        self.threshold = threshold
        self.reference_identity = reference_identity
        self.diagnostics: dict[str, object] = {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def validate_speaker_similarity(
    generated_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    threshold: float,
    reference_identity: str,
) -> float:
    """Validate speaker similarity between generated and reference embeddings.

    Args:
        generated_embedding: Speaker embedding from generated audio.
        reference_embedding: Speaker embedding from reference voice profile.
        threshold: Minimum cosine similarity required (0.0 to 1.0).
        reference_identity: Identity hash of the voice profile for diagnostics.

    Returns:
        The computed similarity score.

    Raises:
        SpeakerSimilarityGuardError: If similarity is below threshold.
    """
    similarity = cosine_similarity(generated_embedding, reference_embedding)

    if similarity < threshold:
        raise SpeakerSimilarityGuardError(
            similarity=similarity,
            threshold=threshold,
            reference_identity=reference_identity,
        )

    return similarity


@dataclass(frozen=True)
class AudioAssemblyResult:
    audio: AudioChunk
    timeline: tuple[dict[str, object], ...]
    expected_frame_count: int


@dataclass(frozen=True)
class StreamingAssemblyResult:
    timeline: tuple[dict[str, object], ...]
    expected_frame_count: int


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
        return self.assemble_with_timeline(text_chunks, audio_chunks).audio

    def assemble_with_timeline(
        self,
        text_chunks: list[TextChunk],
        audio_chunks: list[AudioChunk],
    ) -> AudioAssemblyResult:
        if not text_chunks or len(text_chunks) != len(audio_chunks):
            raise AudioValidationError("audio chunk count does not match text chunks")
        normalized = [normalize_audio_chunk(chunk) for chunk in audio_chunks]
        sample_rate = audio_chunks[0].sample_rate
        channels = normalized[0].shape[1]
        parts: list[NDArray[np.float32]] = []
        timeline: list[dict[str, object]] = []
        current_frame = 0
        for index, (text_chunk, samples) in enumerate(zip(text_chunks, normalized, strict=True)):
            if audio_chunks[index].sample_rate != sample_rate or samples.shape[1] != channels:
                raise AudioValidationError(
                    "provider changed sample rate or channels between chunks"
                )
            start = current_frame
            parts.append(samples)
            current_frame += len(samples)
            timeline.append(
                {
                    "kind": "generated_chunk",
                    "chunk_index": text_chunk.index,
                    "start_seconds": start / sample_rate,
                    "end_seconds": current_frame / sample_rate,
                    "boundary_type": text_chunk.boundary,
                    "text_start_char": text_chunk.start_char,
                    "text_end_char": text_chunk.end_char,
                }
            )
            if index < len(normalized) - 1:
                pause_ms = self.pause_ms.get(text_chunk.boundary, self.pause_ms["fallback"])
                pause_frames = round(sample_rate * pause_ms / 1000)
                if pause_frames:
                    pause_start = current_frame
                    parts.append(np.zeros((pause_frames, channels), dtype=np.float32))
                    current_frame += pause_frames
                    timeline.append(
                        {
                            "kind": "inserted_pause",
                            "chunk_index": text_chunk.index,
                            "start_seconds": pause_start / sample_rate,
                            "end_seconds": current_frame / sample_rate,
                            "duration_seconds": pause_frames / sample_rate,
                            "boundary_type": text_chunk.boundary,
                            "reason": "configured deterministic chunk-boundary pause",
                        }
                    )
        audio = AudioChunk(samples=np.concatenate(parts, axis=0), sample_rate=sample_rate)
        if len(audio.samples) != current_frame:
            raise AudioValidationError("assembled sample count differs from the timeline")
        return AudioAssemblyResult(audio, tuple(timeline), current_frame)


def analyze_audio_chunk_signal(chunk: AudioChunk) -> dict[str, object]:
    samples = normalize_audio_chunk(chunk)
    mono = np.mean(samples, axis=1)
    threshold = 10.0 ** (-45.0 / 20.0)
    mask = np.abs(mono) <= threshold
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    minimum_frames = max(1, round(chunk.sample_rate * 0.15))
    runs = [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= minimum_frames
    ]
    difference = np.abs(np.diff(mono, prepend=mono[0]))
    median = float(np.median(difference))
    mad = max(float(np.median(np.abs(difference - median))) * 1.4826, 1e-9)
    transient_indexes = np.flatnonzero(difference >= max(0.05, median + 12.0 * mad))
    transient_times: list[float] = []
    for index in transient_indexes:
        time_seconds = int(index) / chunk.sample_rate
        if not transient_times or time_seconds - transient_times[-1] >= 0.04:
            transient_times.append(time_seconds)
        if len(transient_times) >= 50:
            break
    duration = len(mono) / chunk.sample_rate
    silence_total = sum(end - start for start, end in runs) / chunk.sample_rate
    return {
        "version": "qwen_chunk_signal_diagnostics_v1",
        "silence_threshold_dbfs": -45.0,
        "silence_run_count": len(runs),
        "silence_total_seconds": round(silence_total, 6),
        "silence_ratio": round(silence_total / duration, 6),
        "longest_silence_seconds": round(
            max((end - start for start, end in runs), default=0) / chunk.sample_rate, 6
        ),
        "silence_runs": [
            {
                "start_seconds": round(start / chunk.sample_rate, 6),
                "end_seconds": round(end / chunk.sample_rate, 6),
            }
            for start, end in runs
        ],
        "transient_candidate_count": len(transient_times),
        "transient_candidate_times_seconds": [round(value, 6) for value in transient_times],
    }


def _inspect_riff(path: Path) -> tuple[int, int]:
    file_size = path.stat().st_size
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise AudioValidationError("published WAV has an invalid RIFF envelope")
            declared_size = struct.unpack_from("<I", header, 4)[0] + 8
            if declared_size != file_size:
                raise AudioValidationError("published WAV RIFF size differs from physical size")
            block_align: int | None = None
            format_chunk_count = 0
            data_sizes: list[int] = []
            position = 12
            while position < file_size:
                chunk_header = stream.read(8)
                if len(chunk_header) != 8:
                    raise AudioValidationError("published WAV chunk header is truncated")
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack_from("<I", chunk_header, 4)[0]
                position += 8
                padded_size = chunk_size + chunk_size % 2
                if position + padded_size > file_size:
                    raise AudioValidationError("published WAV chunk is truncated")
                if chunk_id == b"fmt ":
                    format_chunk_count += 1
                    payload = stream.read(chunk_size)
                    if len(payload) < 16:
                        raise AudioValidationError("published WAV format chunk is truncated")
                    audio_format, channels, _, _, block_align, bits = struct.unpack_from(
                        "<HHIIHH", payload
                    )
                    if audio_format not in {1, 0xFFFE}:
                        raise AudioValidationError("published WAV is not PCM")
                    if channels <= 0 or block_align <= 0 or bits <= 0:
                        raise AudioValidationError("published WAV format chunk is invalid")
                    if padded_size > chunk_size:
                        stream.seek(padded_size - chunk_size, os.SEEK_CUR)
                else:
                    if chunk_id == b"data":
                        data_sizes.append(chunk_size)
                    stream.seek(padded_size, os.SEEK_CUR)
                position += padded_size
    except OSError as exc:
        raise AudioValidationError("published WAV RIFF envelope is unreadable") from exc
    if (
        position != file_size
        or block_align is None
        or format_chunk_count != 1
        or len(data_sizes) != 1
    ):
        raise AudioValidationError("published WAV must contain one complete format and data chunk")
    return block_align, data_sizes[0]


def validate_wav(path: Path, *, max_duration_seconds: float) -> AudioMetrics:
    block_align, riff_data_byte_size = _inspect_riff(path)
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise AudioValidationError("published WAV is unreadable") from exc
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise AudioValidationError("result must be PCM16 WAV")
    if info.frames <= 0 or info.samplerate <= 0:
        raise AudioValidationError("published WAV contains invalid audio")
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0 or duration > max_duration_seconds:
        raise AudioValidationError("published WAV duration is invalid")
    if int(info.frames) * block_align != riff_data_byte_size:
        raise AudioValidationError("published WAV sample count differs from RIFF data size")
    peak = 0.0
    clipping_count = 0
    sample_count = 0
    try:
        with sf.SoundFile(str(path), mode="r") as source:
            for samples in source.blocks(
                blocksize=262_144,
                dtype="float32",
                always_2d=True,
            ):
                if not np.all(np.isfinite(samples)):
                    raise AudioValidationError("published WAV contains invalid audio")
                absolute = np.abs(samples)
                peak = max(peak, float(np.max(absolute)))
                clipping_count += int(np.count_nonzero(absolute >= 0.999))
                sample_count += int(absolute.size)
    except AudioValidationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AudioValidationError("published WAV is unreadable") from exc
    if sample_count == 0:
        raise AudioValidationError("published WAV contains invalid audio")
    if peak <= 1e-6:
        raise AudioValidationError("published WAV is silent")
    if clipping_count / sample_count > 0.001:
        raise AudioValidationError("published WAV contains obvious digital clipping")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AudioValidationError("published WAV is unreadable") from exc
    peak_dbfs = 20 * math.log10(peak) if peak > 0 else float("-inf")
    return AudioMetrics(
        duration_seconds=duration,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=info.subtype,
        byte_size=path.stat().st_size,
        sha256=digest.hexdigest(),
        sample_peak_dbfs=peak_dbfs,
        clipping_sample_count=clipping_count,
        frame_count=int(info.frames),
        riff_data_byte_size=riff_data_byte_size,
    )


class AtomicWavPublisher:
    def __init__(self, output_directory: Path, *, max_duration_seconds: float):
        self.output_directory = output_directory
        self.max_duration_seconds = max_duration_seconds

    def initialize(self) -> None:
        self._ensure_output_directory()
        for temporary in self.output_directory.glob(".*.tmp.wav"):
            temporary.unlink(missing_ok=True)

    def _ensure_output_directory(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("invalid job id")
        return self.output_directory / f"{job_id}.wav"

    def chunk_path_for(self, job_id: str, chunk_index: int) -> Path:
        """Path for a persisted chunk WAV file with stable identity."""
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("invalid job id")
        if chunk_index < 0:
            raise ValueError("invalid chunk index")
        return self.output_directory / "chunks" / job_id / f"chunk_{chunk_index:04d}.wav"

    def chunk_candidate_path_for(self, job_id: str, candidate_id: str) -> Path:
        """Path for one immutable regeneration artifact."""
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("invalid job id")
        if len(candidate_id) != 32 or any(
            char not in "0123456789abcdef" for char in candidate_id
        ):
            raise ValueError("invalid chunk candidate id")
        return (
            self.output_directory
            / "chunks"
            / job_id
            / "candidates"
            / f"{candidate_id}.wav"
        )

    def publish_chunk(
        self,
        job_id: str,
        chunk_index: int,
        audio: AudioChunk,
    ) -> tuple[str, AudioMetrics]:
        """Persist a single chunk WAV with stable identity."""
        samples = normalize_audio_chunk(audio)
        self._ensure_output_directory()
        chunk_dir = self.output_directory / "chunks" / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.chunk_path_for(job_id, chunk_index)
        temporary = self.output_directory / f".chunk_{job_id}_{chunk_index}.{uuid4().hex}.tmp.wav"
        try:
            sf.write(str(temporary), samples, audio.sample_rate, format="WAV", subtype="PCM_16")
            metrics = validate_wav(
                temporary,
                max_duration_seconds=self.max_duration_seconds,
            )
            os.replace(temporary, final_path)
            published = validate_wav(final_path, max_duration_seconds=self.max_duration_seconds)
            if published != metrics:
                final_path.unlink(missing_ok=True)
                raise AudioValidationError("published chunk WAV changed during atomic publication")
            return f"narrations/{job_id}/chunks/{chunk_index}", published
        finally:
            temporary.unlink(missing_ok=True)

    def publish_chunk_candidate(
        self,
        job_id: str,
        candidate_id: str,
        audio: AudioChunk,
    ) -> tuple[str, AudioMetrics]:
        """Publish one candidate exactly once without touching the original chunk."""
        samples = normalize_audio_chunk(audio)
        final_path = self.chunk_candidate_path_for(job_id, candidate_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            published = validate_wav(
                final_path,
                max_duration_seconds=self.max_duration_seconds,
            )
            return (
                f"narrations/{job_id}/chunk-candidates/{candidate_id}/audio",
                published,
            )

        temporary = final_path.parent / f".{candidate_id}.{uuid4().hex}.tmp.wav"
        try:
            sf.write(str(temporary), samples, audio.sample_rate, format="WAV", subtype="PCM_16")
            metrics = validate_wav(
                temporary,
                max_duration_seconds=self.max_duration_seconds,
            )
            try:
                # Hard-link publication is atomic and refuses to replace an existing identity.
                os.link(temporary, final_path)
            except FileExistsError:
                published = validate_wav(
                    final_path,
                    max_duration_seconds=self.max_duration_seconds,
                )
                return (
                    f"narrations/{job_id}/chunk-candidates/{candidate_id}/audio",
                    published,
                )
            published = validate_wav(final_path, max_duration_seconds=self.max_duration_seconds)
            if published != metrics:
                final_path.unlink(missing_ok=True)
                raise AudioValidationError(
                    "published chunk candidate WAV changed during atomic publication"
                )
            return (
                f"narrations/{job_id}/chunk-candidates/{candidate_id}/audio",
                published,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, job_id: str, audio: AudioChunk) -> tuple[str, AudioMetrics]:
        samples = normalize_audio_chunk(audio)
        self._ensure_output_directory()
        final_path = self.path_for(job_id)
        temporary = self.output_directory / f".{job_id}.{uuid4().hex}.tmp.wav"
        try:
            sf.write(str(temporary), samples, audio.sample_rate, format="WAV", subtype="PCM_16")
            metrics = validate_wav(
                temporary,
                max_duration_seconds=self.max_duration_seconds,
            )
            os.replace(temporary, final_path)
            published = validate_wav(final_path, max_duration_seconds=self.max_duration_seconds)
            if published != metrics:
                final_path.unlink(missing_ok=True)
                raise AudioValidationError("published WAV changed during atomic publication")
            return f"narrations/{job_id}/audio", published
        finally:
            temporary.unlink(missing_ok=True)

    def begin_stream(self, job_id: str, assembler: AudioAssembler) -> AtomicWavStream:
        self._ensure_output_directory()
        return AtomicWavStream(self, job_id=job_id, assembler=assembler)

    def reconcile(self, job_id: str) -> AudioMetrics | None:
        path = self.path_for(job_id)
        if not path.is_file():
            return None
        try:
            return validate_wav(path, max_duration_seconds=self.max_duration_seconds)
        except AudioValidationError:
            return None

    def reconcile_chunk(self, job_id: str, chunk_index: int) -> AudioMetrics | None:
        path = self.chunk_path_for(job_id, chunk_index)
        if not path.is_file():
            return None
        try:
            return validate_wav(path, max_duration_seconds=self.max_duration_seconds)
        except AudioValidationError:
            return None

    def reconcile_chunk_candidate(
        self, job_id: str, candidate_id: str
    ) -> AudioMetrics | None:
        path = self.chunk_candidate_path_for(job_id, candidate_id)
        if not path.is_file():
            return None
        try:
            return validate_wav(path, max_duration_seconds=self.max_duration_seconds)
        except AudioValidationError:
            return None

    def delete(self, job_id: str) -> None:
        self.path_for(job_id).unlink(missing_ok=True)


class AtomicWavStream:
    """Incrementally assemble one job and atomically publish only after validation."""

    def __init__(self, publisher: AtomicWavPublisher, *, job_id: str, assembler: AudioAssembler):
        self.publisher = publisher
        self.job_id = job_id
        self.assembler = assembler
        self.final_path = publisher.path_for(job_id)
        self.temporary = publisher.output_directory / f".{job_id}.{uuid4().hex}.tmp.wav"
        self._output: sf.SoundFile | None = None
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._current_frame = 0
        self._timeline: list[dict[str, object]] = []
        self._published = False

    def append(self, text_chunk: TextChunk, audio: AudioChunk, *, append_pause: bool) -> None:
        if self._published:
            raise AudioValidationError("cannot append to a published narration")
        samples = normalize_audio_chunk(audio)
        channels = int(samples.shape[1])
        if self._output is None:
            self._sample_rate = audio.sample_rate
            self._channels = channels
            self._output = sf.SoundFile(
                str(self.temporary),
                mode="w",
                samplerate=audio.sample_rate,
                channels=channels,
                format="WAV",
                subtype="PCM_16",
            )
        if audio.sample_rate != self._sample_rate or channels != self._channels:
            raise AudioValidationError("provider changed sample rate or channels between chunks")
        assert self._output is not None
        assert self._sample_rate is not None
        start_frame = self._current_frame
        self._output.write(samples)
        self._current_frame += len(samples)
        self._timeline.append(
            {
                "kind": "generated_chunk",
                "chunk_index": text_chunk.index,
                "start_seconds": start_frame / self._sample_rate,
                "end_seconds": self._current_frame / self._sample_rate,
                "boundary_type": text_chunk.boundary,
                "text_start_char": text_chunk.start_char,
                "text_end_char": text_chunk.end_char,
            }
        )
        if append_pause:
            pause_ms = self.assembler.pause_ms.get(
                text_chunk.boundary,
                self.assembler.pause_ms["fallback"],
            )
            pause_frames = round(self._sample_rate * pause_ms / 1000)
            if pause_frames:
                pause_start = self._current_frame
                self._output.write(
                    np.zeros((pause_frames, channels), dtype=np.float32)
                )
                self._current_frame += pause_frames
                self._timeline.append(
                    {
                        "kind": "inserted_pause",
                        "chunk_index": text_chunk.index,
                        "start_seconds": pause_start / self._sample_rate,
                        "end_seconds": self._current_frame / self._sample_rate,
                        "duration_seconds": pause_frames / self._sample_rate,
                        "boundary_type": text_chunk.boundary,
                        "reason": "configured deterministic chunk-boundary pause",
                    }
                )

    def publish(self) -> tuple[str, AudioMetrics, StreamingAssemblyResult]:
        if self._output is None or self._current_frame <= 0:
            raise AudioValidationError("cannot publish an empty narration")
        self._output.close()
        self._output = None
        metrics = validate_wav(
            self.temporary,
            max_duration_seconds=self.publisher.max_duration_seconds,
        )
        if metrics.frame_count != self._current_frame:
            raise AudioValidationError("assembled sample count differs from the timeline")
        os.replace(self.temporary, self.final_path)
        published = validate_wav(
            self.final_path,
            max_duration_seconds=self.publisher.max_duration_seconds,
        )
        if published != metrics:
            self.final_path.unlink(missing_ok=True)
            raise AudioValidationError("published WAV changed during atomic publication")
        self._published = True
        return (
            f"narrations/{self.job_id}/audio",
            published,
            StreamingAssemblyResult(tuple(self._timeline), self._current_frame),
        )

    def abort(self) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None
        self.temporary.unlink(missing_ok=True)
