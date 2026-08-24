from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class TextChunk:
    index: int
    start_char: int
    end_char: int
    text: str
    boundary: str
    seed: int


@dataclass(frozen=True)
class AudioChunk:
    samples: NDArray[np.float32]
    sample_rate: int


@dataclass(frozen=True)
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    channels: int
    subtype: str
    byte_size: int
    sha256: str
    sample_peak_dbfs: float
    clipping_sample_count: int


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    version: str
    reference_audio: Path
    transcript: str
    clone_mode: str
    audio_sha256: str
    transcript_sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    identity: str


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    version: str
    parameters: dict[str, Any]
    identity: str


@dataclass
class NarrationJob:
    id: str
    text: str
    language: str
    voice_profile: str
    generation_profile: str
    resolved_seed: int
    requested_seed: int
    status: JobStatus
    created_at: str
    model_id: str
    model_revision: str
    voice_identity: str
    generation_identity: str
    request_fingerprint: str
    idempotency_key: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    current_chunk: int = 0
    total_chunks: int = 0
    output_uri: str | None = None
    result_duration_seconds: float | None = None
    result_sample_rate: int | None = None
    result_channels: int | None = None
    result_subtype: str | None = None
    result_byte_size: int | None = None
    result_sha256: str | None = None
    sample_peak_dbfs: float | None = None
    clipping_sample_count: int | None = None
    error_category: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
