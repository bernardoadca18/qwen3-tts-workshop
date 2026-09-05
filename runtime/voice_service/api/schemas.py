from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from voice_service.domain import ChunkCandidate, GenerationProfile, NarrationJob, VoiceProfile


class NarrationCreate(BaseModel):
    text: str
    language: str = "Portuguese"
    voice_profile: str | None = None
    generation_profile: str | None = None
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class BenchmarkNarrationCreate(BaseModel):
    """Internal benchmark request; the reference is relative to /data/references."""

    text: str
    language: str = "Portuguese"
    reference_id: str = Field(min_length=1, max_length=500)
    generation_profile: str | None = None
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class NarrationOut(BaseModel):
    id: str
    status: str
    created: bool | None = None
    created_at: str
    started_at: str | None
    completed_at: str | None
    text_length: int
    language: str
    voice_profile: str
    generation_profile: str
    resolved_seed: int
    current_chunk: int
    total_chunks: int
    model_id: str
    model_revision: str
    result_duration_seconds: float | None
    result_sample_rate: int | None
    result_channels: int | None
    result_subtype: str | None
    result_byte_size: int | None
    result_sha256: str | None
    result_frame_count: int | None
    result_riff_data_byte_size: int | None
    sample_peak_dbfs: float | None
    clipping_sample_count: int | None
    error_category: str | None
    error_message: str | None
    status_url: str
    audio_url: str
    metadata: dict[str, Any]

    @classmethod
    def from_job(cls, job: NarrationJob, *, created: bool | None = None) -> NarrationOut:
        return cls(
            id=job.id,
            status=job.status.value,
            created=created,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            text_length=len(job.text),
            language=job.language,
            voice_profile=job.voice_profile,
            generation_profile=job.generation_profile,
            resolved_seed=job.resolved_seed,
            current_chunk=job.current_chunk,
            total_chunks=job.total_chunks,
            model_id=job.model_id,
            model_revision=job.model_revision,
            result_duration_seconds=job.result_duration_seconds,
            result_sample_rate=job.result_sample_rate,
            result_channels=job.result_channels,
            result_subtype=job.result_subtype,
            result_byte_size=job.result_byte_size,
            result_sha256=job.result_sha256,
            result_frame_count=job.result_frame_count,
            result_riff_data_byte_size=job.result_riff_data_byte_size,
            sample_peak_dbfs=job.sample_peak_dbfs,
            clipping_sample_count=job.clipping_sample_count,
            error_category=job.error_category,
            error_message=job.error_message,
            status_url=f"/v1/narrations/{job.id}",
            audio_url=f"/v1/narrations/{job.id}/audio",
            metadata=job.metadata,
        )


class VoiceOut(BaseModel):
    name: str
    version: str
    reference_id: str
    clone_mode: str
    audio_sha256: str
    transcript_sha256: str | None
    duration_seconds: float
    sample_rate: int
    channels: int
    identity: str

    @classmethod
    def from_profile(cls, profile: VoiceProfile) -> VoiceOut:
        return cls(
            name=profile.name,
            version=profile.version,
            reference_id=profile.reference_id,
            clone_mode=profile.clone_mode,
            audio_sha256=profile.audio_sha256,
            transcript_sha256=profile.transcript_sha256,
            duration_seconds=profile.duration_seconds,
            sample_rate=profile.sample_rate,
            channels=profile.channels,
            identity=profile.identity,
        )


class GenerationProfileOut(BaseModel):
    name: str
    version: str
    identity: str

    @classmethod
    def from_profile(cls, profile: GenerationProfile) -> GenerationProfileOut:
        return cls(name=profile.name, version=profile.version, identity=profile.identity)


class LegacyGenerateRequest(BaseModel):
    text: str
    language: str = "Portuguese"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048


class SpeakerEmbeddingOut(BaseModel):
    """Speaker embedding extracted from audio."""

    dimension: int
    sample_rate: int
    extraction_method: str = "qwen_ecapa_tdnn"
    embedding: list[float]

    @classmethod
    def from_numpy(
        cls, embedding: np.ndarray, sample_rate: int
    ) -> SpeakerEmbeddingOut:
        return cls(
            dimension=len(embedding),
            sample_rate=sample_rate,
            embedding=embedding.tolist(),
        )


class ChunkArtifactOut(BaseModel):
    """Persisted chunk artifact with stable identity."""

    chunk_id: str
    job_id: str
    chunk_index: int
    text: str
    synthesis_text: str
    seed: int
    duration_seconds: float
    sample_rate: int
    byte_size: int
    sha256: str
    frame_count: int
    start_char: int
    end_char: int
    boundary: str
    chunking_identity: str
    chunking_strategy_version: str
    voice_identity: str
    generation_identity: str
    created_at: str
    audio_url: str


class ChunkRegenerateRequest(BaseModel):
    """Request to regenerate a single chunk with a new seed."""

    seed: int = Field(ge=0, le=2**31 - 1)


class ChunkRegenerateOut(BaseModel):
    """Response from regenerating a single chunk."""

    candidate_id: str
    job_id: str
    chunk_index: int
    text: str
    synthesis_text: str
    seed: int
    duration_seconds: float
    sample_rate: int
    byte_size: int
    sha256: str
    frame_count: int
    status: str
    created: bool
    created_at: str
    audio_url: str

    @classmethod
    def from_candidate(
        cls, candidate: ChunkCandidate, *, created: bool
    ) -> ChunkRegenerateOut:
        if (
            candidate.duration_seconds is None
            or candidate.sample_rate is None
            or candidate.byte_size is None
            or candidate.sha256 is None
            or candidate.frame_count is None
        ):
            raise ValueError("completed chunk candidate metadata is incomplete")
        return cls(
            candidate_id=candidate.id,
            job_id=candidate.job_id,
            chunk_index=candidate.chunk_index,
            text=candidate.text,
            synthesis_text=candidate.synthesis_text,
            seed=candidate.seed,
            duration_seconds=candidate.duration_seconds,
            sample_rate=candidate.sample_rate,
            byte_size=candidate.byte_size,
            sha256=candidate.sha256,
            frame_count=candidate.frame_count,
            status=candidate.status,
            created=created,
            created_at=candidate.created_at,
            audio_url=(
                f"/v1/narrations/{candidate.job_id}/chunk-candidates/"
                f"{candidate.id}/audio"
            ),
        )
