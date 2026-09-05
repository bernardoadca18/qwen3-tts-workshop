from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from voice_service.core.config import Settings
from voice_service.domain import (
    AudioMetrics,
    ChunkCandidate,
    JobStatus,
    NarrationJob,
    VoiceProfile,
)
from voice_service.providers import ProviderError, ProviderOutOfMemoryError, TTSProvider
from voice_service.repositories import NarrationJobRepository
from voice_service.services.audio import (
    AtomicWavPublisher,
    AtomicWavStream,
    AudioAssembler,
    AudioValidationError,
    ChunkQualityGuardError,
    analyze_audio_chunk_signal,
    validate_audio_chunk_quality,
)
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    ProfileConfigurationError,
    VoiceProfileRegistry,
    resolve_ephemeral_xvector_reference,
)
from voice_service.services.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class NarrationClientError(ValueError):
    pass


class IdempotencyConflictError(NarrationClientError):
    pass


class NarrationQueueFullError(RuntimeError):
    pass


class ChunkRegenerationFailedError(RuntimeError):
    def __init__(self, candidate: ChunkCandidate):
        super().__init__(candidate.error_message or "chunk regeneration failed")
        self.candidate = candidate


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_seed(seed: int) -> int:
    if seed < 0 or seed > 2**31 - 1:
        raise NarrationClientError("seed must be between 0 and 2147483647")
    return secrets.randbelow(2**31 - 1) + 1 if seed == 0 else seed


class NarrationService:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: TTSProvider,
        repository: NarrationJobRepository,
        voices: VoiceProfileRegistry,
        generations: GenerationProfileRegistry,
        publisher: AtomicWavPublisher,
    ):
        self.settings = settings
        self.provider = provider
        self.repository = repository
        self.voices = voices
        self.generations = generations
        self.publisher = publisher
        self.chunker = TextChunker(
            target_chars=settings.target_chunk_chars,
            max_chars=settings.max_chunk_chars,
        )
        self.assembler = AudioAssembler(
            sentence_pause_ms=settings.sentence_pause_ms,
            paragraph_pause_ms=settings.paragraph_pause_ms,
            fallback_pause_ms=settings.fallback_pause_ms,
            hard_split_pause_ms=settings.hard_split_pause_ms,
        )
        self.queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(
            maxsize=settings.max_queue_size + 1
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._creation_lock = asyncio.Lock()
        self._stopping = False

    @property
    def worker_operational(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def start(self) -> None:
        self.publisher.initialize()
        cutoff = (
            datetime.now(UTC) - timedelta(hours=self.settings.job_retention_hours)
        ).isoformat()
        for job_id in self.repository.prune_terminal_before(cutoff):
            self.publisher.delete(job_id)
        self._worker_task = asyncio.create_task(self._worker(), name="qwen-tts-gpu-worker")
        for job in self.repository.recoverable():
            if job.status is JobStatus.RUNNING:
                metrics = await asyncio.to_thread(self.publisher.reconcile, job.id)
                if metrics is not None:
                    self.repository.mark_completed(
                        job.id,
                        completed_at=utc_now(),
                        output_uri=f"narrations/{job.id}/audio",
                        metrics=metrics,
                        metadata={**job.metadata, "recovered_from_atomic_output": True},
                    )
                    continue
                self.publisher.delete(job.id)
                self.repository.requeue(job.id)
            await self.queue.put(("narration", job.id))
        for candidate in self.repository.recoverable_chunk_candidates():
            await self.queue.put(("chunk_candidate", candidate.id))

    async def stop(self) -> None:
        self._stopping = True
        if self._worker_task is not None:
            await self.queue.put(None)
            await self._worker_task
            self._worker_task = None

    async def create(
        self,
        *,
        text: str,
        language: str,
        voice_profile_name: str | None,
        generation_profile_name: str | None,
        requested_seed: int,
        idempotency_key: str | None,
    ) -> tuple[NarrationJob, bool]:
        return await self._create(
            text=text,
            language=language,
            voice_profile_name=voice_profile_name,
            generation_profile_name=generation_profile_name,
            requested_seed=requested_seed,
            idempotency_key=idempotency_key,
        )

    async def create_benchmark(
        self,
        *,
        text: str,
        language: str,
        reference_id: str,
        generation_profile_name: str | None,
        requested_seed: int,
        idempotency_key: str | None,
    ) -> tuple[NarrationJob, bool]:
        try:
            voice = resolve_ephemeral_xvector_reference(
                reference_id,
                references_directory=self.settings.references_directory,
                model_id=self.settings.model_id,
                model_revision=self.settings.model_revision,
            )
        except ProfileConfigurationError as exc:
            raise NarrationClientError(str(exc)) from exc
        reference_metadata: dict[str, object] = {
            "reference_id": voice.reference_audio.relative_to(
                self.settings.references_directory.resolve()
            ).as_posix(),
            "reference_sha256": voice.audio_sha256,
            "reference_identity": voice.identity,
            "clone_mode": "x_vector_only",
        }
        return await self._create(
            text=text,
            language=language,
            voice_profile_name=None,
            generation_profile_name=generation_profile_name,
            requested_seed=requested_seed,
            idempotency_key=idempotency_key,
            voice_override=voice,
            benchmark_reference=reference_metadata,
        )

    async def _create(
        self,
        *,
        text: str,
        language: str,
        voice_profile_name: str | None,
        generation_profile_name: str | None,
        requested_seed: int,
        idempotency_key: str | None,
        voice_override: VoiceProfile | None = None,
        benchmark_reference: dict[str, object] | None = None,
    ) -> tuple[NarrationJob, bool]:
        if not text.strip():
            raise NarrationClientError("text must not be empty")
        if len(text) > self.settings.max_text_chars:
            raise NarrationClientError("text exceeds the configured maximum")
        if not language.strip() or len(language) > 64:
            raise NarrationClientError("language is invalid")
        voice_name = voice_profile_name or self.settings.default_voice_profile
        generation_name = generation_profile_name or self.settings.default_generation_profile
        try:
            voice = voice_override or self.voices.get(voice_name)
            generation = self.generations.get(generation_name)
        except ProfileConfigurationError as exc:
            raise NarrationClientError(str(exc)) from exc
        fingerprint = self._fingerprint(
            text=text,
            language=language,
            voice_identity=voice.identity,
            generation_identity=generation.identity,
            chunking_identity=self.chunker.identity,
            requested_seed=requested_seed,
        )
        async with self._creation_lock:
            if idempotency_key is not None:
                existing = self.repository.get_by_idempotency_key(idempotency_key)
                if existing is not None:
                    if existing.request_fingerprint != fingerprint:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was already used with different generation inputs"
                        )
                    return existing, False
            if self._stopping or not self.worker_operational:
                raise RuntimeError("narration worker is unavailable")
            if self.queue.qsize() >= self.settings.max_queue_size:
                raise NarrationQueueFullError("narration queue is full")
            resolved_seed = resolve_seed(requested_seed)
            profile_metadata: dict[str, object] = {
                "voice_profile_version": voice.version,
                "voice_reference_id": voice.reference_id,
                "voice_reference_sha256": voice.audio_sha256,
                "voice_transcript_sha256": voice.transcript_sha256,
                "voice_clone_mode": voice.clone_mode,
                "generation_profile_version": generation.version,
            }
            if benchmark_reference is not None:
                profile_metadata["benchmark_reference"] = benchmark_reference
            job = NarrationJob(
                id=uuid4().hex,
                text=text,
                language=language,
                voice_profile=voice.name,
                generation_profile=generation.name,
                resolved_seed=resolved_seed,
                requested_seed=requested_seed,
                status=JobStatus.QUEUED,
                created_at=utc_now(),
                model_id=self.settings.model_id,
                model_revision=self.settings.model_revision,
                voice_identity=voice.identity,
                generation_identity=generation.identity,
                chunking_identity=self.chunker.identity,
                request_fingerprint=fingerprint,
                idempotency_key=idempotency_key,
                metadata=profile_metadata,
            )
            self.repository.create(job)
            try:
                self.queue.put_nowait(("narration", job.id))
            except asyncio.QueueFull as exc:
                self.repository.delete(job.id)
                raise NarrationQueueFullError("narration queue is full") from exc
            return job, True

    def get(self, job_id: str) -> NarrationJob:
        job = self.repository.get(job_id)
        if job is None:
            raise LookupError("narration job not found")
        return job

    def verified_audio(self, job_id: str) -> tuple[Path, AudioMetrics]:
        job = self.get(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise AudioValidationError("narration audio is not complete")
        metrics = self.publisher.reconcile(job.id)
        if metrics is None:
            raise AudioValidationError("narration audio failed physical reconciliation")
        comparisons = (
            (job.result_sha256, metrics.sha256),
            (job.result_byte_size, metrics.byte_size),
            (job.result_sample_rate, metrics.sample_rate),
            (job.result_channels, metrics.channels),
            (job.result_subtype, metrics.subtype),
            (job.result_frame_count, metrics.frame_count),
            (job.result_riff_data_byte_size, metrics.riff_data_byte_size),
        )
        if any(expected is not None and expected != physical for expected, physical in comparisons):
            raise AudioValidationError("narration audio differs from persisted result metadata")
        if job.result_duration_seconds != metrics.duration_seconds:
            raise AudioValidationError("narration duration differs from persisted result metadata")
        return self.publisher.path_for(job.id), metrics

    def verified_chunk(self, job_id: str, chunk_index: int) -> tuple[Path, AudioMetrics]:
        job = self.get(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise AudioValidationError("narration audio is not complete")
        chunk = self._chunk_metadata(job, chunk_index)
        metrics = self.publisher.reconcile_chunk(job_id, chunk_index)
        if metrics is None:
            raise AudioValidationError("chunk audio failed physical reconciliation")
        expected = {
            "sha256": chunk.get("chunk_sha256"),
            "byte_size": chunk.get("chunk_byte_size"),
            "sample_rate": chunk.get("chunk_sample_rate"),
            "frame_count": chunk.get("chunk_frame_count"),
        }
        physical = {
            "sha256": metrics.sha256,
            "byte_size": metrics.byte_size,
            "sample_rate": metrics.sample_rate,
            "frame_count": metrics.frame_count,
        }
        if any(value is not None and value != physical[key] for key, value in expected.items()):
            raise AudioValidationError("chunk audio differs from persisted metadata")
        return self.publisher.chunk_path_for(job_id, chunk_index), metrics

    def get_chunk_candidate(self, candidate_id: str) -> ChunkCandidate:
        candidate = self.repository.get_chunk_candidate(candidate_id)
        if candidate is None:
            raise LookupError("chunk candidate not found")
        return candidate

    def verified_chunk_candidate(
        self, job_id: str, candidate_id: str
    ) -> tuple[Path, AudioMetrics]:
        candidate = self.get_chunk_candidate(candidate_id)
        if candidate.job_id != job_id:
            raise LookupError("chunk candidate not found")
        if candidate.status != "completed":
            raise AudioValidationError("chunk candidate audio is not complete")
        metrics = self.publisher.reconcile_chunk_candidate(job_id, candidate_id)
        if metrics is None:
            raise AudioValidationError("chunk candidate failed physical reconciliation")
        comparisons = (
            (candidate.sha256, metrics.sha256),
            (candidate.byte_size, metrics.byte_size),
            (candidate.sample_rate, metrics.sample_rate),
            (candidate.channels, metrics.channels),
            (candidate.subtype, metrics.subtype),
            (candidate.frame_count, metrics.frame_count),
            (candidate.riff_data_byte_size, metrics.riff_data_byte_size),
        )
        if any(expected is not None and expected != physical for expected, physical in comparisons):
            raise AudioValidationError("chunk candidate differs from persisted metadata")
        return self.publisher.chunk_candidate_path_for(job_id, candidate_id), metrics

    async def create_chunk_candidate(
        self,
        *,
        job_id: str,
        chunk_index: int,
        seed: int,
        idempotency_key: str,
    ) -> tuple[ChunkCandidate, bool]:
        if seed < 0 or seed > 2**31 - 1:
            raise NarrationClientError("seed must be between 0 and 2147483647")
        if not idempotency_key:
            raise NarrationClientError("Idempotency-Key is required for chunk regeneration")
        job = self.get(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise NarrationClientError("narration job is not completed")
        chunk = self._chunk_metadata(job, chunk_index)
        synthesis_text = chunk.get("synthesis_text")
        if not isinstance(synthesis_text, str) or not synthesis_text:
            raise NarrationClientError("chunk has no frozen synthesis text")
        text = chunk.get("text")
        if not isinstance(text, str):
            raise NarrationClientError("chunk has invalid canonical text")
        fingerprint = self._chunk_candidate_fingerprint(
            job=job,
            chunk_index=chunk_index,
            synthesis_text=synthesis_text,
            seed=seed,
        )
        created = False
        async with self._creation_lock:
            candidate = self.repository.get_chunk_candidate_by_idempotency_key(
                idempotency_key
            )
            if candidate is not None:
                if candidate.request_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "Idempotency-Key was already used with different chunk inputs"
                    )
            else:
                if self._stopping or not self.worker_operational:
                    raise RuntimeError("narration worker is unavailable")
                if self.queue.qsize() >= self.settings.max_queue_size:
                    raise NarrationQueueFullError("narration queue is full")
                candidate = ChunkCandidate(
                    id=uuid4().hex,
                    job_id=job.id,
                    chunk_index=chunk_index,
                    text=text,
                    synthesis_text=synthesis_text,
                    seed=seed,
                    voice_identity=job.voice_identity,
                    generation_identity=job.generation_identity,
                    request_fingerprint=fingerprint,
                    idempotency_key=idempotency_key,
                    status="queued",
                    created_at=utc_now(),
                )
                self.repository.create_chunk_candidate(candidate)
                self.queue.put_nowait(("chunk_candidate", candidate.id))
                created = True

        while candidate.status in {"queued", "running"}:
            await asyncio.sleep(0.01)
            candidate = self.get_chunk_candidate(candidate.id)
        if candidate.status == "failed":
            raise ChunkRegenerationFailedError(candidate)
        return candidate, created

    async def wait_terminal(self, job_id: str, *, timeout: float) -> NarrationJob:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            job = self.get(job_id)
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                return job
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("narration did not complete before the legacy timeout")
            await asyncio.sleep(0.05)

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                kind, artifact_id = item
                if kind == "narration":
                    await self._process(artifact_id)
                else:
                    await self._process_chunk_candidate(artifact_id)
            finally:
                self.queue.task_done()

    async def _process_chunk_candidate(self, candidate_id: str) -> None:
        candidate = self.repository.get_chunk_candidate(candidate_id)
        if candidate is None or candidate.status not in {"queued", "running"}:
            return
        try:
            existing = await asyncio.to_thread(
                self.publisher.reconcile_chunk_candidate,
                candidate.job_id,
                candidate.id,
            )
            if existing is not None:
                self.repository.mark_chunk_candidate_completed(
                    candidate.id,
                    completed_at=utc_now(),
                    output_uri=(
                        f"narrations/{candidate.job_id}/chunk-candidates/"
                        f"{candidate.id}/audio"
                    ),
                    metrics=existing,
                )
                return

            job = self.get(candidate.job_id)
            if job.status is not JobStatus.COMPLETED:
                raise NarrationClientError("narration job is not completed")
            voice = self._resolve_job_voice(job)
            generation = self.generations.get(job.generation_profile)
            if (
                voice.identity != candidate.voice_identity
                or generation.identity != candidate.generation_identity
            ):
                raise ProfileConfigurationError(
                    "persisted candidate profile identity no longer matches configuration"
                )
            _, original_before = await asyncio.to_thread(
                self.verified_chunk,
                job.id,
                candidate.chunk_index,
            )

            self.repository.mark_chunk_candidate_running(
                candidate.id,
                started_at=utc_now(),
            )
            await self.provider.prepare_voice(voice)
            audio = await self.provider.generate(
                text=candidate.synthesis_text,
                language=job.language,
                voice_profile=voice,
                generation_profile=generation,
                seed=candidate.seed,
            )
            validate_audio_chunk_quality(
                audio,
                text=candidate.synthesis_text,
                max_duration_seconds=self.settings.chunk_max_duration_seconds,
                max_seconds_per_char=self.settings.chunk_max_seconds_per_char,
                duration_grace_seconds=self.settings.chunk_duration_grace_seconds,
                min_peak_amplitude=self.settings.chunk_min_peak_amplitude,
            )
            output_uri, metrics = await asyncio.to_thread(
                self.publisher.publish_chunk_candidate,
                job.id,
                candidate.id,
                audio,
            )
            _, original_after = await asyncio.to_thread(
                self.verified_chunk,
                job.id,
                candidate.chunk_index,
            )
            if original_after != original_before:
                raise AudioValidationError("original chunk changed during regeneration")
            self.repository.mark_chunk_candidate_completed(
                candidate.id,
                completed_at=utc_now(),
                output_uri=output_uri,
                metrics=metrics,
            )
        except ChunkQualityGuardError:
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="chunk_quality_guard",
                message="generated speech chunk failed quality validation",
            )
        except ProviderOutOfMemoryError:
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="cuda_oom",
                message="GPU capacity was exhausted",
            )
        except ProviderError:
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="provider_failure",
                message="speech synthesis provider failed",
            )
        except AudioValidationError:
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="audio_validation",
                message="generated audio failed validation",
            )
        except (NarrationClientError, ProfileConfigurationError):
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="profile_configuration",
                message="persisted regeneration inputs are unavailable",
            )
        except OSError:
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="storage_failure",
                message="chunk candidate could not be stored",
            )
        except Exception:
            logger.exception("chunk candidate generation failed candidate_id=%s", candidate.id)
            self.repository.mark_chunk_candidate_failed(
                candidate.id,
                completed_at=utc_now(),
                category="synthesis_failure",
                message="chunk regeneration failed",
            )

    async def _process(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return
        started = time.monotonic()
        stream: AtomicWavStream | None = None
        chunk_diagnostics: list[dict[str, object]] = []
        try:
            voice = self._resolve_job_voice(job)
            generation = self.generations.get(job.generation_profile)
            if (
                voice.identity != job.voice_identity
                or generation.identity != job.generation_identity
                or job.chunking_identity != self.chunker.identity
            ):
                raise ProfileConfigurationError(
                    "persisted profile identity no longer matches configuration"
                )
            await self.provider.prepare_voice(voice)
            chunks = self.chunker.chunk(job.text, base_seed=job.resolved_seed)
            self.repository.mark_running(job.id, started_at=utc_now(), total_chunks=len(chunks))
            stream = self.publisher.begin_stream(job.id, self.assembler)
            for index, chunk in enumerate(chunks, start=1):
                chunk_started = time.monotonic()
                audio = await self.provider.generate(
                    text=chunk.text,
                    language=job.language,
                    voice_profile=voice,
                    generation_profile=generation,
                    seed=chunk.seed,
                )
                synthesis_seconds = time.monotonic() - chunk_started
                try:
                    quality = validate_audio_chunk_quality(
                        audio,
                        text=chunk.text,
                        max_duration_seconds=self.settings.chunk_max_duration_seconds,
                        max_seconds_per_char=self.settings.chunk_max_seconds_per_char,
                        duration_grace_seconds=self.settings.chunk_duration_grace_seconds,
                        min_peak_amplitude=self.settings.chunk_min_peak_amplitude,
                    )
                except ChunkQualityGuardError as exc:
                    rejected = {
                        "index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "boundary": chunk.boundary,
                        "seed": chunk.seed,
                        "text_length_chars": len(chunk.text),
                        "audio_duration_seconds": exc.metrics.audio_duration_seconds,
                        "synthesis_seconds": synthesis_seconds,
                        "real_time_factor": (
                            synthesis_seconds / exc.metrics.audio_duration_seconds
                            if exc.metrics.audio_duration_seconds > 0
                            else None
                        ),
                        "quality_guard_status": "rejected",
                        "quality_guard_reason": exc.reason,
                        "maximum_allowed_duration_seconds": (
                            exc.metrics.maximum_allowed_duration_seconds
                        ),
                    }
                    logger.warning(
                        "chunk rejected by quality guard job_id=%s chunk_index=%s "
                        "text_length_chars=%s audio_duration_seconds=%.3f seed=%s reason=%s",
                        job.id,
                        chunk.index,
                        len(chunk.text),
                        exc.metrics.audio_duration_seconds,
                        chunk.seed,
                        exc.reason,
                    )
                    exc.diagnostics = rejected
                    raise
                signal_diagnostics = analyze_audio_chunk_signal(audio)

                # Persist chunk WAV with stable identity
                chunk_uri, chunk_metrics = await asyncio.to_thread(
                    self.publisher.publish_chunk,
                    job.id,
                    chunk.index,
                    audio,
                )

                await asyncio.to_thread(
                    stream.append,
                    chunk,
                    audio,
                    append_pause=index < len(chunks),
                )
                chunk_diagnostics.append(
                    {
                        "index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "boundary": chunk.boundary,
                        "pause_after_frames": (
                            self.assembler.pause_frames_for(
                                chunk.boundary, chunk_metrics.sample_rate
                            )
                            if index < len(chunks)
                            else 0
                        ),
                        "seed": chunk.seed,
                        "text": chunk.text,
                        "synthesis_text": chunk.text,
                        "chunk_uri": chunk_uri,
                        "chunk_duration_seconds": chunk_metrics.duration_seconds,
                        "chunk_byte_size": chunk_metrics.byte_size,
                        "chunk_sha256": chunk_metrics.sha256,
                        "chunk_sample_rate": chunk_metrics.sample_rate,
                        "chunk_frame_count": chunk_metrics.frame_count,
                        "chunk_channels": chunk_metrics.channels,
                        "chunk_subtype": chunk_metrics.subtype,
                        "chunk_riff_data_byte_size": chunk_metrics.riff_data_byte_size,
                        "text_length_chars": quality.text_length_chars,
                        "audio_duration_seconds": quality.audio_duration_seconds,
                        "synthesis_seconds": synthesis_seconds,
                        "real_time_factor": synthesis_seconds / quality.audio_duration_seconds,
                        "quality_guard_status": "passed",
                        "maximum_allowed_duration_seconds": (
                            quality.maximum_allowed_duration_seconds
                        ),
                        "signal_diagnostics": signal_diagnostics,
                    }
                )
                self.repository.update_progress(job.id, current_chunk=index)
                logger.info(
                    "chunk synthesized job_id=%s voice_profile=%s "
                    "generation_profile=%s seed=%s chunk=%s/%s chars=%s",
                    job.id,
                    job.voice_profile,
                    job.generation_profile,
                    chunk.seed,
                    index,
                    len(chunks),
                    len(chunk.text),
                )
            output_uri, metrics, assembly = await asyncio.to_thread(stream.publish)
            if metrics.frame_count != assembly.expected_frame_count:
                self.publisher.delete(job.id)
                raise AudioValidationError(
                    "published narration sample count differs from assembled sample count"
                )
            elapsed = time.monotonic() - started
            metadata: dict[str, object] = {
                **job.metadata,
                "provider": self.provider.metadata().provider,
                "model_id": job.model_id,
                "model_revision": job.model_revision,
                "voice_identity": job.voice_identity,
                "voice_clone_mode": voice.clone_mode,
                "generation_identity": job.generation_identity,
                "chunking_strategy_version": self.chunker.strategy_version,
                "chunking_identity": self.chunker.identity,
                "chunking": {
                    "target_chars": self.chunker.target_chars,
                    "max_chars": self.chunker.max_chars,
                },
                "chunk_seed_strategy": "sha256:qwen3-tts-chunk-v1",
                "chunks": chunk_diagnostics,
                "assembly_timeline": list(assembly.timeline),
                "artifact_integrity": {
                    "status": "passed",
                    "expected_frame_count": assembly.expected_frame_count,
                    "physical": asdict(metrics),
                    "expected_matches_physical": assembly.expected_frame_count
                    == metrics.frame_count,
                },
                "synthesis_seconds": elapsed,
                "real_time_factor": elapsed / metrics.duration_seconds,
            }
            self.repository.mark_completed(
                job.id,
                completed_at=utc_now(),
                output_uri=output_uri,
                metrics=metrics,
                metadata=metadata,
            )
            logger.info(
                "narration completed job_id=%s voice_profile=%s "
                "generation_profile=%s seed=%s chunks=%s "
                "synthesis_seconds=%.3f output_seconds=%.3f "
                "rtf=%.3f sample_rate=%s",
                job.id,
                job.voice_profile,
                job.generation_profile,
                job.resolved_seed,
                len(chunks),
                elapsed,
                metrics.duration_seconds,
                elapsed / metrics.duration_seconds,
                metrics.sample_rate,
            )
        except ChunkQualityGuardError as exc:
            self.publisher.delete(job.id)
            self._fail(
                job,
                "chunk_quality_guard",
                "generated speech chunk failed quality validation",
                metadata={
                    "provider": self.provider.metadata().provider,
                    "model_id": job.model_id,
                    "model_revision": job.model_revision,
                    "voice_identity": job.voice_identity,
                    "generation_identity": job.generation_identity,
                    "chunking_strategy_version": self.chunker.strategy_version,
                    "chunking_identity": self.chunker.identity,
                    "quality_guard_status": "rejected",
                    "rejected_chunk": exc.diagnostics,
                    "chunks": [*chunk_diagnostics, exc.diagnostics],
                },
            )
        except ProviderOutOfMemoryError:
            self._fail(job, "cuda_oom", "GPU capacity was exhausted")
        except ProviderError:
            self._fail(job, "provider_failure", "speech synthesis provider failed")
        except AudioValidationError:
            self._fail(job, "audio_validation", "generated audio failed validation")
        except ProfileConfigurationError:
            self._fail(job, "profile_configuration", "persisted profile is unavailable")
        except OSError:
            self._fail(job, "storage_failure", "audio result could not be stored")
        except Exception:
            self._fail(job, "synthesis_failure", "narration generation failed")
        finally:
            if stream is not None:
                stream.abort()

    def _fail(
        self,
        job: NarrationJob,
        category: str,
        public_message: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        logger.exception(
            "narration failed job_id=%s voice_profile=%s generation_profile=%s seed=%s category=%s",
            job.id,
            job.voice_profile,
            job.generation_profile,
            job.resolved_seed,
            category,
        )
        persisted_metadata = dict(job.metadata)
        if metadata is not None:
            persisted_metadata.update(metadata)
        self.repository.mark_failed(
            job.id,
            completed_at=utc_now(),
            category=category,
            message=public_message,
            metadata=persisted_metadata or None,
        )

    def _resolve_job_voice(self, job: NarrationJob) -> VoiceProfile:
        benchmark = job.metadata.get("benchmark_reference")
        if benchmark is None:
            return self.voices.get(job.voice_profile)
        if not isinstance(benchmark, dict):
            raise ProfileConfigurationError("benchmark reference metadata is invalid")
        reference_id = benchmark.get("reference_id")
        if not isinstance(reference_id, str):
            raise ProfileConfigurationError("benchmark reference metadata is incomplete")
        voice = resolve_ephemeral_xvector_reference(
            reference_id,
            references_directory=self.settings.references_directory,
            model_id=self.settings.model_id,
            model_revision=self.settings.model_revision,
        )
        if voice.identity != job.voice_identity:
            raise ProfileConfigurationError(
                "benchmark reference identity no longer matches request"
            )
        return voice

    def _fingerprint(
        self,
        *,
        text: str,
        language: str,
        voice_identity: str,
        generation_identity: str,
        chunking_identity: str,
        requested_seed: int,
    ) -> str:
        payload = {
            "text": text,
            "language": language,
            "voice_identity": voice_identity,
            "generation_identity": generation_identity,
            "chunking_identity": chunking_identity,
            "requested_seed": requested_seed,
            "model_id": self.settings.model_id,
            "model_revision": self.settings.model_revision,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_metadata(job: NarrationJob, chunk_index: int) -> dict[str, object]:
        chunks = job.metadata.get("chunks")
        if not isinstance(chunks, list):
            raise NarrationClientError("narration job has no physical chunks")
        for chunk in chunks:
            if isinstance(chunk, dict) and chunk.get("index") == chunk_index:
                return chunk
        raise NarrationClientError("physical chunk not found")

    def _chunk_candidate_fingerprint(
        self,
        *,
        job: NarrationJob,
        chunk_index: int,
        synthesis_text: str,
        seed: int,
    ) -> str:
        payload = {
            "job_id": job.id,
            "chunk_index": chunk_index,
            "synthesis_text": synthesis_text,
            "seed": seed,
            "voice_identity": job.voice_identity,
            "generation_identity": job.generation_identity,
            "model_id": job.model_id,
            "model_revision": job.model_revision,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
