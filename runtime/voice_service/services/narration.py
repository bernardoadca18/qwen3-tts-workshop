from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from voice_service.core.config import Settings
from voice_service.domain import JobStatus, NarrationJob
from voice_service.providers import ProviderError, ProviderOutOfMemoryError, TTSProvider
from voice_service.repositories import NarrationJobRepository
from voice_service.services.audio import (
    AtomicWavPublisher,
    AudioAssembler,
    AudioValidationError,
)
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    ProfileConfigurationError,
    VoiceProfileRegistry,
)
from voice_service.services.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class NarrationClientError(ValueError):
    pass


class IdempotencyConflictError(NarrationClientError):
    pass


class NarrationQueueFullError(RuntimeError):
    pass


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
        self.queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=settings.max_queue_size + 1)
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
            await self.queue.put(job.id)

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
        if not text.strip():
            raise NarrationClientError("text must not be empty")
        if len(text) > self.settings.max_text_chars:
            raise NarrationClientError("text exceeds the configured maximum")
        if not language.strip() or len(language) > 64:
            raise NarrationClientError("language is invalid")
        voice_name = voice_profile_name or self.settings.default_voice_profile
        generation_name = generation_profile_name or self.settings.default_generation_profile
        try:
            voice = self.voices.get(voice_name)
            generation = self.generations.get(generation_name)
        except ProfileConfigurationError as exc:
            raise NarrationClientError(str(exc)) from exc
        fingerprint = self._fingerprint(
            text=text,
            language=language,
            voice_identity=voice.identity,
            generation_identity=generation.identity,
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
                request_fingerprint=fingerprint,
                idempotency_key=idempotency_key,
            )
            self.repository.create(job)
            try:
                self.queue.put_nowait(job.id)
            except asyncio.QueueFull as exc:
                self.repository.delete(job.id)
                raise NarrationQueueFullError("narration queue is full") from exc
            return job, True

    def get(self, job_id: str) -> NarrationJob:
        job = self.repository.get(job_id)
        if job is None:
            raise LookupError("narration job not found")
        return job

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
            job_id = await self.queue.get()
            try:
                if job_id is None:
                    return
                await self._process(job_id)
            finally:
                self.queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return
        started = time.monotonic()
        try:
            voice = self.voices.get(job.voice_profile)
            generation = self.generations.get(job.generation_profile)
            if (
                voice.identity != job.voice_identity
                or generation.identity != job.generation_identity
            ):
                raise ProfileConfigurationError(
                    "persisted profile identity no longer matches configuration"
                )
            chunks = self.chunker.chunk(job.text, base_seed=job.resolved_seed)
            self.repository.mark_running(job.id, started_at=utc_now(), total_chunks=len(chunks))
            audio_chunks = []
            chunk_timings = []
            for index, chunk in enumerate(chunks, start=1):
                chunk_started = time.monotonic()
                audio = await self.provider.generate(
                    text=chunk.text,
                    language=job.language,
                    voice_profile=voice,
                    generation_profile=generation,
                    seed=chunk.seed,
                )
                audio_chunks.append(audio)
                chunk_timings.append(time.monotonic() - chunk_started)
                self.repository.update_progress(job.id, current_chunk=index)
                logger.info(
                    "chunk synthesized job_id=%s voice_profile=%s "
                    "generation_profile=%s seed=%s chunk=%s/%s chars=%s",
                    job.id,
                    job.voice_profile,
                    job.generation_profile,
                    job.resolved_seed,
                    index,
                    len(chunks),
                    len(chunk.text),
                )
            assembled = self.assembler.assemble(chunks, audio_chunks)
            output_uri, metrics = await asyncio.to_thread(self.publisher.publish, job.id, assembled)
            elapsed = time.monotonic() - started
            metadata: dict[str, object] = {
                "provider": self.provider.metadata().provider,
                "model_id": job.model_id,
                "model_revision": job.model_revision,
                "voice_identity": job.voice_identity,
                "generation_identity": job.generation_identity,
                "chunk_seed_strategy": "sha256:qwen3-tts-chunk-v1",
                "chunks": [
                    {
                        "index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "boundary": chunk.boundary,
                        "seed": chunk.seed,
                        "synthesis_seconds": chunk_timings[chunk.index],
                    }
                    for chunk in chunks
                ],
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

    def _fail(self, job: NarrationJob, category: str, public_message: str) -> None:
        logger.exception(
            "narration failed job_id=%s voice_profile=%s generation_profile=%s seed=%s category=%s",
            job.id,
            job.voice_profile,
            job.generation_profile,
            job.resolved_seed,
            category,
        )
        self.repository.mark_failed(
            job.id,
            completed_at=utc_now(),
            category=category,
            message=public_message,
        )

    def _fingerprint(
        self,
        *,
        text: str,
        language: str,
        voice_identity: str,
        generation_identity: str,
        requested_seed: int,
    ) -> str:
        payload = {
            "text": text,
            "language": language,
            "voice_identity": voice_identity,
            "generation_identity": generation_identity,
            "requested_seed": requested_seed,
            "model_id": self.settings.model_id,
            "model_revision": self.settings.model_revision,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
