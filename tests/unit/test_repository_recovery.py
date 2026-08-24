from __future__ import annotations

import time
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from voice_service.core.config import Settings
from voice_service.domain import AudioChunk, JobStatus, NarrationJob
from voice_service.main import create_app
from voice_service.repositories import NarrationJobRepository
from voice_service.services.audio import AtomicWavPublisher
from voice_service.services.narration import utc_now
from voice_service.services.profiles import (
    GenerationProfileRegistry,
    VoiceProfileRegistry,
)

from tests.fakes import FakeTTSProvider


def make_job(settings: Settings, *, status: JobStatus, seed: int = 777) -> NarrationJob:
    voice = VoiceProfileRegistry.load(
        settings.voice_profiles_file,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
    ).get(settings.default_voice_profile)
    generation = GenerationProfileRegistry.load(settings.generation_profiles_file).get(
        settings.default_generation_profile
    )
    return NarrationJob(
        id=uuid4().hex,
        text="Texto recuperado depois do reinicio.",
        language="Portuguese",
        voice_profile=voice.name,
        generation_profile=generation.name,
        resolved_seed=seed,
        requested_seed=seed,
        status=status,
        created_at=utc_now(),
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        voice_identity=voice.identity,
        generation_identity=generation.identity,
        request_fingerprint=uuid4().hex,
    )


def wait_completed(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        response = client.get(f"/v1/narrations/{job_id}")
        if response.status_code != 200:
            time.sleep(0.01)
            continue
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_repository_survives_reopen_and_lists_recoverable(
    configured_settings: Settings,
) -> None:
    first = NarrationJobRepository(configured_settings.job_database_path)
    first.initialize()
    queued = make_job(configured_settings, status=JobStatus.QUEUED)
    running = make_job(configured_settings, status=JobStatus.QUEUED)
    first.create(queued)
    first.create(running)
    first.mark_running(running.id, started_at=utc_now(), total_chunks=2)
    reopened = NarrationJobRepository(configured_settings.job_database_path)
    reopened.initialize()
    assert reopened.get(queued.id) == queued
    assert [job.id for job in reopened.recoverable()] == [queued.id, running.id]


def test_running_job_is_requeued_with_same_seed(
    configured_settings: Settings,
) -> None:
    repository = NarrationJobRepository(configured_settings.job_database_path)
    repository.initialize()
    job = make_job(configured_settings, status=JobStatus.QUEUED, seed=991)
    repository.create(job)
    repository.mark_running(job.id, started_at=utc_now(), total_chunks=1)
    provider = FakeTTSProvider(configured_settings)
    app = create_app(
        settings=configured_settings,
        provider_factory=lambda _: provider,
    )
    with TestClient(app) as client:
        result = wait_completed(client, job.id)
    assert result["status"] == "completed"
    assert result["resolved_seed"] == 991
    assert provider.calls


def test_running_job_with_atomic_output_is_reconciled_without_generation(
    configured_settings: Settings,
) -> None:
    repository = NarrationJobRepository(configured_settings.job_database_path)
    repository.initialize()
    job = make_job(configured_settings, status=JobStatus.QUEUED)
    repository.create(job)
    repository.mark_running(job.id, started_at=utc_now(), total_chunks=1)
    publisher = AtomicWavPublisher(
        configured_settings.output_directory,
        max_duration_seconds=configured_settings.max_result_duration_seconds,
    )
    phase = np.arange(2400, dtype=np.float32)
    publisher.publish(
        job.id,
        AudioChunk(0.1 * np.sin(2 * np.pi * 220 * phase / 24000), 24000),
    )
    provider = FakeTTSProvider(configured_settings)
    app = create_app(settings=configured_settings, provider_factory=lambda _: provider)
    with TestClient(app) as client:
        result = wait_completed(client, job.id)
    assert result["status"] == "completed"
    assert result["metadata"]["recovered_from_atomic_output"] is True
    assert provider.calls == []
