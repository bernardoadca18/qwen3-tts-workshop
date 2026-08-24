from __future__ import annotations

import time
from collections.abc import Callable

from fastapi.testclient import TestClient


def wait_terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(300):
        response = client.get(f"/v1/narrations/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not become terminal")


def test_live_is_available_before_readiness_and_ready_after_lifespan(app_factory: Callable) -> None:
    app, provider = app_factory()
    client = TestClient(app)
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").status_code == 503
    with TestClient(app) as running:
        ready = running.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["worker_operational"] is True
        assert running.get("/health").json()["status"] == "ready"
        voices = running.get("/v1/voices").json()
        assert voices[0]["name"] == "test_voice"
        assert "reference_audio" not in voices[0]
        generations = running.get("/v1/generation-profiles").json()
        assert generations == [
            {
                "name": "test_generation",
                "version": "1",
                "identity": app.state.runtime.generations.get("test_generation").identity,
            }
        ]
    assert provider.initialized is False
    assert len(provider.prepared) == 1


def test_failed_initialization_cleans_up_provider_and_never_becomes_ready(
    app_factory: Callable,
) -> None:
    app, provider = app_factory(fail_initialize=True)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
    assert app.state.runtime.ready is False
    assert app.state.runtime.initialization_error == "initialization_failed"
    assert provider.shutdown_called is True


def test_create_status_audio_and_safe_metadata(app_factory: Callable) -> None:
    app, _ = app_factory()
    with TestClient(app) as client:
        created = client.post(
            "/v1/narrations",
            json={"text": "Narracao curta.", "seed": 42},
        )
        assert created.status_code == 202
        body = created.json()
        assert body["resolved_seed"] == 42
        assert body["created"] is True
        assert "text" not in body
        result = wait_terminal(client, body["id"])
        assert result["status"] == "completed"
        assert result["result_sample_rate"] == 24000
        assert result["result_subtype"] == "PCM_16"
        assert len(result["result_sha256"]) == 64
        assert result["current_chunk"] == result["total_chunks"]
        audio = client.get(result["audio_url"])
        assert audio.status_code == 200
        assert audio.content[:4] == b"RIFF"
        assert client.get("/v1/narrations/missing").status_code == 404
        assert client.get("/v1/narrations/missing/audio").status_code == 404


def test_idempotency_replays_seed_zero_and_conflicts_on_changed_payload(
    app_factory: Callable,
) -> None:
    app, _ = app_factory()
    headers = {"Idempotency-Key": "stable-key"}
    with TestClient(app) as client:
        first = client.post(
            "/v1/narrations", json={"text": "Mesmo texto.", "seed": 0}, headers=headers
        )
        second = client.post(
            "/v1/narrations", json={"text": "Mesmo texto.", "seed": 0}, headers=headers
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["id"] == second.json()["id"]
        assert first.json()["resolved_seed"] == second.json()["resolved_seed"]
        assert second.json()["created"] is False
        conflict = client.post(
            "/v1/narrations", json={"text": "Texto diferente.", "seed": 0}, headers=headers
        )
        assert conflict.status_code == 409


def test_provider_failure_is_persisted_without_internal_detail(app_factory: Callable) -> None:
    app, _ = app_factory(fail=True)
    with TestClient(app) as client:
        created = client.post("/v1/narrations", json={"text": "Falha segura."}).json()
        result = wait_terminal(client, created["id"])
        assert result["status"] == "failed"
        assert result["error_category"] == "provider_failure"
        assert result["error_message"] == "speech synthesis provider failed"
        assert "internal" not in result["error_message"]
        assert client.get(result["audio_url"]).status_code == 409


def test_worker_serializes_generation_and_queue_is_bounded(app_factory: Callable) -> None:
    app, provider = app_factory(delay=0.25)
    with TestClient(app) as client:
        responses = [
            client.post("/v1/narrations", json={"text": f"Job {index}."}) for index in range(4)
        ]
        accepted = [response for response in responses if response.status_code == 202]
        rejected = [response for response in responses if response.status_code == 503]
        assert len(accepted) == 3
        assert len(rejected) == 1
        for response in accepted:
            assert wait_terminal(client, response.json()["id"])["status"] == "completed"
    assert provider.maximum_active == 1


def test_validation_and_profile_errors_are_client_safe(app_factory: Callable) -> None:
    app, _ = app_factory()
    with TestClient(app) as client:
        assert client.post("/v1/narrations", json={"text": "   "}).status_code == 400
        assert (
            client.post(
                "/v1/narrations",
                json={"text": "ok", "voice_profile": "../reference.wav"},
            ).status_code
            == 400
        )
        assert client.post("/v1/narrations", json={"text": "x" * 1001}).status_code == 400


def test_legacy_generate_uses_shared_job_pipeline_and_rejects_raw_overrides(
    app_factory: Callable,
) -> None:
    app, provider = app_factory()
    with TestClient(app) as client:
        rejected = client.post("/generate", json={"text": "Teste.", "temperature": 0.2})
        assert rejected.status_code == 422
        response = client.post("/generate", json={"text": "Teste legado.", "seed": 55})
        assert response.status_code == 200
        assert response.headers["x-qwen-seed"] == "55"
        assert response.headers["deprecation"] == "true"
        assert response.content[:4] == b"RIFF"
    assert len(provider.calls) == 1
