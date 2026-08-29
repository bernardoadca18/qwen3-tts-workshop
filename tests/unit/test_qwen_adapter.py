from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from voice_service.domain import VoiceProfile
from voice_service.providers.qwen import QwenTTSProvider


def test_qwen_adapter_import_does_not_import_qwen_or_torch() -> None:
    assert "qwen_tts" not in sys.modules


def test_initialize_resolves_pinned_snapshot_before_loading(
    monkeypatch, configured_settings
) -> None:
    calls = {}
    hub = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, revision):
        calls["snapshot"] = (repo_id, revision)
        return "/models/pinned-snapshot"

    hub.snapshot_download = snapshot_download
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    qwen = ModuleType("qwen_tts")

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["load"] = (path, kwargs)
            return cls()

    qwen.Qwen3TTSModel = FakeModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen)
    provider = QwenTTSProvider(configured_settings)
    provider._initialize_sync()
    assert calls["snapshot"] == (
        configured_settings.model_id,
        configured_settings.model_revision,
    )
    assert calls["load"][0] == "/models/pinned-snapshot"
    assert calls["load"][1]["device_map"] == "cpu"


def _voice(
    name: str,
    mode: str,
    transcript: str | None,
    *,
    identity: str | None = None,
    ephemeral: bool = False,
) -> VoiceProfile:
    return VoiceProfile(
        name=name,
        version="1",
        reference_audio=Path("/reference.wav"),
        transcript=transcript,
        clone_mode=mode,
        audio_sha256="a" * 64,
        transcript_sha256="b" * 64 if transcript else None,
        duration_seconds=10.0,
        sample_rate=24000,
        channels=1,
        identity=identity or ("c" if mode == "icl" else "d") * 64,
        ephemeral=ephemeral,
    )


def test_prompt_preparation_keeps_icl_and_xvector_semantics_separate(
    configured_settings,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        def create_voice_clone_prompt(self, **kwargs):
            calls.append(kwargs)
            return {"call": len(calls)}

    provider = QwenTTSProvider(configured_settings)
    provider._initialized = True
    provider._model = FakeModel()
    icl = _voice("icl", "icl", "Texto de referência.")
    xvec = _voice("xvec", "x_vector_only", None)
    provider._prepare_voice_sync(icl)
    provider._prepare_voice_sync(xvec)
    provider._prepare_voice_sync(xvec)

    assert calls == [
        {
            "ref_audio": "/reference.wav",
            "ref_text": "Texto de referência.",
            "x_vector_only_mode": False,
        },
        {
            "ref_audio": "/reference.wav",
            "ref_text": None,
            "x_vector_only_mode": True,
        },
    ]
    assert provider._voice_prompts[icl.identity] != provider._voice_prompts[xvec.identity]


def test_ephemeral_prompt_cache_is_lru_bounded_while_registered_profiles_stay_pinned(
    configured_settings,
) -> None:
    class FakeModel:
        def create_voice_clone_prompt(self, **kwargs):
            return kwargs["ref_audio"] + kwargs["x_vector_only_mode"].__str__()

    settings = configured_settings.model_copy(update={"max_ephemeral_prompt_cache_entries": 2})
    provider = QwenTTSProvider(settings)
    provider._initialized = True
    provider._model = FakeModel()
    pinned = _voice("pinned", "icl", "Texto.", identity="p" * 64)
    ephemeral = [
        _voice(
            f"ephemeral-{index}",
            "x_vector_only",
            None,
            identity=str(index) * 64,
            ephemeral=True,
        )
        for index in range(1, 4)
    ]

    provider._prepare_voice_sync(pinned)
    provider._prepare_voice_sync(ephemeral[0])
    provider._prepare_voice_sync(ephemeral[1])
    provider._prepare_voice_sync(ephemeral[0])
    provider._prepare_voice_sync(ephemeral[2])

    assert pinned.identity in provider._voice_prompts
    assert list(provider._ephemeral_voice_prompts) == [
        ephemeral[0].identity,
        ephemeral[2].identity,
    ]
    assert provider.prompt_cache_diagnostics() == {
        "policy": "registered_pinned_ephemeral_lru_v1",
        "entries": 3,
        "pinned_entries": 1,
        "registered_entries": 1,
        "ephemeral_entries": 2,
        "maximum_ephemeral_entries": 2,
        "ephemeral_limit": 2,
        "active_entries": 0,
        "hits": 1,
        "misses": 4,
        "evictions": 1,
        "estimated_bytes": None,
        "memory_measurement": "unavailable_without_tensor_introspection",
    }


def test_active_ephemeral_prompt_is_not_evicted(configured_settings) -> None:
    class FakeModel:
        def create_voice_clone_prompt(self, **kwargs):
            return object()

    settings = configured_settings.model_copy(update={"max_ephemeral_prompt_cache_entries": 1})
    provider = QwenTTSProvider(settings)
    provider._initialized = True
    provider._model = FakeModel()
    first = _voice(
        "first", "x_vector_only", None, identity="1" * 64, ephemeral=True
    )
    second = _voice(
        "second", "x_vector_only", None, identity="2" * 64, ephemeral=True
    )
    provider._prepare_voice_sync(first)
    provider._acquire_prompt(first.identity)
    provider._prepare_voice_sync(second)

    assert first.identity in provider._ephemeral_voice_prompts
    assert second.identity in provider._ephemeral_voice_prompts
    provider._release_prompt(first.identity)
    assert list(provider._ephemeral_voice_prompts) == [second.identity]
