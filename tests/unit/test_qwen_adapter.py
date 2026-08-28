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


def _voice(name: str, mode: str, transcript: str | None) -> VoiceProfile:
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
        identity=("c" if mode == "icl" else "d") * 64,
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
