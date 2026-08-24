from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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
