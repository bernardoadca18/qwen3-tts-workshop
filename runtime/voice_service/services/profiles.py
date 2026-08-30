from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, Field

from voice_service.domain import GenerationProfile, VoiceProfile


class ProfileConfigurationError(ValueError):
    pass


EPHEMERAL_REFERENCE_VERSION = "benchmark_reference_v1"


class GenerationControls(BaseModel):
    do_sample: bool = True
    temperature: float = Field(default=0.9, gt=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=50, ge=1, le=500)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2)
    subtalker_dosample: bool = True
    subtalker_temperature: float = Field(default=0.9, gt=0, le=2)
    subtalker_top_p: float = Field(default=1.0, gt=0, le=1)
    subtalker_top_k: int = Field(default=50, ge=1, le=500)
    max_new_tokens: int = Field(default=2048, ge=64, le=8192)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference_id(value: object, audio_path: Path) -> str:
    reference_id = str(value or audio_path.name).strip()
    relative = Path(reference_id)
    if (
        not reference_id
        or relative.is_absolute()
        or "\x00" in reference_id
        or "\\" in reference_id
        or ".." in relative.parts
    ):
        raise ProfileConfigurationError("voice profile reference_id must be a safe relative path")
    return relative.as_posix()


def _expected_reference_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProfileConfigurationError(f"voice profile reference_sha256 is invalid: {name}")
    return digest


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileConfigurationError(f"cannot read profile configuration: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProfileConfigurationError(f"profile configuration must be a mapping: {path.name}")
    return value


class VoiceProfileRegistry:
    def __init__(self, profiles: dict[str, VoiceProfile]):
        self._profiles = profiles

    @classmethod
    def load(cls, path: Path, *, model_id: str, model_revision: str) -> VoiceProfileRegistry:
        raw = _read_yaml(path).get("voice_profiles")
        if not isinstance(raw, dict) or not raw:
            raise ProfileConfigurationError("voice_profiles must be a non-empty mapping")
        profiles: dict[str, VoiceProfile] = {}
        for name, config in raw.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                raise ProfileConfigurationError("invalid voice profile entry")
            profiles[name] = cls._load_one(name, config, model_id, model_revision)
        return cls(profiles)

    @staticmethod
    def _load_one(
        name: str,
        config: dict[str, Any],
        model_id: str,
        model_revision: str,
        *,
        ephemeral: bool = False,
    ) -> VoiceProfile:
        version = str(config.get("version", "")).strip()
        audio_path = Path(str(config.get("reference_audio", "")))
        raw_text_path = config.get("reference_text")
        text_path = Path(str(raw_text_path)) if raw_text_path else None
        clone_mode = str(config.get("clone_mode", "icl")).strip()
        reference_id = _reference_id(config.get("reference_id"), audio_path)
        expected_audio_hash = _expected_reference_sha256(config.get("reference_sha256"), name)
        if not version or clone_mode not in {"icl", "x_vector_only"}:
            raise ProfileConfigurationError(f"invalid voice profile metadata: {name}")
        if not audio_path.is_file() or (
            clone_mode == "icl" and (text_path is None or not text_path.is_file())
        ):
            raise ProfileConfigurationError(f"voice profile files are missing: {name}")
        try:
            transcript = (
                text_path.read_text(encoding="utf-8").strip()
                if clone_mode == "icl" and text_path is not None
                else None
            )
            info = sf.info(str(audio_path))
            decoded_frames = 0
            with sf.SoundFile(str(audio_path), mode="r") as source:
                for samples in source.blocks(
                    blocksize=262_144,
                    dtype="float32",
                    always_2d=True,
                ):
                    if not np.all(np.isfinite(samples)):
                        raise ProfileConfigurationError(
                            f"voice profile audio has non-finite samples: {name}"
                        )
                    decoded_frames += len(samples)
        except (OSError, RuntimeError, UnicodeError) as exc:
            raise ProfileConfigurationError(f"voice profile is unreadable: {name}") from exc
        if clone_mode == "icl" and not transcript:
            raise ProfileConfigurationError(f"voice profile transcript is empty: {name}")
        if decoded_frames <= 0 or info.samplerate <= 0 or info.frames <= 0:
            raise ProfileConfigurationError(f"voice profile audio is empty: {name}")
        audio_hash = _sha256_file(audio_path)
        if expected_audio_hash is not None and audio_hash != expected_audio_hash:
            raise ProfileConfigurationError(f"voice profile reference SHA-256 mismatch: {name}")
        transcript_hash = _sha256_text(transcript) if transcript is not None else None
        identity = _identity(
            {
                "name": name,
                "version": version,
                "audio_sha256": audio_hash,
                "transcript_sha256": transcript_hash,
                "clone_mode": clone_mode,
                "model_id": model_id,
                "model_revision": model_revision,
            }
        )
        return VoiceProfile(
            name=name,
            version=version,
            reference_id=reference_id,
            reference_audio=audio_path,
            transcript=transcript,
            clone_mode=clone_mode,
            audio_sha256=audio_hash,
            transcript_sha256=transcript_hash,
            duration_seconds=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            identity=identity,
            ephemeral=ephemeral,
        )

    def get(self, name: str) -> VoiceProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise ProfileConfigurationError("unknown voice profile") from exc

    def all(self) -> list[VoiceProfile]:
        return [self._profiles[name] for name in sorted(self._profiles)]


def resolve_ephemeral_xvector_reference(
    reference_id: str,
    *,
    references_directory: Path,
    model_id: str,
    model_revision: str,
) -> VoiceProfile:
    """Load one benchmark-only x-vector profile without mutating the registry."""
    relative = Path(reference_id)
    if (
        not reference_id.strip()
        or relative.is_absolute()
        or "\x00" in reference_id
        or "\\" in reference_id
        or ".." in relative.parts
    ):
        raise ProfileConfigurationError("benchmark reference must be a safe relative path")
    root = references_directory.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProfileConfigurationError(
            "benchmark reference escapes the reference directory"
        ) from exc
    if candidate.suffix.casefold() != ".wav" or not candidate.is_file():
        raise ProfileConfigurationError("benchmark reference must be an existing WAV file")
    normalized_id = candidate.relative_to(root).as_posix()
    return VoiceProfileRegistry._load_one(
        f"benchmark:{normalized_id}",
        {
            "version": EPHEMERAL_REFERENCE_VERSION,
            "reference_audio": str(candidate),
            "clone_mode": "x_vector_only",
        },
        model_id,
        model_revision,
        ephemeral=True,
    )


class GenerationProfileRegistry:
    def __init__(self, profiles: dict[str, GenerationProfile]):
        self._profiles = profiles

    @classmethod
    def load(cls, path: Path) -> GenerationProfileRegistry:
        raw = _read_yaml(path).get("generation_profiles")
        if not isinstance(raw, dict) or not raw:
            raise ProfileConfigurationError("generation_profiles must be a non-empty mapping")
        profiles: dict[str, GenerationProfile] = {}
        for name, config in raw.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                raise ProfileConfigurationError("invalid generation profile entry")
            version = str(config.get("version", "")).strip()
            if not version:
                raise ProfileConfigurationError(f"generation profile version is missing: {name}")
            controls = GenerationControls.model_validate(config.get("parameters", {}))
            parameters = controls.model_dump()
            profiles[name] = GenerationProfile(
                name=name,
                version=version,
                parameters=parameters,
                identity=_identity({"name": name, "version": version, "parameters": parameters}),
            )
        return cls(profiles)

    def get(self, name: str) -> GenerationProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise ProfileConfigurationError("unknown generation profile") from exc

    def all(self) -> list[GenerationProfile]:
        return [self._profiles[name] for name in sorted(self._profiles)]
