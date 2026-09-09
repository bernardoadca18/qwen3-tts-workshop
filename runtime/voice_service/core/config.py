from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PINNED_MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_revision: str = PINNED_MODEL_REVISION
    device: str = "cuda:0"
    dtype: str = "float16"
    attention_impl: str = "sdpa"

    voice_profiles_file: Path = Path("config/voices.yaml")
    generation_profiles_file: Path = Path("config/generation_profiles.yaml")
    references_directory: Path = Path("/data/references")
    default_voice_profile: str = "bernardo_xvec_v2"
    default_generation_profile: str = "narration_v2"

    max_queue_size: int = Field(default=32, ge=1, le=1000)
    max_text_chars: int = Field(default=50000, ge=1)
    target_chunk_chars: int = Field(default=250, ge=32)
    max_chunk_chars: int = Field(default=400, ge=64)
    chunk_max_duration_seconds: float = Field(default=90.0, gt=0)
    chunk_max_seconds_per_char: float = Field(default=0.35, gt=0)
    chunk_duration_grace_seconds: float = Field(default=10.0, ge=0)
    chunk_min_peak_amplitude: float = Field(default=1e-6, gt=0, le=1)
    job_retention_hours: int = Field(default=168, ge=1)
    max_ephemeral_prompt_cache_entries: int = Field(default=8, ge=1, le=128)
    speaker_identity_diagnostics_enabled: bool = True
    speaker_similarity_warning_threshold: float = Field(default=0.985, ge=-1, le=1)

    output_directory: Path = Path("data/outputs")
    job_database_path: Path = Path("data/jobs/narrations.sqlite3")
    warmup_enabled: bool = False
    warmup_text: str = "Teste de voz."

    sentence_pause_ms: int = Field(default=220, ge=0, le=5000)
    paragraph_pause_ms: int = Field(default=420, ge=0, le=5000)
    fallback_pause_ms: int = Field(default=120, ge=0, le=5000)
    hard_split_pause_ms: int = Field(default=80, ge=0, le=5000)
    max_result_duration_seconds: float = Field(default=7200.0, gt=0)
    legacy_generate_timeout_seconds: float = Field(default=1800.0, gt=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_runtime(self) -> "Settings":
        if not self.model_revision.strip():
            raise ValueError("MODEL_REVISION must be pinned")
        if self.device != "cpu" and not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("DEVICE must be cpu, cuda, or cuda:<index>")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("DTYPE must be float16, bfloat16, or float32")
        if self.device == "cpu" and self.dtype == "float16":
            raise ValueError("float16 on CPU is unsupported; use float32 or bfloat16")
        if self.attention_impl not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("unsupported ATTENTION_IMPL")
        if self.target_chunk_chars > self.max_chunk_chars:
            raise ValueError("TARGET_CHUNK_CHARS cannot exceed MAX_CHUNK_CHARS")
        if self.max_chunk_chars > self.max_text_chars:
            raise ValueError("MAX_CHUNK_CHARS cannot exceed MAX_TEXT_CHARS")
        if self.chunk_duration_grace_seconds >= self.chunk_max_duration_seconds:
            raise ValueError(
                "CHUNK_DURATION_GRACE_SECONDS must be below CHUNK_MAX_DURATION_SECONDS"
            )
        if not self.warmup_text.strip():
            raise ValueError("WARMUP_TEXT must not be empty")
        return self
