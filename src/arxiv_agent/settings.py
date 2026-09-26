"""Validated application configuration; importing this module has no side effects."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_AGENT_", env_file=".env", extra="forbid", frozen=True
    )

    generation_model: Literal["qwen2.5:3b"] = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_model: str = "Qwen/Qwen2.5-3B-Instruct"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    tokenizer_revision: str = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    context_tokens: int = Field(default=4096, ge=1024, le=32768)
    temperature: float = Field(default=0, ge=0, le=1)
    model_timeout_seconds: float = Field(default=180, gt=0)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    embedding_device: Literal["cpu"] = "cpu"
    candidate_count: int = Field(default=10, ge=1, le=10)
    retrieval_candidates: int = Field(default=12, ge=1)
    evidence_chunks: int = Field(default=6, ge=1)
    qa_max_distance: float = Field(default=0.85, ge=0, le=2)
    chunk_target_tokens: int = Field(default=200, ge=1, le=224)
    chunk_max_tokens: int = Field(default=224, ge=1, le=224)
    chunk_overlap_tokens: int = Field(default=30, ge=0)
    max_pdf_mb: int = Field(default=50, ge=1)
    max_pdf_pages: int = Field(default=60, ge=1)
    pdf_timeout_seconds: float = Field(default=60, gt=0, le=300)
    api_interval_seconds: float = Field(default=3, ge=3)
    api_timeout_seconds: float = Field(default=20, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=2)
    graph_recursion_limit: int = Field(default=100, ge=50, le=200)
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model_cache_dir: Path = Path(".cache/huggingface")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("ollama_base_url")
    @classmethod
    def local_ollama_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use a local Ollama HTTP origin, e.g. http://127.0.0.1:11434")
        _ = parsed.port  # Validate malformed/out-of-range ports.
        return value.rstrip("/")

    @model_validator(mode="after")
    def coherent_limits(self) -> "Settings":
        if not self.chunk_overlap_tokens < self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("Require chunk overlap < target <= maximum")
        if self.evidence_chunks > self.retrieval_candidates:
            raise ValueError("Evidence chunks cannot exceed retrieval candidates")
        return self

    def prepare_directories(self) -> None:
        for folder in ("pdfs", "parsed", "vectors", "evidence", "sessions"):
            (self.data_dir / folder).mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
