import pytest
from pydantic import ValidationError

from arxiv_agent.settings import Settings


@pytest.mark.parametrize(
    "patch",
    [
        {"generation_model": "qwen2.5:7b"},
        {"context_tokens": 32769},
        {"chunk_overlap_tokens": 200},
        {"chunk_max_tokens": 100},
        {"evidence_chunks": 13},
        {"api_interval_seconds": 2},
        {"pdf_timeout_seconds": 0},
        {"max_retries": 3},
        {"ollama_base_url": "https://example.com"},
        {"ollama_base_url": "http://user:password@localhost:11434"},
    ],
)
def test_invalid_configuration_fails_early(patch):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **patch)


def test_environment_override(monkeypatch):
    monkeypatch.setenv("ARXIV_AGENT_CONTEXT_TOKENS", "2048")
    assert Settings(_env_file=None).context_tokens == 2048


def test_runtime_folders_created_only_explicitly(settings):
    assert not settings.data_dir.exists()
    settings.prepare_directories()
    assert all(
        (settings.data_dir / name).is_dir() for name in ["pdfs", "parsed", "vectors", "sessions"]
    )
    assert settings.output_dir.is_dir()
    assert settings.model_cache_dir.is_dir()
