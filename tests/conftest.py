import os

import pytest

from arxiv_agent.settings import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """No test reads a personal .env or writes into the project's real data/cache directories."""
    for key in os.environ:
        if key.startswith("ARXIV_AGENT_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        model_cache_dir=tmp_path / "models",
    )
