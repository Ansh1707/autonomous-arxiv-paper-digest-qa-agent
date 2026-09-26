import httpx
import pytest

from arxiv_agent.diagnostics import ollama_check


@pytest.mark.parametrize(
    "models,passed",
    [
        ([{"name": "qwen2.5:3b", "digest": "test-digest"}], True),
        ([{"name": "qwen2.5:7b"}], False),
        ([], False),
    ],
)
def test_doctor_checks_exact_model_tag(monkeypatch, settings, models, passed):
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": models}))
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs)
    )
    result = ollama_check(settings)
    assert result.ok is passed
    if not passed:
        assert "ollama pull qwen2.5:3b" in result.detail


def test_doctor_explains_unavailable_server(monkeypatch, settings):
    original_client = httpx.Client

    def unavailable(request):
        raise httpx.ConnectError("Connection refused", request=request)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(unavailable), **kwargs),
    )
    result = ollama_check(settings)
    assert result.ok is False
    assert "ollama serve" in result.detail
