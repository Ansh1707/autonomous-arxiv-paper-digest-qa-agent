import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pymupdf
import pytest

from arxiv_agent.contracts import PaperMetadata, SessionState, Stage
from arxiv_agent.graph import build_download_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import InputUnderstandingServices, TopicExtraction
from arxiv_agent.services.pdf_download import ArxivPdfServices, PdfDownloader


def sample_pdf(pages=1):
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()
    result = document.tobytes()
    document.close()
    return result


def paper(version=1):
    return PaperMetadata(
        arxiv_id="2106.09685",
        version=version,
        title="LoRA",
        authors=["A. Author"],
        abstract="An abstract",
        categories=["cs.CL"],
        published="2021-06-17",
        updated="2021-06-18",
        abstract_url=f"https://arxiv.org/abs/2106.09685v{version}",
        pdf_url=f"https://arxiv.org/pdf/2106.09685v{version}",
    )


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_streamed_download_checksum_pages_and_cache_reuse(settings):
    content = sample_pdf()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=content)

    downloader = PdfDownloader(settings, client(handler))
    first = downloader.download(paper())
    assert Path(first["pdf_path"]).read_bytes() == content
    assert first["pdf_checksum"] == hashlib.sha256(content).hexdigest()
    assert first["pdf_pages"] == 1
    assert first["pdf_size_bytes"] == len(content)
    assert calls == ["https://arxiv.org/pdf/2106.09685v1"]
    assert downloader.download(paper()) == first
    assert len(calls) == 1


def test_corrupted_cache_is_refetched(settings):
    content = sample_pdf()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=content)

    downloader = PdfDownloader(settings, client(handler))
    first = downloader.download(paper())
    Path(first["pdf_path"]).write_bytes(b"broken")
    again = downloader.download(paper())
    assert again == first
    assert len(calls) == 2


def test_interrupted_download_leaves_no_partial_pdf(settings):
    def interrupted(request):
        raise httpx.ReadError("connection dropped", request=request)

    downloader = PdfDownloader(settings, client(interrupted))
    with pytest.raises(StageFailure, match="PDF request failed") as exc:
        downloader.download(paper())
    assert exc.value.retryable
    assert not list((settings.data_dir / "pdfs").glob("*.pdf"))
    assert not list((settings.data_dir / "pdfs").glob(".partial-*"))


def test_versioned_cache_files_do_not_collide(settings):
    content = sample_pdf()
    downloader = PdfDownloader(settings, client(lambda request: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=content
    )))
    one = downloader.download(paper(1))
    two = downloader.download(paper(2))
    assert one["pdf_path"] != two["pdf_path"]
    assert Path(one["pdf_path"]).exists() and Path(two["pdf_path"]).exists()


class Interpreter:
    def interpret(self, topic):
        return TopicExtraction(terms=["LoRA"], date_intent="none")


class Result:
    title = "LoRA"
    authors = [SimpleNamespace(name="A. Author")]
    summary = "An abstract"
    primary_category = "cs.CL"
    categories = ["cs.CL"]
    published = datetime(2021, 6, 17, tzinfo=UTC)
    updated = datetime(2021, 6, 18, tzinfo=UTC)

    def get_short_id(self):
        return "2106.09685v1"


class ArxivClient:
    def results(self, search):
        return iter([Result()])


def run_graph(settings, handler):
    services = ArxivPdfServices(
        settings,
        InputUnderstandingServices(Interpreter()),
        arxiv_client=ArxivClient(),
        downloader=PdfDownloader(settings, client(handler)),
    )
    return SessionState.model_validate(
        build_download_graph(services, settings).invoke(SessionState(user_input="2106.09685v1"))
    )


def test_graph_stops_after_real_download(settings):
    content = sample_pdf()
    state = run_graph(settings, lambda request: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=content
    ))
    assert state.error is None
    assert state.stage_history == [Stage.UNDERSTAND, Stage.LOOKUP, Stage.DOWNLOAD]
    assert state.pdf_path and state.pdf_checksum
    assert state.pdf_pages == 1
    assert Stage.PARSE not in state.stage_history


def test_non_pdf_response_rejected_without_cached_artifact(settings):
    state = run_graph(settings, lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, content=b"<html>error</html>"
    ))
    assert state.error.code == "INVALID_PDF"
    assert state.status == "failed"
    assert not list((settings.data_dir / "pdfs").glob("*.pdf"))
    assert not list((settings.data_dir / "pdfs").glob(".partial-*"))


def test_too_many_pages_rejected(settings):
    limited = type(settings)(**(settings.model_dump() | {"max_pdf_pages": 1}))
    content = sample_pdf(2)
    state = run_graph(limited, lambda request: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=content
    ))
    assert state.error.code == "PDF_TOO_LONG"
    assert not list((limited.data_dir / "pdfs").glob("*.pdf"))


def test_content_length_limit_rejected_before_stream(settings):
    limited = type(settings)(**(settings.model_dump() | {"max_pdf_mb": 1}))
    state = run_graph(limited, lambda request: httpx.Response(
        200,
        headers={"content-type": "application/pdf", "content-length": str(2 * 1024 * 1024)},
        content=b"%PDF-bad",
    ))
    assert state.error.code == "PDF_TOO_LARGE"


def test_stream_limit_rejects_false_content_length(settings):
    limited = type(settings)(**(settings.model_dump() | {"max_pdf_mb": 1}))
    state = run_graph(limited, lambda request: httpx.Response(
        200,
        headers={"content-type": "application/pdf", "content-length": "5"},
        content=b"%PDF-" + b"x" * (1024 * 1024),
    ))
    assert state.error.code == "PDF_TOO_LARGE"
    assert not list((limited.data_dir / "pdfs").glob(".partial-*"))


def test_redirect_outside_arxiv_is_blocked(settings):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.com/evil.pdf"})

    state = run_graph(settings, handler)
    assert state.error.code == "PDF_REDIRECT_ERROR"
    assert seen == ["https://arxiv.org/pdf/2106.09685v1"]


def test_redirect_within_arxiv_is_allowed(settings):
    content = sample_pdf()
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "/pdf/2106.09685v1?download=1"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=content)

    state = run_graph(settings, handler)
    assert state.error is None
    assert len(seen) == 2


def test_transient_http_failure_retries_at_most_twice(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    state = run_graph(settings, handler)
    assert state.error.code == "PDF_HTTP_ERROR"
    assert state.status == "failed"
    assert state.stage_history.count(Stage.DOWNLOAD) == 3
    assert len(calls) == 3


def test_http_404_does_not_retry(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    state = run_graph(settings, handler)
    assert state.error.code == "PDF_HTTP_ERROR"
    assert len(calls) == 1
