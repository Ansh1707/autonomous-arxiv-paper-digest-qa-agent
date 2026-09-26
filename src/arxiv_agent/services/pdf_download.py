"""Version-pinned arXiv PDF download with bounded streaming and verified reuse."""

import hashlib
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from urllib.parse import urljoin, urlsplit

import httpx
import pymupdf

from arxiv_agent.contracts import PaperMetadata, SessionState, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import InputUnderstandingServices, normalize_arxiv_id
from arxiv_agent.services.selection import ArxivSelectionServices, EmbeddingRanker
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)
_REDIRECTS = {301, 302, 303, 307, 308}
_PDF_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


def _pdf_properties(path: Path, max_pages: int) -> int:
    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError("PDF is encrypted")
            pages = document.page_count
            if pages < 1:
                raise ValueError("PDF has no pages")
            if pages > max_pages:
                raise StageFailure(
                    "PDF_TOO_LONG", f"PDF has {pages} pages; limit is {max_pages}.",
                    "Choose a shorter paper or increase ARXIV_AGENT_MAX_PDF_PAGES.",
                )
            document.load_page(0)
            document.load_page(pages - 1)
            return pages
    except StageFailure:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError, RuntimeError) as exc:
        raise StageFailure(
            "INVALID_PDF", "Downloaded file is not a readable, unencrypted PDF.",
            "Retry later or choose another arXiv paper.",
        ) from exc


class PdfDownloader:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            timeout=settings.pdf_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "arxiv-digest-agent/0.1 (educational research)"},
        )

    def _paths(self, paper: PaperMetadata) -> tuple[str, Path, Path]:
        short_id = f"{paper.arxiv_id}v{paper.version}"
        if normalize_arxiv_id(short_id) != short_id:
            raise StageFailure(
                "INVALID_PAPER_ID", "Selected paper has an invalid arXiv versioned ID.",
                "Select a paper returned by the official arXiv API.",
            )
        expected_url = f"https://arxiv.org/pdf/{short_id}"
        if str(paper.pdf_url).rstrip("/") != expected_url:
            raise StageFailure(
                "INVALID_PDF_URL", "Selected paper has an unexpected PDF URL.",
                "Select a paper returned by the official arXiv API.",
            )
        stem = short_id.replace("/", "_")
        folder = self.settings.data_dir / "pdfs"
        return expected_url, folder / f"{stem}.pdf", folder / f"{stem}.json"

    def _cached(self, path: Path, manifest: Path, short_id: str) -> dict | None:
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if record["arxiv_id"] != short_id:
                return None
            size = path.stat().st_size
            if size != record["size_bytes"] or size > self.settings.max_pdf_mb * 1024 * 1024:
                return None
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != record["sha256"]:
                return None
            pages = _pdf_properties(path, self.settings.max_pdf_pages)
            if pages != record["pages"]:
                return None
            return {"pdf_path": str(path.resolve()), "pdf_checksum": checksum,
                    "pdf_pages": pages, "pdf_size_bytes": size}
        except (OSError, ValueError, KeyError, TypeError, StageFailure):
            return None

    def download(self, paper: PaperMetadata) -> dict:
        url, path, manifest = self._paths(paper)
        short_id = f"{paper.arxiv_id}v{paper.version}"
        cached = self._cached(path, manifest, short_id)
        if cached:
            return cached
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "wb", dir=path.parent, prefix=".partial-", delete=False
            ) as file:
                temporary = Path(file.name)
        except OSError as exc:
            raise StageFailure(
                "PDF_STORAGE_ERROR", "Could not create a temporary PDF file.",
                "Check free disk space and ARXIV_AGENT_DATA_DIR permissions.",
            ) from exc
        try:
            size, checksum = self._fetch(url, temporary)
            pages = _pdf_properties(temporary, self.settings.max_pdf_pages)
            temporary.replace(path)
            record = {"arxiv_id": short_id, "size_bytes": size,
                      "sha256": checksum, "pages": pages, "url": url}
            manifest_temp = None
            try:
                with NamedTemporaryFile(
                    "w", dir=path.parent, encoding="utf-8", delete=False
                ) as file:
                    manifest_temp = Path(file.name)
                    json.dump(record, file)
                manifest_temp.replace(manifest)
            finally:
                if manifest_temp is not None:
                    manifest_temp.unlink(missing_ok=True)
            return {"pdf_path": str(path.resolve()), "pdf_checksum": checksum,
                    "pdf_pages": pages, "pdf_size_bytes": size}
        except OSError as exc:
            raise StageFailure(
                "PDF_STORAGE_ERROR", "Could not save the downloaded PDF.",
                "Check free disk space and ARXIV_AGENT_DATA_DIR permissions.",
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _fetch(self, url: str, destination: Path) -> tuple[int, str]:
        deadline = monotonic() + self.settings.pdf_timeout_seconds
        current = url
        for _ in range(5):
            if monotonic() > deadline:
                raise StageFailure(
                    "PDF_TIMEOUT", "PDF download exceeded the time limit.",
                    "Retry later or increase ARXIV_AGENT_PDF_TIMEOUT_SECONDS.", retryable=True,
                )
            try:
                with self.client.stream("GET", current) as response:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise StageFailure(
                                "PDF_REDIRECT_ERROR", "arXiv returned an empty redirect.",
                                "Retry later or choose another paper.",
                            )
                        current = urljoin(current, location)
                        parsed = urlsplit(current)
                        if (parsed.scheme != "https" or parsed.hostname not in _PDF_HOSTS
                                or parsed.username or parsed.password or parsed.port):
                            raise StageFailure(
                                "PDF_REDIRECT_ERROR", "arXiv redirected outside approved hosts.",
                                "Retry later or check the paper PDF URL on arxiv.org.",
                            )
                        continue
                    if response.status_code != 200:
                        retry = response.status_code == 429 or 500 <= response.status_code < 600
                        raise StageFailure(
                            "PDF_HTTP_ERROR", f"PDF request returned HTTP {response.status_code}.",
                            "Retry later or check the paper PDF on arxiv.org.", retryable=retry,
                        )
                    content_type = response.headers.get("content-type", "").split(";", 1)[0]
                    if content_type and content_type not in {
                        "application/pdf", "application/octet-stream"
                    }:
                        raise StageFailure(
                            "INVALID_PDF", f"arXiv returned {content_type} instead of a PDF.",
                            "Retry later or choose another paper.",
                        )
                    limit = self.settings.max_pdf_mb * 1024 * 1024
                    declared = response.headers.get("content-length")
                    if declared and declared.isdecimal() and int(declared) > limit:
                        raise StageFailure(
                            "PDF_TOO_LARGE",
                            f"PDF exceeds the {self.settings.max_pdf_mb} MB limit.",
                            "Choose a smaller paper or increase ARXIV_AGENT_MAX_PDF_MB.",
                        )
                    digest = hashlib.sha256()
                    size = 0
                    prefix = b""
                    with destination.open("wb") as file:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            if monotonic() > deadline:
                                raise StageFailure(
                                    "PDF_TIMEOUT", "PDF download exceeded the time limit.",
                                    "Retry later or increase ARXIV_AGENT_PDF_TIMEOUT_SECONDS.",
                                    retryable=True,
                                )
                            size += len(chunk)
                            if size > limit:
                                raise StageFailure(
                                    "PDF_TOO_LARGE",
                                    f"PDF exceeds the {self.settings.max_pdf_mb} MB limit.",
                                    "Choose a smaller paper or increase ARXIV_AGENT_MAX_PDF_MB.",
                                )
                            prefix = (prefix + chunk)[:5]
                            digest.update(chunk)
                            file.write(chunk)
                    if size < 5 or prefix != b"%PDF-":
                        raise StageFailure(
                            "INVALID_PDF", "Downloaded content lacks a PDF header.",
                            "Retry later or choose another paper.",
                        )
                    return size, digest.hexdigest()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                logger.warning("PDF transport failed: %s", exc)
                raise StageFailure(
                    "PDF_NETWORK_ERROR", f"PDF request failed: {type(exc).__name__}.",
                    "Check internet access and retry.", retryable=True,
                ) from exc
        raise StageFailure(
            "PDF_REDIRECT_ERROR", "arXiv redirected too many times.",
            "Retry later or check the paper PDF URL on arxiv.org.",
        )


class ArxivPdfServices(ArxivSelectionServices):
    def __init__(
        self,
        settings: Settings,
        understanding: InputUnderstandingServices,
        *,
        selection_rank: int = 1,
        ranker: EmbeddingRanker | None = None,
        arxiv_client=None,
        downloader: PdfDownloader | None = None,
    ):
        super().__init__(
            settings, understanding, selection_rank=selection_rank,
            ranker=ranker, client=arxiv_client,
        )
        self.downloader = downloader or PdfDownloader(settings)

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage == Stage.DOWNLOAD:
            if state.selected_paper is None:
                raise StageFailure(
                    "NO_SELECTED_PAPER", "PDF download needs a selected paper.",
                    "Select a paper first.",
                )
            return self.downloader.download(state.selected_paper)
        return super().run_stage(stage, state)
