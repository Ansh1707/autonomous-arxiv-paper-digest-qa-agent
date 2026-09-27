"""Page-aware scholarly PDF text extraction; no OCR or LLM rewriting."""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile

import pymupdf
from pydantic import ValidationError

from arxiv_agent.contracts import (
    ParsedBlock,
    ParsedPaper,
    ParsedSection,
    SessionState,
    Stage,
)
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import InputUnderstandingServices
from arxiv_agent.services.pdf_download import ArxivPdfServices, PdfDownloader
from arxiv_agent.services.selection import EmbeddingRanker
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)
_NUMBERED = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?$")
_REFERENCE = re.compile(r"^\[\d+\]")
_CAPTION = re.compile(r"^(?:Figure|Fig\.|Table)\s+\d+[.:]", re.IGNORECASE)
_UNNUMBERED = {
    "abstract", "references", "bibliography", "acknowledgments", "acknowledgements",
    "introduction", "conclusion", "conclusions", "limitations", "appendix",
}


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)
    text = re.sub(r"-\s*\n\s*", "-", text)
    return " ".join(text.split())


def _heading(raw: str, x0: float, page_width: float) -> str | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) == 2 and x0 <= page_width * 0.3 and _NUMBERED.fullmatch(lines[0]):
        title = _clean(lines[1])
        if 3 <= len(title) <= 110 and len(title.split()) <= 15:
            return title
    if len(lines) == 1:
        line = _clean(lines[0])
        if line.casefold() in _UNNUMBERED:
            return "References" if line.casefold() == "bibliography" else line.title()
        match = re.fullmatch(r"(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+(.+)", line)
        if (
            match and x0 <= page_width * 0.3
            and 3 <= len(match[1]) <= 110 and len(match[1].split()) <= 15
        ):
            return match[1]
    return None


def _ordered_blocks(page: pymupdf.Page) -> list[tuple]:
    width, height = page.rect.width, page.rect.height
    blocks = []
    for block in page.get_text("blocks"):
        if block[6] != 0 or not block[4].strip():
            continue
        text = _clean(block[4])
        if not text:
            continue
        if re.fullmatch(r"\d{1,3}", text) and block[1] > 0.9 * height:
            continue
        if text.lower().startswith("arxiv:") and block[2] < 0.12 * width:
            continue
        blocks.append(block)
    middle = width / 2
    gap = width * 0.03
    left = [b for b in blocks if b[2] <= middle + gap and (b[0] + b[2]) / 2 < middle]
    right = [b for b in blocks if b[0] >= middle - gap and (b[0] + b[2]) / 2 >= middle]
    left_text = sum(len(_clean(b[4])) for b in left)
    right_text = sum(len(_clean(b[4])) for b in right)
    overlap = any(a[1] < b[3] and b[1] < a[3] for a in left for b in right)
    if len(left) < 2 or len(right) < 2 or min(left_text, right_text) < 150 or not overlap:
        return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))
    spans = sorted((b for b in blocks if b not in left and b not in right), key=lambda b: b[1])
    ordered = []
    lower = float("-inf")
    for span in [*spans, None]:
        upper = span[1] if span is not None else float("inf")
        ordered.extend(sorted((b for b in left if lower <= b[1] < upper), key=lambda b: b[1]))
        ordered.extend(sorted((b for b in right if lower <= b[1] < upper), key=lambda b: b[1]))
        if span is not None:
            ordered.append(span)
            lower = span[1]
    return ordered


class PdfParser:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _input_path(self, state: SessionState) -> Path:
        paper = state.selected_paper
        if not paper or not state.pdf_path or not state.pdf_checksum:
            raise StageFailure(
                "MISSING_PDF", "A selected paper and downloaded PDF are required.",
                "Run fetch-paper before parsing.",
            )
        expected = (
            self.settings.data_dir / "pdfs" /
            f"{paper.arxiv_id}v{paper.version}.pdf".replace("/", "_")
        ).resolve()
        path = Path(state.pdf_path).resolve()
        if path != expected or not path.is_file():
            raise StageFailure(
                "MISSING_PDF", "The selected version's cached PDF is missing or unexpected.",
                "Run fetch-paper again.",
            )
        digest = hashlib.sha256()
        try:
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(65536), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise StageFailure(
                "PDF_READ_ERROR", "Could not read the cached PDF.",
                "Check file permissions and rerun fetch-paper.",
            ) from exc
        if digest.hexdigest() != state.pdf_checksum:
            raise StageFailure(
                "PDF_CHANGED", "Cached PDF checksum changed before parsing.",
                "Rerun fetch-paper to restore the verified PDF.",
            )
        return path

    def parse(self, state: SessionState) -> tuple[Path, ParsedPaper]:
        path = self._input_path(state)
        paper = state.selected_paper
        safe_id = f"{paper.arxiv_id}v{paper.version}".replace("/", "_")
        output = self.settings.data_dir / "parsed" / f"{safe_id}-{state.pdf_checksum[:12]}.json"
        try:
            cached = ParsedPaper.model_validate_json(output.read_text(encoding="utf-8"))
            if cached.paper == paper and cached.pdf_checksum == state.pdf_checksum:
                return output.resolve(), cached
        except (OSError, ValidationError, ValueError):
            pass
        parsed = self._extract(path, state)
        temporary = None
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", dir=output.parent, encoding="utf-8", delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(parsed.model_dump_json(indent=2))
            temporary.replace(output)
        except OSError as exc:
            raise StageFailure(
                "PARSE_STORAGE_ERROR", "Could not save parsed paper text.",
                "Check free disk space and ARXIV_AGENT_DATA_DIR permissions.",
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return output.resolve(), parsed

    def _extract(self, path: Path, state: SessionState) -> ParsedPaper:
        paper = state.selected_paper
        blocks: list[ParsedBlock] = []
        sections: list[ParsedSection] = []
        warnings = []
        current = "Front matter"
        section_start = 1
        page_count = 0
        pages_with_text = 0
        total_chars = 0
        try:
            with pymupdf.open(path) as document:
                page_count = document.page_count
                if (
                    document.is_encrypted
                    or page_count < 1
                    or page_count > self.settings.max_pdf_pages
                ):
                    raise StageFailure(
                        "UNREADABLE_PDF", "PDF is encrypted, empty, or exceeds the page limit.",
                        "Choose a readable text-based paper within the configured page limit.",
                    )
                if state.pdf_pages and page_count != state.pdf_pages:
                    raise StageFailure(
                        "PDF_CHANGED", "PDF page count differs from the downloaded record.",
                        "Rerun fetch-paper before parsing.",
                    )
                for number, page in enumerate(document, start=1):
                    ordered = _ordered_blocks(page)
                    page_chars = 0
                    for index, raw in enumerate(ordered, start=1):
                        text = _clean(raw[4])
                        if number > 1 and text.casefold() == paper.title.casefold():
                            continue  # Repeated running title is a page header, not a section.
                        heading = _heading(raw[4], raw[0], page.rect.width)
                        if heading:
                            if blocks:
                                sections.append(ParsedSection(
                                    title=current, page_start=section_start,
                                    page_end=blocks[-1].page,
                                ))
                            current = heading
                            section_start = number
                            kind = "heading"
                        else:
                            kind = "caption" if _CAPTION.match(text) else "body"
                        blocks.append(ParsedBlock(
                            block_id=f"p{number}-b{index}", page=number,
                            section=current, kind=kind, text=text,
                            bbox=tuple(round(float(value), 2) for value in raw[:4]),
                        ))
                        page_chars += sum(char.isalnum() for char in text)
                    total_chars += page_chars
                    if page_chars >= 50:
                        pages_with_text += 1
        except StageFailure:
            raise
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
            raise StageFailure(
                "UNREADABLE_PDF", "PDF text could not be extracted reliably.",
                "Choose a text-based paper; OCR is not supported yet.",
            ) from exc
        if total_chars < 200 or pages_with_text < max(1, (page_count + 4) // 5):
            raise StageFailure(
                "UNREADABLE_PDF", "PDF contains too little selectable text for reliable parsing.",
                "Choose a text-based arXiv paper; OCR is not supported yet.",
            )
        sections.append(ParsedSection(
            title=current, page_start=section_start, page_end=page_count
        ))
        abstract_blocks = [
            block.text for block in blocks
            if block.section.casefold() == "abstract" and block.kind == "body"
        ]
        abstract = " ".join(abstract_blocks)
        source = "pdf"
        if len(abstract) < 100:
            abstract = paper.abstract
            source = "metadata"
            warnings.append("PDF abstract was not recovered; used arXiv metadata abstract.")
        reference_blocks = [
            block.text for block in blocks
            if block.section.casefold() in {"references", "bibliography"}
            and block.kind == "body"
        ]
        references = [text for text in reference_blocks if _REFERENCE.match(text)]
        if not references:
            warnings.append("No numbered reference entries were identified in the PDF.")
        return ParsedPaper(
            paper=paper, pdf_checksum=state.pdf_checksum, page_count=page_count,
            pages_with_text=pages_with_text, abstract=abstract, abstract_source=source,
            sections=sections, references=references, blocks=blocks, warnings=warnings,
        )


class ArxivParseServices(ArxivPdfServices):
    def __init__(
        self,
        settings: Settings,
        understanding: InputUnderstandingServices,
        *,
        selection_rank: int = 1,
        ranker: EmbeddingRanker | None = None,
        arxiv_client=None,
        downloader: PdfDownloader | None = None,
        parser: PdfParser | None = None,
    ):
        super().__init__(
            settings, understanding, selection_rank=selection_rank,
            ranker=ranker, arxiv_client=arxiv_client, downloader=downloader,
        )
        self.parser = parser or PdfParser(settings)

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage == Stage.PARSE:
            output, parsed = self.parser.parse(state)
            return {
                "parsed_path": str(output),
                "sections": [section.title for section in parsed.sections],
                "abstract_text": parsed.abstract,
                "abstract_source": parsed.abstract_source,
                "references_found": bool(parsed.references),
                "parsed_reference_count": len(parsed.references),
                "parsed_block_count": len(parsed.blocks),
                "parsed_pages_with_text": parsed.pages_with_text,
                "warnings": [*state.warnings, *parsed.warnings],
            }
        return super().run_stage(stage, state)
