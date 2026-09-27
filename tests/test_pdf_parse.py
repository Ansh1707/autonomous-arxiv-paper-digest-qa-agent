import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from arxiv_agent.contracts import PaperMetadata, ParsedPaper, SessionState, Stage
from arxiv_agent.graph import build_parse_graph
from arxiv_agent.services.input_understanding import InputUnderstandingServices, TopicExtraction
from arxiv_agent.services.pdf_parse import ArxivParseServices, PdfParser, _heading, _ordered_blocks


def fixture_pdf(path: Path, *, abstract=True, references=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    if abstract:
        page.insert_text((72, 80), "Abstract", fontsize=13)
        text = ("This paper studies compact adaptation of language models and reports "
                "controlled experiments with clear measurements. ") * 3
        assert page.insert_textbox(pymupdf.Rect(72, 95, 540, 235), text, fontsize=10) >= 0
    page.insert_text((72, 260), "1 Introduction", fontsize=13)
    body = ("Researchers compare the proposed adaptation with conventional tuning "
            "on several tasks and discuss practical limitations. ") * 5
    assert page.insert_textbox(pymupdf.Rect(72, 275, 540, 560), body, fontsize=10) >= 0
    if references:
        page = document.new_page()
        page.insert_text((72, 80), "References", fontsize=13)
        page.insert_textbox(
            pymupdf.Rect(72, 100, 540, 170),
            "[1] A. Author. A useful prior study on adaptation. Conference 2020.", fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(72, 185, 540, 255),
            "[2] B. Author. Another reference with experimental details. Journal 2021.",
            fontsize=10,
        )
    document.save(path)
    document.close()
    return path


def paper():
    return PaperMetadata(
        arxiv_id="2106.09685", version=1, title="LoRA", authors=["A. Author"],
        abstract="Metadata abstract describing the research topic and its methods in detail.",
        categories=["cs.CL"], published="2021-06-17", updated="2021-06-18",
        abstract_url="https://arxiv.org/abs/2106.09685v1",
        pdf_url="https://arxiv.org/pdf/2106.09685v1",
    )


def state_for(path: Path):
    with pymupdf.open(path) as document:
        pages = document.page_count
    return SessionState(
        user_input="2106.09685v1", selected_paper=paper(), pdf_path=str(path),
        pdf_checksum=hashlib.sha256(path.read_bytes()).hexdigest(), pdf_pages=pages,
        pdf_size_bytes=path.stat().st_size,
    )


def test_extracts_abstract_sections_references_and_page_provenance(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf")
    output, parsed = PdfParser(settings).parse(state_for(path))
    assert output.is_file()
    assert parsed.abstract_source == "pdf"
    assert parsed.abstract.startswith("This paper studies")
    assert [item.title for item in parsed.sections] == [
        "Front matter", "Abstract", "Introduction", "References"
    ] or [item.title for item in parsed.sections] == [
        "Abstract", "Introduction", "References"
    ]
    assert len(parsed.references) == 2
    assert parsed.page_count == 2
    assert {block.page for block in parsed.blocks} == {1, 2}
    assert all(block.block_id and len(block.bbox) == 4 for block in parsed.blocks)
    assert ParsedPaper.model_validate_json(output.read_text()) == parsed


def test_numbered_headings_with_trailing_period_are_recognized():
    assert _heading("2. Method", 72, 595) == "Method"
    assert _heading("2.1. The Contrastive Learning Framework", 72, 595) == (
        "The Contrastive Learning Framework"
    )


def test_missing_pdf_abstract_uses_metadata_with_warning(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf", abstract=False)
    _, parsed = PdfParser(settings).parse(state_for(path))
    assert parsed.abstract_source == "metadata"
    assert parsed.abstract == paper().abstract
    assert any("metadata abstract" in warning for warning in parsed.warnings)


def test_headingless_text_pdf_retains_body_and_metadata_abstract(settings):
    path = settings.data_dir / "pdfs" / "2106.09685v1.pdf"
    path.parent.mkdir(parents=True)
    document = pymupdf.open()
    page = document.new_page()
    paragraph = (
        "This study examines compact adaptation of large language models and reports "
        "controlled evidence about training and evaluation. "
    ) * 5
    assert page.insert_textbox(
        pymupdf.Rect(72, 80, 540, 650), paragraph, fontsize=10
    ) >= 0
    document.save(path)
    document.close()
    _, parsed = PdfParser(settings).parse(state_for(path))
    assert parsed.abstract_source == "metadata"
    assert parsed.sections and parsed.blocks
    assert all(block.page == 1 for block in parsed.blocks)
    assert "compact adaptation" in " ".join(block.text for block in parsed.blocks)


def test_missing_references_is_reported_without_inventing_entries(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf", references=False)
    _, parsed = PdfParser(settings).parse(state_for(path))
    assert parsed.references == []
    assert any("No numbered reference" in warning for warning in parsed.warnings)


def test_image_only_pdf_fails_clearly(settings):
    path = settings.data_dir / "pdfs" / "2106.09685v1.pdf"
    path.parent.mkdir(parents=True)
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    from arxiv_agent.services.base import StageFailure

    try:
        PdfParser(settings).parse(state_for(path))
    except StageFailure as exc:
        assert exc.code == "UNREADABLE_PDF"
        assert "OCR" in exc.recovery
    else:
        raise AssertionError("Blank PDF should not parse")


def test_changed_pdf_checksum_blocks_parsing(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf")
    state = state_for(path)
    state.pdf_checksum = "0" * 64
    from arxiv_agent.services.base import StageFailure

    try:
        PdfParser(settings).parse(state)
    except StageFailure as exc:
        assert exc.code == "PDF_CHANGED"
    else:
        raise AssertionError("Changed PDF should not parse")


def test_corrupt_parsed_cache_is_rebuilt(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf")
    parser = PdfParser(settings)
    output, first = parser.parse(state_for(path))
    output.write_text("not json")
    again_path, again = parser.parse(state_for(path))
    assert again_path == output
    assert again == first


def test_two_column_blocks_read_left_then_right(settings):
    document = pymupdf.open()
    page = document.new_page()
    left = "Left column discusses the method and presents evidence for the result. " * 3
    right = "Right column continues the argument with comparisons and limitations. " * 3
    page.insert_textbox(pymupdf.Rect(72, 90, 280, 225), left, fontsize=9)
    page.insert_textbox(pymupdf.Rect(72, 245, 280, 380), left, fontsize=9)
    page.insert_textbox(pymupdf.Rect(330, 90, 540, 225), right, fontsize=9)
    page.insert_textbox(pymupdf.Rect(330, 245, 540, 380), right, fontsize=9)
    order = [_text[4].strip() for _text in _ordered_blocks(page)]
    assert len(order) == 4
    assert order[0].startswith("Left") and order[1].startswith("Left")
    assert order[2].startswith("Right") and order[3].startswith("Right")
    document.close()


def test_spanning_block_never_drops_overlapping_column_text():
    document = pymupdf.open()
    page = document.new_page()
    left = "Left column evidence about an experimental comparison. " * 4
    right = "Right column evidence about a measured limitation. " * 4
    page.insert_textbox(pymupdf.Rect(72, 90, 280, 225), left, fontsize=9)
    page.insert_textbox(pymupdf.Rect(72, 245, 280, 380), left, fontsize=9)
    page.insert_textbox(pymupdf.Rect(330, 90, 540, 225), right, fontsize=9)
    page.insert_textbox(pymupdf.Rect(330, 245, 540, 380), right, fontsize=9)
    page.insert_text((145, 200), "Full width note across both columns")
    raw = [block for block in page.get_text("blocks") if block[6] == 0]
    ordered = _ordered_blocks(page)
    assert len(ordered) == len(raw)
    assert sorted(block[4] for block in ordered) == sorted(block[4] for block in raw)
    document.close()


class Interpreter:
    def interpret(self, topic):
        return TopicExtraction(terms=["LoRA"], date_intent="none")


class Result:
    title = "LoRA"
    authors = [SimpleNamespace(name="A. Author")]
    summary = "Metadata abstract describing the research topic and its methods in detail."
    primary_category = "cs.CL"
    categories = ["cs.CL"]
    published = datetime(2021, 6, 17, tzinfo=UTC)
    updated = datetime(2021, 6, 18, tzinfo=UTC)

    def get_short_id(self):
        return "2106.09685v1"


class ArxivClient:
    def results(self, search):
        return iter([Result()])


class CachedDownloader:
    def __init__(self, path):
        self.path = path

    def download(self, selected):
        return {
            "pdf_path": str(self.path),
            "pdf_checksum": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "pdf_pages": 2,
            "pdf_size_bytes": self.path.stat().st_size,
        }


def test_graph_stops_after_parse_and_records_artifact(settings):
    path = fixture_pdf(settings.data_dir / "pdfs" / "2106.09685v1.pdf")
    services = ArxivParseServices(
        settings, InputUnderstandingServices(Interpreter()),
        arxiv_client=ArxivClient(), downloader=CachedDownloader(path),
    )
    state = SessionState.model_validate(
        build_parse_graph(services, settings).invoke(SessionState(user_input="2106.09685v1"))
    )
    assert state.error is None
    assert state.stage_history == [Stage.UNDERSTAND, Stage.LOOKUP, Stage.DOWNLOAD, Stage.PARSE]
    assert state.parsed_path and Path(state.parsed_path).is_file()
    assert state.abstract_source == "pdf"
    assert state.parsed_reference_count == 2
    assert state.parsed_block_count > 4
    assert Stage.INDEX not in state.stage_history
