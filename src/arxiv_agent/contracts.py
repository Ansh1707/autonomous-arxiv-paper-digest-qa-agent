"""Serializable domain contracts. Structural validity does not prove grounding."""

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Stage(StrEnum):
    UNDERSTAND = "understand"
    LOOKUP = "lookup_metadata"
    SEARCH = "search"
    BROADEN = "broaden_query"
    RANK = "rank_select"
    DOWNLOAD = "download"
    PARSE = "parse"
    INDEX = "chunk_embed"
    EVIDENCE = "extract_evidence"
    BRIEF = "summarize"
    READY = "qa_ready"
    QA_CHECK = "check_qa_ready"
    RETRIEVE = "retrieve_evidence"
    ANSWER = "answer_question"
    VALIDATE = "validate_answer"
    ERROR = "handle_error"


class Intent(Record):
    kind: Literal["lookup", "topic"]
    arxiv_id: Text | None = None
    query: Text | None = None
    search_terms: list[Text] = Field(default_factory=list, max_length=5)
    date_from: date | None = None
    date_to: date | None = None
    date_interpretation: Literal["none", "recent", "explicit"] = "none"
    arxiv_query: Text | None = None

    @model_validator(mode="after")
    def required_input(self) -> "Intent":
        if self.kind == "lookup":
            if (
                not self.arxiv_id
                or self.query is not None
                or self.search_terms
                or self.date_from is not None
                or self.date_to is not None
                or self.arxiv_query is not None
                or self.date_interpretation != "none"
            ):
                raise ValueError("Lookup requires only an arxiv_id")
        elif not self.query or self.arxiv_id is not None:
            raise ValueError("Topic requires query and no arxiv_id")
        elif (self.date_from is None) != (self.date_to is None):
            raise ValueError("Topic date bounds must be set together")
        elif self.date_from is not None and self.date_from > self.date_to:
            raise ValueError("Topic date bounds must be ordered")
        elif self.date_interpretation == "none" and self.date_from is not None:
            raise ValueError("Unfiltered topic cannot have date bounds")
        elif self.date_interpretation != "none" and self.date_from is None:
            raise ValueError("Filtered topic requires date bounds")
        elif self.search_terms and not self.arxiv_query:
            raise ValueError("Topic search terms require an arXiv query")
        elif self.arxiv_query and not self.search_terms:
            raise ValueError("arXiv query requires search terms")
        return self


class PaperMetadata(Record):
    arxiv_id: Text
    version: int = Field(ge=1)
    title: Text
    authors: list[Text] = Field(min_length=1)
    abstract: Text
    categories: list[Text] = Field(min_length=1)
    published: date
    updated: date
    abstract_url: HttpUrl
    pdf_url: HttpUrl

    @model_validator(mode="after")
    def valid_dates(self) -> "PaperMetadata":
        if self.updated < self.published:
            raise ValueError("Update date cannot precede publication")
        return self


class Candidate(Record):
    paper: PaperMetadata
    score: float | None = Field(default=None, ge=-1, le=1)


class ParsedBlock(Record):
    block_id: Text
    page: int = Field(ge=1)
    section: Text
    kind: Literal["heading", "body", "caption"]
    text: Text
    bbox: tuple[float, float, float, float]


class ParsedSection(Record):
    title: Text
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_pages(self) -> "ParsedSection":
        if self.page_end < self.page_start:
            raise ValueError("Section page span must be ordered")
        return self


class ParsedPaper(Record):
    schema_version: Literal[3] = 3
    paper: PaperMetadata
    pdf_checksum: Text
    page_count: int = Field(ge=1)
    pages_with_text: int = Field(ge=1)
    abstract: Text
    abstract_source: Literal["pdf", "metadata"]
    sections: list[ParsedSection] = Field(min_length=1)
    references: list[Text] = Field(default_factory=list)
    blocks: list[ParsedBlock] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_provenance(self) -> "ParsedPaper":
        if self.pages_with_text > self.page_count:
            raise ValueError("Text page count cannot exceed PDF page count")
        if any(block.page > self.page_count for block in self.blocks):
            raise ValueError("Block page exceeds PDF page count")
        if any(section.page_end > self.page_count for section in self.sections):
            raise ValueError("Section page exceeds PDF page count")
        return self


class Chunk(Record):
    chunk_id: Text
    arxiv_id: Text
    version: int = Field(ge=1)
    section: Text
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: Text
    embedding_tokens: int = Field(ge=1, le=224)
    source_block_ids: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_pages(self) -> "Chunk":
        if self.page_end < self.page_start:
            raise ValueError("Page range must be ordered")
        return self


class EvidenceClaim(Record):
    text: Text
    chunk_ids: list[Text] = Field(min_length=1)


class EvidenceNote(Record):
    facet: Literal["problem", "method", "result", "limitation"]
    text: Text
    chunk_id: Text
    page: int = Field(ge=1)
    section: Text
    source_block_ids: list[Text] = Field(min_length=1)
    distance: float | None = Field(default=None, ge=0)


class EvidenceBatch(Record):
    batch_number: int = Field(ge=1)
    chunk_ids: list[Text] = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    qwen_tokens: int = Field(ge=1)


class Briefing(Record):
    paper: PaperMetadata
    plain_english_summary: EvidenceClaim
    problem_statement: EvidenceClaim
    method: list[EvidenceClaim] = Field(min_length=1)
    key_results: list[EvidenceClaim] = Field(min_length=1)
    limitations_status: Literal["reported", "not_found"]
    limitations: list[EvidenceClaim]
    limitations_note: Text
    processing_notes: list[Text] = Field(default_factory=list)
    follow_up_questions: list[Text] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def explicit_limitations(self) -> "Briefing":
        if (self.limitations_status == "reported") != bool(self.limitations):
            raise ValueError("Reported limitations need evidence; not_found needs an empty list")
        if len({question.casefold() for question in self.follow_up_questions}) != 3:
            raise ValueError("Follow-up questions must be distinct")
        return self


class Citation(Record):
    chunk_id: Text
    arxiv_id: Text
    version: int = Field(ge=1)
    page: int = Field(ge=1)
    section: Text


class QAQuote(Record):
    chunk_id: Text
    source_quote: Text


class QAAnswer(Record):
    status: Literal["answered", "insufficient_evidence"]
    text: Text
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def answered_requires_citations(self) -> "QAAnswer":
        if self.status == "answered" and not self.citations:
            raise ValueError("An answered question requires citations")
        return self


class ConversationTurn(Record):
    question: Text
    answer: QAAnswer


class WorkflowError(Record):
    stage: Stage
    code: Text
    message: Text
    recovery: Text
    retryable: bool = False


class RetrievalIndex(Record):
    collection: Text
    fingerprint: Text
    chunk_count: int = Field(ge=1)


class EvidenceBundle(Record):
    schema_version: Literal[1] = 1
    paper: PaperMetadata
    index: RetrievalIndex
    notes: list[EvidenceNote] = Field(min_length=1)
    candidate_notes: list[EvidenceNote] = Field(default_factory=list)
    batch_audit: list[EvidenceBatch] = Field(default_factory=list)
    processed_chunk_count: int = Field(default=0, ge=0)
    limitations_status: Literal["reported", "not_found"]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_limitations(self) -> "EvidenceBundle":
        facets = {note.facet for note in self.notes}
        if not {"problem", "method", "result"} <= facets:
            raise ValueError("Evidence bundle needs problem, method, and result passages")
        found = "limitation" in facets
        if (self.limitations_status == "reported") != found:
            raise ValueError("Limitations status must match extracted evidence")
        if self.batch_audit:
            covered = [chunk_id for batch in self.batch_audit for chunk_id in batch.chunk_ids]
            if len(covered) != self.processed_chunk_count or len(set(covered)) != len(covered):
                raise ValueError("Evidence batches must cover each processed chunk once")
            candidates = {(note.facet, note.chunk_id): note for note in self.candidate_notes}
            if any(note.chunk_id not in covered for note in self.candidate_notes):
                raise ValueError("Evidence candidates must come from processed batches")
            if any(candidates.get((note.facet, note.chunk_id)) != note for note in self.notes):
                raise ValueError("Reduced evidence notes must retain candidate citations")
        return self


class SessionState(Record):
    schema_version: Literal[1] = 1
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_mode: Literal["synthetic", "live"] = "live"
    user_input: Text
    stage: Stage = Stage.UNDERSTAND
    status: Literal["running", "ready", "failed"] = "running"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    intent: Intent | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    selected_paper: PaperMetadata | None = None
    selection_rank: int | None = Field(default=None, ge=1, le=10)
    selection_source: Literal["lookup", "top_ranked", "override"] | None = None
    pdf_path: str | None = None
    pdf_checksum: str | None = None
    pdf_pages: int | None = Field(default=None, ge=1)
    pdf_size_bytes: int | None = Field(default=None, ge=1)
    parsed_path: str | None = None
    sections: list[str] = Field(default_factory=list)
    abstract_text: Text | None = None
    abstract_source: Literal["pdf", "metadata"] | None = None
    references_found: bool = False
    parsed_block_count: int | None = Field(default=None, ge=1)
    parsed_pages_with_text: int | None = Field(default=None, ge=1)
    parsed_reference_count: int | None = Field(default=None, ge=0)
    index: RetrievalIndex | None = None
    evidence_notes: list[EvidenceNote] = Field(default_factory=list)
    evidence_path: str | None = None
    limitations_evidence_status: Literal["reported", "not_found"] | None = None
    briefing: Briefing | None = None
    output_paths: list[str] = Field(default_factory=list)
    question: Text | None = None
    retrieval_query: Text | None = None
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    answer: QAAnswer | None = None
    answer_support_quotes: list[QAQuote] = Field(default_factory=list)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    search_broadened: bool = False
    retry_counts: dict[str, int] = Field(default_factory=dict)
    stage_history: list[Stage] = Field(default_factory=list)
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: WorkflowError | None = None
