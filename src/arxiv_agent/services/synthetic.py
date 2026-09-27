"""Fictional, deterministic fixtures for graph verification; no network or model calls."""

from datetime import date
from typing import Literal

from arxiv_agent.contracts import (
    Briefing,
    Candidate,
    Chunk,
    Citation,
    ConversationTurn,
    EvidenceClaim,
    EvidenceNote,
    Intent,
    PaperMetadata,
    QAAnswer,
    RetrievalIndex,
    SessionState,
    Stage,
)
from arxiv_agent.services.base import StageFailure

Scenario = Literal["success", "empty-search", "retry-search", "parse-failure", "transient-download"]
SCENARIOS = ("success", "empty-search", "retry-search", "parse-failure", "transient-download")


def synthetic_paper() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="0000.00000",
        version=1,
        title="[SYNTHETIC] Graph routing fixture",
        authors=["Fictional Author"],
        abstract="Synthetic text for testing the workflow only.",
        categories=["cs.AI"],
        published=date(2024, 1, 1),
        updated=date(2024, 1, 1),
        abstract_url="https://arxiv.org/abs/0000.00000v1",
        pdf_url="https://arxiv.org/pdf/0000.00000v1",
    )


def synthetic_chunk() -> Chunk:
    return Chunk(
        chunk_id="synthetic-v1-p1-c1",
        arxiv_id="0000.00000",
        version=1,
        section="Synthetic method",
        page_start=1,
        page_end=1,
        text="This fictional fixture demonstrates graph routing, not scientific findings.",
        embedding_tokens=15,
        source_block_ids=["synthetic-p1-b1"],
    )


class SyntheticServices:
    def __init__(self, intent: Literal["lookup", "topic"], scenario: Scenario = "success"):
        if intent not in {"lookup", "topic"} or scenario not in SCENARIOS:
            raise ValueError("Unknown synthetic intent or scenario")
        if intent != "topic" and scenario in {"empty-search", "retry-search"}:
            raise ValueError("Search scenarios require topic intent")
        self.intent = intent
        self.scenario = scenario

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if state.execution_mode != "synthetic":
            raise StageFailure(
                "DEMO_ONLY",
                "Synthetic services cannot process a live session.",
                "Use the explicitly labeled demo command.",
            )
        paper = synthetic_paper()
        chunk = synthetic_chunk()
        claim = EvidenceClaim(text=chunk.text, chunk_ids=[chunk.chunk_id])
        match stage:
            case Stage.UNDERSTAND:
                # Intent is explicitly supplied by the demo CLI, not inferred from real user input.
                intent = (
                    Intent(kind="lookup", arxiv_id=paper.arxiv_id)
                    if self.intent == "lookup"
                    else Intent(
                        kind="topic",
                        query="synthetic graph routing",
                        search_terms=["synthetic graph routing"],
                        arxiv_query='all:"synthetic graph routing"',
                    )
                )
                return {
                    "intent": intent,
                    "warnings": [
                        "SYNTHETIC DEMO: no paper was retrieved, parsed, embedded, or summarized."
                    ],
                }
            case Stage.LOOKUP:
                return {"selected_paper": paper}
            case Stage.SEARCH:
                empty = self.scenario == "empty-search" or (
                    self.scenario == "retry-search" and not state.search_broadened
                )
                return {"candidates": [] if empty else [Candidate(paper=paper)]}
            case Stage.BROADEN:
                return {"search_broadened": True}
            case Stage.RANK:
                if not state.candidates:
                    raise StageFailure("NO_CANDIDATES", "No candidates.", "Refine the topic.")
                return {
                    "candidates": [Candidate(paper=state.candidates[0].paper, score=1)],
                    "selected_paper": state.candidates[0].paper,
                }
            case Stage.DOWNLOAD:
                if self.scenario == "transient-download" and not state.retry_counts.get(
                    stage.value
                ):
                    raise StageFailure(
                        "DOWNLOAD_TIMEOUT",
                        "Simulated download timeout.",
                        "Retry the download.",
                        retryable=True,
                    )
                return {
                    "pdf_path": "synthetic://paper.pdf",
                    "pdf_checksum": "synthetic-checksum",
                    "pdf_pages": 1,
                    "pdf_size_bytes": 1,
                }
            case Stage.PARSE:
                if self.scenario == "parse-failure":
                    raise StageFailure(
                        "UNREADABLE_PDF",
                        "Simulated scanned or unreadable PDF.",
                        "Check the PDF and install Tesseract with English data for OCR.",
                    )
                return {
                    "parsed_path": "synthetic://parsed.json",
                    "sections": ["Abstract", "Synthetic method", "References"],
                    "abstract_text": "Synthetic abstract.",
                    "abstract_source": "pdf",
                    "references_found": True,
                    "parsed_block_count": 1,
                    "parsed_pages_with_text": 1,
                }
            case Stage.INDEX:
                return {
                    "index": RetrievalIndex(
                        collection="synthetic-only", fingerprint="synthetic-v1", chunk_count=1
                    )
                }
            case Stage.EVIDENCE:
                return {
                    "evidence_notes": [EvidenceNote(
                        facet="method", text=chunk.text, chunk_id=chunk.chunk_id,
                        page=chunk.page_start, section=chunk.section,
                        source_block_ids=chunk.source_block_ids,
                    )],
                    "evidence_path": "synthetic://evidence.json",
                    "limitations_evidence_status": "not_found",
                }
            case Stage.BRIEF:
                return {
                    "output_paths": ["synthetic://briefing.json"],
                    "briefing": Briefing(
                        paper=paper,
                        plain_english_summary=claim,
                        problem_statement=claim,
                        method=[claim],
                        key_results=[claim],
                        limitations_status="not_found",
                        limitations=[],
                        limitations_note="Synthetic fixture; no scientific claims.",
                        follow_up_questions=[
                            "What does this synthetic fixture demonstrate?",
                            "How does the synthetic method work?",
                            "What result does the synthetic example show?",
                        ],
                    )
                }
            case Stage.READY:
                if not state.index or not state.briefing:
                    raise StageFailure(
                        "INCOMPLETE_BRIEFING",
                        "Missing index or briefing.",
                        "Complete earlier stages.",
                    )
                return {"status": "ready"}
            case Stage.RETRIEVE:
                return {"retrieved_chunks": [chunk]}
            case Stage.ANSWER:
                return {
                    "answer": QAAnswer(
                        status="answered",
                        text=chunk.text,
                        citations=[
                            Citation(
                                chunk_id=chunk.chunk_id,
                                arxiv_id=chunk.arxiv_id,
                                version=chunk.version,
                                page=chunk.page_start,
                                section=chunk.section,
                            )
                        ],
                    )
                }
            case Stage.VALIDATE:
                if not state.answer or not state.question:
                    raise StageFailure("NO_ANSWER", "Missing answer or question.", "Retry QA.")
                return {
                    "conversation": [
                        *state.conversation,
                        ConversationTurn(question=state.question, answer=state.answer),
                    ]
                }
            case _:
                raise StageFailure("UNKNOWN_STAGE", f"Unknown stage {stage}.", "Check the graph.")
