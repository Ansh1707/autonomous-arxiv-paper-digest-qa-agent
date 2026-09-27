"""Full-body, bounded, extractive evidence selection for the briefing stage."""

import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from arxiv_agent.contracts import (
    Chunk,
    EvidenceBatch,
    EvidenceBundle,
    EvidenceNote,
    RetrievalIndex,
    SessionState,
    Stage,
)
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.indexing import ArxivIndexServices, ChromaIndexStore
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)
EVIDENCE_VERSION = 5
FACETS = (
    (
        "problem",
        1,
        "What problem or motivation does the paper address and why does it matter?",
        ("problem", "introduction", "abstract", "motivation"),
        re.compile(r"\b(problem|challenge|expensive|cost|difficult|need|motivat)", re.I),
    ),
    (
        "method",
        2,
        "What method or architecture does the paper propose and how does it work?",
        (
            "our method", "model architecture", "method", "approach", "framework",
            "applying", "model", "training",
        ),
        re.compile(r"\b(propos|method|approach|architect|adapt|train|attention)", re.I),
    ),
    (
        "result",
        2,
        "What key experimental results, measured comparisons, or claims does the paper report?",
        (
            "performance",
            "result",
            "experiment",
            "evaluation",
            "translation",
            "parsing",
            "variation",
            "conclusion",
        ),
        re.compile(r"\b(result|outperform|improv|reduc|evaluat|compar|achiev)", re.I),
    ),
)
LIMITATION = re.compile(
    r"\b(?:has (?:its|a) limitations?|limitations? (?:include|is|are)|"
    r"not straightforward|do not expect|our method cannot|"
    r"our model cannot|we cannot|we do not evaluate|we did not evaluate)\b",
    re.I,
)
EXCLUDED = {"references", "bibliography", "acknowledgments", "acknowledgements"}


def _eligible(chunk: Chunk) -> bool:
    return chunk.section.casefold() not in EXCLUDED and len(chunk.text.strip()) >= 40


class EvidenceExtractor:
    def __init__(self, settings: Settings, store: ChromaIndexStore | None = None, tokenizer=None):
        self.settings = settings
        self.store = store or ChromaIndexStore(settings)
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                path = snapshot_download(
                    self.settings.tokenizer_model,
                    revision=self.settings.tokenizer_revision,
                    cache_dir=str(self.settings.model_cache_dir.resolve()),
                    local_files_only=True,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    path, use_fast=True, local_files_only=True
                )
            except (OSError, ValueError) as exc:
                raise StageFailure(
                    "QWEN_TOKENIZER_MISSING",
                    "The pinned local Qwen tokenizer is unavailable for evidence batching.",
                    "Run 'python -m arxiv_agent doctor --download-models' once online.",
                ) from exc
        return self._tokenizer

    def _batches(self, chunks: list[Chunk]) -> list[tuple[EvidenceBatch, list[Chunk]]]:
        # The extractor is deterministic, but batches also fit a future Qwen prompt.
        # Reserve at least 512 tokens for instructions, schema, and response.
        budget = min(2048, self.settings.context_tokens - 512)
        batches: list[tuple[EvidenceBatch, list[Chunk]]] = []
        current: list[Chunk] = []
        token_count = 0

        def flush() -> None:
            nonlocal current, token_count
            if current:
                batches.append(
                    (
                        EvidenceBatch(
                            batch_number=len(batches) + 1,
                            chunk_ids=[chunk.chunk_id for chunk in current],
                            page_start=min(chunk.page_start for chunk in current),
                            page_end=max(chunk.page_end for chunk in current),
                            qwen_tokens=token_count,
                        ),
                        current,
                    )
                )
                current = []
                token_count = 0

        for chunk in chunks:
            # The section/id wrapper is included in the bound, not just paper text.
            cost = (
                len(
                    self.tokenizer.encode(
                        f"[{chunk.chunk_id}] {chunk.section}\n{chunk.text}",
                        add_special_tokens=False,
                    )
                )
                + 16
            )
            if cost > budget:
                raise StageFailure(
                    "EVIDENCE_CHUNK_TOO_LONG",
                    "A paper chunk exceeds the Qwen evidence budget.",
                    "Inspect the index and shorten the source chunk.",
                )
            if current and token_count + cost > budget:
                flush()
            current.append(chunk)
            token_count += cost
        flush()
        return batches

    @staticmethod
    def _rank(
        chunk: Chunk,
        facet: str,
        section_terms: tuple[str, ...],
        signal: re.Pattern,
        distance: float | None,
    ) -> tuple:
        section = chunk.section.casefold()
        section_rank = next(
            (i for i, term in enumerate(section_terms) if term in section),
            len(section_terms),
        )
        signal_score = 0 if signal.search(chunk.text) else 1
        prose_score = 0
        if facet == "problem":
            prose_score = (
                0
                if re.search(
                    r"\b(?:drawback|preclud|challeng|difficult|expens|barrier)",
                    chunk.text,
                    re.I,
                )
                else 1
                if re.search(r"\b(?:cost|memory|constraint|limitation)", chunk.text, re.I)
                else 2
            )
            if not chunk.text.rstrip().endswith((".", "!", "?")):
                prose_score += 1
        if facet == "result":
            prose_score = (
                0
                if re.search(
                    r"\b(?:outperform\w*|improv\w*|reduc\w*|achiev\w*|better|comparable)\b",
                    chunk.text,
                    re.I,
                )
                else 1
            )
            if sum(char.isdigit() for char in chunk.text) > len(chunk.text) // 5:
                prose_score += 2
            if re.search(r"\bTable\s+\d+\s*:", chunk.text, re.I):
                prose_score += 1
        return (
            section_rank,
            prose_score,
            signal_score,
            distance if distance is not None else 2.0,
            chunk.page_start,
            chunk.chunk_id,
        )

    @staticmethod
    def _note(facet: str, chunk: Chunk, distance: float | None) -> EvidenceNote:
        return EvidenceNote(
            facet=facet,
            text=chunk.text,
            chunk_id=chunk.chunk_id,
            page=chunk.page_start,
            section=chunk.section,
            source_block_ids=chunk.source_block_ids,
            distance=distance,
        )

    def extract(self, state: SessionState) -> tuple[Path, EvidenceBundle]:
        if not state.index or not state.selected_paper:
            raise StageFailure(
                "MISSING_INDEX",
                "Evidence extraction needs a selected paper and its index.",
                "Run index-paper first.",
            )
        index: RetrievalIndex = state.index
        corpus = {chunk.chunk_id: chunk for chunk in self.store.all_chunks(index)}
        if len(corpus) != index.chunk_count or any(
            chunk.arxiv_id != state.selected_paper.arxiv_id
            or chunk.version != state.selected_paper.version
            for chunk in corpus.values()
        ):
            raise StageFailure(
                "EVIDENCE_SOURCE_MISMATCH",
                "Index chunks do not match the selected paper.",
                "Rebuild the selected paper index.",
            )
        notes: list[EvidenceNote] = []
        used: set[str] = set()
        reference_pages = [
            chunk.page_start
            for chunk in corpus.values()
            if chunk.section.casefold() in {"references", "bibliography"}
        ]
        main_end = (
            min(reference_pages)
            if reference_pages
            else max(chunk.page_start for chunk in corpus.values())
        )
        body = sorted(
            (
                chunk
                for chunk in corpus.values()
                if _eligible(chunk) and chunk.page_start <= main_end
            ),
            key=lambda chunk: (chunk.page_start, chunk.chunk_id),
        )
        if not body:
            raise StageFailure(
                "NO_BODY_EVIDENCE",
                "No eligible main-body text exists in the selected index.",
                "Inspect the parsed PDF and rebuild the index.",
            )
        batches = self._batches(body)
        candidate_notes: list[EvidenceNote] = []
        distances_by_facet: dict[str, dict[str, float]] = {}
        for facet, _, prompt, _, _ in FACETS:
            candidates = self.store.query(index, prompt, self.settings.retrieval_candidates)
            distances = {}
            for chunk, distance in candidates:
                if chunk.chunk_id not in corpus or chunk != corpus[chunk.chunk_id]:
                    raise StageFailure(
                        "EVIDENCE_SOURCE_MISMATCH",
                        "Retrieved chunk differs from stored text.",
                        "Rebuild the selected paper index.",
                    )
                distances[chunk.chunk_id] = distance
            distances_by_facet[facet] = distances

        for _, batch_chunks in batches:
            for facet, _, _, section_terms, signal in FACETS:
                distances = distances_by_facet[facet]
                eligible = [
                    chunk
                    for chunk in batch_chunks
                    if (
                        any(term in chunk.section.casefold() for term in section_terms)
                        and (facet != "result" or signal.search(chunk.text))
                    )
                    or (signal.search(chunk.text) and chunk.chunk_id in distances)
                ]
                eligible.sort(
                    key=lambda chunk: self._rank(
                        chunk, facet, section_terms, signal, distances.get(chunk.chunk_id)
                    )
                )
                # Two per facet per batch preserve alternatives for later reduction.
                candidate_notes.extend(
                    self._note(facet, chunk, distances.get(chunk.chunk_id))
                    for chunk in eligible[:2]
                )
            limit_chunks = [
                chunk
                for chunk in batch_chunks
                if LIMITATION.search(chunk.text)
                and "problem" not in chunk.section.casefold()
                and "introduction" not in chunk.section.casefold()
            ]
            candidate_notes.extend(self._note("limitation", chunk, None) for chunk in limit_chunks)

        for facet, quota, _, section_terms, signal in FACETS:
            distances = distances_by_facet[facet]
            facet_candidates = [
                note
                for note in candidate_notes
                if note.facet == facet and note.chunk_id not in used
            ]
            facet_candidates.sort(
                key=lambda note: self._rank(
                    corpus[note.chunk_id],
                    facet,
                    section_terms,
                    signal,
                    distances.get(note.chunk_id),
                )
            )
            if not facet_candidates:
                raise StageFailure(
                    "INSUFFICIENT_EVIDENCE",
                    f"Could not find source text for the {facet} facet.",
                    "Inspect the parsed PDF and its vector index.",
                )
            chosen = facet_candidates[:quota]
            if facet == "result" and quota > 1:
                # Keep a strong later result, rather than allowing early tables to monopolize
                # the reduced briefing context.
                median_page = body[len(body) // 2].page_start
                later = next(
                    (
                        note
                        for note in facet_candidates[1:]
                        if note.page >= median_page and note.page != chosen[0].page
                    ),
                    None,
                )
                if later is not None:
                    chosen = [chosen[0], later]
            for note in sorted(chosen, key=lambda item: (item.page, item.chunk_id)):
                notes.append(note)
                used.add(note.chunk_id)

        limitations = [note for note in candidate_notes if note.facet == "limitation"]
        limitations.sort(
            key=lambda note: (
                0
                if "limitation" in note.section.casefold()
                or re.search(r"\blimitation", note.text, re.I)
                else 1,
                0 if re.search(r"\b(?:our|we|lora|the transformer)\b", note.text, re.I) else 1,
                note.page,
                note.chunk_id,
            )
        )
        if limitations:
            notes.append(
                next((note for note in limitations if note.chunk_id not in used), limitations[0])
            )
        status = "reported" if limitations else "not_found"
        warnings = []
        if status == "not_found":
            warnings.append(
                "No explicit limitation passage was detected; the later briefing must say "
                "limitations were not found, rather than inventing one."
            )
        bundle = EvidenceBundle(
            paper=state.selected_paper,
            index=index,
            notes=notes,
            candidate_notes=candidate_notes,
            batch_audit=[audit for audit, _ in batches],
            processed_chunk_count=len(body),
            limitations_status=status,
            warnings=warnings,
        )
        safe_id = f"{bundle.paper.arxiv_id}v{bundle.paper.version}".replace("/", "_")
        output = (
            self.settings.data_dir
            / "evidence"
            / f"{safe_id}-{index.fingerprint[:12]}-e{EVIDENCE_VERSION}.json"
        )
        temporary = None
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", dir=output.parent, encoding="utf-8", delete=False) as file:
                temporary = Path(file.name)
                file.write(bundle.model_dump_json(indent=2))
            temporary.replace(output)
        except OSError as exc:
            raise StageFailure(
                "EVIDENCE_STORAGE_ERROR",
                "Could not save extracted evidence.",
                "Check free disk space and data/evidence permissions.",
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return output.resolve(), bundle


class ArxivEvidenceServices:
    def __init__(
        self,
        settings: Settings,
        understanding,
        *,
        selection_rank: int = 1,
        extractor: EvidenceExtractor | None = None,
        **kwargs,
    ):
        self.indexing = ArxivIndexServices(
            settings, understanding, selection_rank=selection_rank, **kwargs
        )
        self.store = self.indexing.store
        self.extractor = extractor or EvidenceExtractor(settings, self.store)

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage == Stage.EVIDENCE:
            path, bundle = self.extractor.extract(state)
            return {
                "evidence_notes": bundle.notes,
                "evidence_path": str(path),
                "limitations_evidence_status": bundle.limitations_status,
                "warnings": [*state.warnings, *bundle.warnings],
            }
        return self.indexing.run_stage(stage, state)
