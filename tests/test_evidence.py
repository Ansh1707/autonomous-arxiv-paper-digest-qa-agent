import pytest

from arxiv_agent.contracts import (
    Chunk,
    EvidenceBundle,
    RetrievalIndex,
    SessionState,
    Stage,
)
from arxiv_agent.graph import build_evidence_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.evidence import EvidenceExtractor
from arxiv_agent.services.synthetic import SyntheticServices, synthetic_paper


def chunks(with_limitation=True):
    contents = [
        ("Abstract", 1, "Existing systems face a costly memory constraint in adaptation."),
        ("Our Method", 2, "We propose a compact method and freeze the original weights."),
        ("Our Method", 2, "The method trains low rank matrices for each new task."),
        ("Results", 3, "The method improves accuracy by 3% in our evaluation."),
        ("Results", 3, "The model achieves 91% accuracy and reduces parameters."),
    ]
    if with_limitation:
        contents.append(
            (
                "Discussion",
                4,
                "Our method has a limitation: different tasks cannot be batched together.",
            )
        )
    contents.append(("References", 5, "A reference describes a limitation of another model."))
    return [
        Chunk(
            chunk_id=f"paper-p{page}-c{number}",
            arxiv_id="0000.00000",
            version=1,
            section=section,
            page_start=page,
            page_end=page,
            text=text,
            embedding_tokens=20,
            source_block_ids=[f"p{page}-b{number}"],
        )
        for number, (section, page, text) in enumerate(contents)
    ]


class FakeStore:
    def __init__(self, corpus):
        self.corpus = corpus

    def all_chunks(self, index):
        return self.corpus

    def query(self, index, question, limit):
        return [(chunk, 0.5) for chunk in reversed(self.corpus[:limit])]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def extractor(settings, corpus):
    return EvidenceExtractor(settings, FakeStore(corpus), FakeTokenizer())


def state(corpus):
    return SessionState(
        user_input="0000.00000v1",
        selected_paper=synthetic_paper(),
        index=RetrievalIndex(collection="fake", fingerprint="fixture", chunk_count=len(corpus)),
    )


def test_extracts_verbatim_source_linked_facets_and_persists(settings):
    corpus = chunks()
    path, bundle = extractor(settings, corpus).extract(state(corpus))
    assert path.is_file()
    assert EvidenceBundle.model_validate_json(path.read_text()) == bundle
    assert [note.facet for note in bundle.notes] == [
        "problem",
        "method",
        "method",
        "result",
        "result",
        "limitation",
    ]
    assert bundle.limitations_status == "reported"
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    for note in bundle.notes:
        source = by_id[note.chunk_id]
        assert note.text == source.text
        assert note.page == source.page_start
        assert note.section == source.section
        assert note.source_block_ids == source.source_block_ids
    assert all(note.section != "References" for note in bundle.notes)


def test_missing_explicit_limitation_is_not_invented(settings):
    corpus = chunks(with_limitation=False)
    _, bundle = extractor(settings, corpus).extract(state(corpus))
    assert bundle.limitations_status == "not_found"
    assert not any(note.facet == "limitation" for note in bundle.notes)
    assert bundle.warnings


def test_all_body_chunks_are_batched_and_late_results_keep_citations(settings):
    settings = settings.model_copy(update={"context_tokens": 1024})
    corpus = chunks(with_limitation=False)[:-1]
    filler = "The paper explains the training setup and evaluation design in detail. " * 6
    for number in range(20):
        page = 4 + number
        section = "Experiments" if number < 18 else "Results"
        text = filler
        if number == 19:
            text = (
                "On Dataset Z, our method reaches 91% accuracy compared with the "
                "baseline's 87% accuracy, a four percentage point improvement. " + filler
            )
        corpus.append(
            Chunk(
                chunk_id=f"paper-p{page}-late{number}",
                arxiv_id="0000.00000",
                version=1,
                section=section,
                page_start=page,
                page_end=page,
                text=text,
                embedding_tokens=100,
                source_block_ids=[f"p{page}-late{number}"],
            )
        )
    corpus.append(
        Chunk(
            chunk_id="paper-p25-ref",
            arxiv_id="0000.00000",
            version=1,
            section="References",
            page_start=25,
            page_end=25,
            text="A cited work reports a 99% improvement on Dataset Z.",
            embedding_tokens=20,
            source_block_ids=["p25-ref"],
        )
    )
    _, bundle = extractor(settings, corpus).extract(state(corpus))
    covered = [chunk_id for batch in bundle.batch_audit for chunk_id in batch.chunk_ids]
    assert len(bundle.batch_audit) > 1
    assert bundle.processed_chunk_count == len(corpus) - 1
    assert set(covered) == {chunk.chunk_id for chunk in corpus[:-1]}
    assert all(batch.qwen_tokens <= settings.context_tokens - 512 for batch in bundle.batch_audit)
    assert any(note.chunk_id == "paper-p23-late19" for note in bundle.candidate_notes)
    assert any(note.chunk_id == "paper-p23-late19" for note in bundle.notes)
    assert all(note.chunk_id != "paper-p25-ref" for note in bundle.candidate_notes)
    late = next(note for note in bundle.notes if note.chunk_id == "paper-p23-late19")
    assert "Dataset Z" in late.text and "91%" in late.text and "87%" in late.text


def test_missing_or_mismatched_index_fails_clearly(settings):
    corpus = chunks()
    evidence_extractor = extractor(settings, corpus)
    with pytest.raises(StageFailure, match="selected paper and its index"):
        evidence_extractor.extract(SessionState(user_input="paper"))
    bad = state(corpus)
    bad.index = RetrievalIndex(collection="fake", fingerprint="fixture", chunk_count=1)
    with pytest.raises(StageFailure, match="do not match"):
        evidence_extractor.extract(bad)


def test_evidence_graph_stops_before_briefing(settings):
    result = SessionState.model_validate(
        build_evidence_graph(SyntheticServices("lookup"), settings).invoke(
            SessionState(user_input="synthetic input", execution_mode="synthetic"),
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    )
    assert result.error is None
    assert result.stage_history[-1] == Stage.EVIDENCE
    assert result.evidence_notes
    assert result.briefing is None
