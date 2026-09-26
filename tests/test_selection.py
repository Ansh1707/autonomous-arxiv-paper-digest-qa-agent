from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from arxiv_agent.contracts import Candidate, Intent, PaperMetadata, SessionState, Stage
from arxiv_agent.graph import build_selection_graph
from arxiv_agent.services.input_understanding import InputUnderstandingServices, TopicExtraction
from arxiv_agent.services.selection import ArxivSelectionServices, EmbeddingRanker


def paper(identifier: str, title: str) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=identifier,
        version=1,
        title=title,
        authors=["A. Author"],
        abstract=f"Abstract of {title}",
        categories=["cs.LG"],
        published="2024-01-01",
        updated="2024-01-02",
        abstract_url=f"https://arxiv.org/abs/{identifier}v1",
        pdf_url=f"https://arxiv.org/pdf/{identifier}v1",
    )


class FixedEncoder:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors)
        self.texts = None

    def encode(self, texts):
        self.texts = texts
        return self.vectors


def test_semantic_ranking_uses_title_and_abstract_and_stable_ties():
    encoder = FixedEncoder(
        [
            [1, 0],  # query
            [0, 1], [1, 0], [1, 0],  # three titles
            [0, 1], [0, 1], [0, 1],  # three abstracts
        ]
    )
    ranker = EmbeddingRanker(encoder)
    intent = Intent(kind="topic", query="graph networks", search_terms=["graph networks"],
                    arxiv_query='all:"graph networks"')
    candidates = [
        Candidate(paper=paper("2401.00001", "unrelated")),
        Candidate(paper=paper("2401.00002", "matching")),
        Candidate(paper=paper("2401.00003", "also matching")),
    ]
    ranked = ranker.rank(intent, candidates)
    assert [item.paper.arxiv_id for item in ranked] == [
        "2401.00002", "2401.00003", "2401.00001"
    ]
    assert ranked[0].score == ranked[1].score == 0.6
    assert ranked[2].score == 0
    assert encoder.texts[0] == "graph networks"
    assert candidates[0].score is None  # no mutation of API-order candidates


class Interpreter:
    def interpret(self, topic):
        return TopicExtraction(terms=["graph networks"], date_intent="none")


class Result:
    def __init__(self, identifier):
        self.identifier = identifier
        self.title = f"Paper {identifier}"
        self.authors = [SimpleNamespace(name="A. Author")]
        self.summary = "Abstract text"
        self.primary_category = "cs.LG"
        self.categories = ["cs.LG"]
        self.published = datetime(2024, 1, 1, tzinfo=UTC)
        self.updated = datetime(2024, 1, 2, tzinfo=UTC)

    def get_short_id(self):
        return self.identifier


class Client:
    def __init__(self, results):
        self.data = results

    def results(self, search):
        return iter(self.data)


def run(settings, raw, results, rank, vectors):
    service = ArxivSelectionServices(
        settings,
        InputUnderstandingServices(Interpreter()),
        client=Client(results),
        selection_rank=rank,
        ranker=EmbeddingRanker(FixedEncoder(vectors)),
    )
    return SessionState.model_validate(
        build_selection_graph(service, settings).invoke(SessionState(user_input=raw))
    )


def test_topic_ranking_and_override_are_in_graph_state(settings):
    results = [Result("2401.00001v1"), Result("2401.00002v1")]
    vectors = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    state = run(settings, "graph networks", results, 2, vectors)
    assert state.error is None
    assert [c.paper.arxiv_id for c in state.candidates] == ["2401.00002", "2401.00001"]
    assert state.selected_paper == state.candidates[1].paper
    assert state.selection_rank == 2
    assert state.selection_source == "override"
    assert state.stage_history == [Stage.UNDERSTAND, Stage.SEARCH, Stage.RANK]


def test_default_selects_top_ranked(settings):
    results = [Result("2401.00001v1"), Result("2401.00002v1")]
    vectors = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    state = run(settings, "graph networks", results, 1, vectors)
    assert state.selected_paper.arxiv_id == "2401.00002"
    assert state.selection_source == "top_ranked"


def test_out_of_range_selection_fails_with_recovery(settings):
    results = [Result("2401.00001v1")]
    state = run(settings, "graph networks", results, 2, [[1, 0], [1, 0], [1, 0]])
    assert state.status == "failed"
    assert state.error.code == "INVALID_SELECTION"
    assert state.selected_paper is None


def test_lookup_skips_ranker_and_rejects_nonfirst_override(settings):
    state = run(settings, "2401.00001v1", [Result("2401.00001v1")], 1, [])
    assert state.error is None
    assert state.selection_source == "lookup"
    assert Stage.RANK not in state.stage_history
    invalid = run(settings, "2401.00001v1", [Result("2401.00001v1")], 2, [])
    assert invalid.error.code == "INVALID_SELECTION"


def test_invalid_embedding_shape_fails_without_selection(settings):
    state = run(
        settings, "graph networks", [Result("2401.00001v1")], 1, [[1, 0]],
    )
    assert state.error.code == "INVALID_EMBEDDINGS"
    assert state.selected_paper is None
