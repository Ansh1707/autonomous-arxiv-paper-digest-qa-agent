from datetime import UTC, date, datetime
from types import SimpleNamespace

import arxiv
import requests

from arxiv_agent.contracts import Intent, SessionState, Stage
from arxiv_agent.graph import build_discovery_graph
from arxiv_agent.services.discovery import ArxivDiscoveryServices
from arxiv_agent.services.input_understanding import InputUnderstandingServices, TopicExtraction


class Interpreter:
    def interpret(self, topic):
        return TopicExtraction(terms=["graph neural networks", "biology"], date_intent="explicit")


class FakeResult:
    def __init__(self, short_id="2106.09685v2"):
        self.short_id = short_id
        self.title = "  A   Test Paper "
        self.authors = [SimpleNamespace(name="Ada Example")]
        self.summary = "  A   meaningful abstract. "
        self.primary_category = "cs.LG"
        self.categories = ["cs.LG", "cs.AI"]
        self.published = datetime(2021, 6, 18, tzinfo=UTC)
        self.updated = datetime(2022, 1, 2, tzinfo=UTC)

    def get_short_id(self):
        return self.short_id


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return iter(result)


def run(settings, tmp_path, raw, responses):
    settings = type(settings)(**(
        settings.model_dump() | {
            "data_dir": tmp_path, "candidate_count": 10,
            "candidate_oldest_count": 0, "candidate_phrase_count": 0,
        }
    ))
    client = FakeClient(responses)
    service = ArxivDiscoveryServices(
        settings,
        InputUnderstandingServices(Interpreter(), today=date(2025, 1, 1)),
        client,
    )
    state = SessionState.model_validate(
        build_discovery_graph(service, settings).invoke(
            SessionState(user_input=raw),
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    )
    return state, client, service


def test_lookup_keeps_explicit_version_and_all_metadata(settings, tmp_path):
    state, client, _ = run(settings, tmp_path, "2106.09685v2", [[FakeResult()]])
    assert state.error is None
    assert state.selected_paper.arxiv_id == "2106.09685"
    assert state.selected_paper.version == 2
    assert state.selected_paper.published == date(2021, 6, 18)
    assert state.selected_paper.updated == date(2022, 1, 2)
    assert str(state.selected_paper.pdf_url).endswith("2106.09685v2")
    assert state.selected_paper.categories == ["cs.LG", "cs.AI"]
    assert client.searches[0].id_list == ["2106.09685v2"]
    assert state.stage_history == [Stage.UNDERSTAND, Stage.LOOKUP]


def test_unversioned_resolves_returned_version(settings, tmp_path):
    state, client, _ = run(settings, tmp_path, "2106.09685", [[FakeResult("2106.09685v4")]])
    assert state.selected_paper.version == 4
    assert client.searches[0].id_list == ["2106.09685"]


def test_version_mismatch_and_missing_paper_fail_clearly(settings, tmp_path):
    mismatch, _, _ = run(settings, tmp_path / "a", "2106.09685v1", [[FakeResult()]])
    missing, _, _ = run(settings, tmp_path / "b", "2106.09685", [[]])
    assert mismatch.status == missing.status == "failed"
    assert mismatch.error.code == "VERSION_NOT_FOUND"
    assert missing.error.code == "PAPER_NOT_FOUND"
    assert mismatch.selected_paper is None


def test_topic_caps_candidates_and_preserves_date_on_broadening(settings, tmp_path):
    results = [FakeResult(f"2106.{i:05d}v1") for i in range(1, 15)]
    state, client, _ = run(
        settings,
        tmp_path,
        "graph neural networks in biology from 2024-01-01 to 2024-12-31",
        [[], results],
    )
    assert state.error is None
    assert state.search_broadened
    assert len(state.candidates) == 10
    assert state.selected_paper is None
    assert len(client.searches) == 2
    for search in client.searches:
        assert "submittedDate:[202401010000 TO 202412312359]" in search.query
        assert search.max_results == 10
    assert " OR " in client.searches[1].query
    assert state.stage_history.count(Stage.BROADEN) == 1


def test_topic_discovery_unions_relevance_oldest_and_phrase_pools(settings, tmp_path):
    settings = type(settings)(**(
        settings.model_dump() | {
            "data_dir": tmp_path, "candidate_count": 2,
            "candidate_oldest_count": 2, "candidate_phrase_count": 2,
        }
    ))
    client = FakeClient([
        [FakeResult("2401.00001v1"), FakeResult("2401.00002v1")],
        [FakeResult("2201.00003v1"), FakeResult("2401.00001v1")],
        [FakeResult("2101.00004v1")],
    ])
    service = ArxivDiscoveryServices(settings, InputUnderstandingServices(Interpreter()), client)
    intent = Intent(
        kind="topic", query="chain of thought prompting", search_terms=["chain of thought"],
        arxiv_query='all:"chain of thought"',
        date_from=date(2020, 1, 1), date_to=date(2025, 1, 1),
        date_interpretation="explicit",
    )
    patch = service._search(SessionState(user_input=intent.query, intent=intent))
    assert [item.paper.arxiv_id for item in patch["candidates"]] == [
        "2401.00001", "2401.00002", "2201.00003", "2101.00004",
    ]
    assert len(client.searches) == 3
    assert client.searches[1].sort_by == arxiv.SortCriterion.SubmittedDate
    assert 'all:"chain of thought"' in client.searches[2].query
    assert "submittedDate:[202001010000 TO 202501012359]" in client.searches[2].query


def test_empty_search_broadens_only_once(settings, tmp_path):
    state, client, _ = run(settings, tmp_path, "biology from 2024-01-01 to 2024-12-31", [[], []])
    assert state.error.code == "NO_RESULTS"
    assert len(client.searches) == 2
    assert state.stage_history.count(Stage.BROADEN) == 1


def test_transient_request_retries_twice_then_succeeds(settings, tmp_path):
    state, client, _ = run(
        settings,
        tmp_path,
        "2106.09685",
        [requests.Timeout(), requests.ConnectionError(), [FakeResult()]],
    )
    assert state.error is None
    assert len(client.searches) == 3
    assert state.retry_counts[Stage.LOOKUP.value] == 2


def test_transient_request_stops_after_two_retries(settings, tmp_path):
    state, client, _ = run(
        settings,
        tmp_path,
        "2106.09685",
        [requests.Timeout(), requests.Timeout(), requests.Timeout()],
    )
    assert state.error.code == "ARXIV_NETWORK_ERROR"
    assert state.status == "failed"
    assert len(client.searches) == 3


def test_nonretryable_http_error_stops_immediately(settings, tmp_path):
    state, client, _ = run(
        settings,
        tmp_path,
        "2106.09685",
        [arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 400)],
    )
    assert state.error.code == "ARXIV_API_ERROR"
    assert not state.error.retryable
    assert len(client.searches) == 1


def test_rate_limit_retries_with_bounded_graph_budget(settings, tmp_path):
    limited = arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 429)
    state, client, _ = run(
        settings, tmp_path, "2106.09685", [limited, limited, limited]
    )
    assert state.error.code == "ARXIV_API_ERROR"
    assert state.error.retryable
    assert state.status == "failed"
    assert len(client.searches) == 3


def test_successful_lookup_reuses_cache(settings, tmp_path):
    _, client, service = run(settings, tmp_path, "2106.09685", [[FakeResult()]])
    again = service.run_stage(Stage.LOOKUP, SessionState(user_input="2106.09685", intent={
        "kind": "lookup", "arxiv_id": "2106.09685"
    }))
    assert again["selected_paper"].version == 2
    assert len(client.searches) == 1
