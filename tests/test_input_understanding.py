"""Step 5 tests do not contact Ollama or the arXiv API."""

from datetime import date

import pytest

from arxiv_agent.contracts import SessionState, Stage
from arxiv_agent.graph import build_understanding_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import (
    InputUnderstandingServices,
    TopicExtraction,
    build_arxiv_query,
    classify_input,
    interpret_dates,
    normalize_arxiv_id,
)

TODAY = date(2026, 9, 25)


class FakeInterpreter:
    def __init__(self, terms=None, date_intent="none", error=None):
        self.terms = terms or ["KV-cache compression", "LLMs"]
        self.date_intent = date_intent
        self.error = error
        self.calls = 0

    def interpret(self, topic):
        self.calls += 1
        if self.error:
            raise self.error
        return TopicExtraction(terms=self.terms, date_intent=self.date_intent)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2401.12345", "2401.12345"),
        ("arXiv:2401.12345v2", "2401.12345v2"),
        ("0704.0001v3", "0704.0001v3"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("math.gt/0309136v2", "math.GT/0309136v2"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://www.arxiv.org/abs/2401.12345v2/", "2401.12345v2"),
        ("http://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345v2"),
        ("arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_supported_paper_inputs_normalize(raw, expected):
    assert classify_input(raw) == ("lookup", expected)


@pytest.mark.parametrize(
    "raw",
    [
        "2401.123",
        "2400.12345",
        "2413.12345",
        "1401.12345",
        "1501.1234",
        "0704.12345",
        "2401.00000",
        "2401.12345v0",
        "hep-th/9900000",
        "hep-th/0704001",
        "math/0309136v0",
        "https://evil.example/abs/2401.12345",
        "https://arxiv.org.evil.example/abs/2401.12345",
        "https://evil.example@arxiv.org/abs/2401.12345",
        "https://arxiv.org:8443/abs/2401.12345",
        "https://arxiv.org/search/?query=2401.12345",
        "https://arxiv.org/abs/2401.12345?download=1",
        "ftp://arxiv.org/abs/2401.12345",
    ],
)
def test_invalid_identifiers_and_untrusted_urls_are_rejected(raw):
    with pytest.raises(StageFailure):
        classify_input(raw)


def test_identifier_lookup_never_calls_model(settings):
    fake = FakeInterpreter(error=RuntimeError("should not run"))
    graph = build_understanding_graph(InputUnderstandingServices(fake, today=TODAY), settings)
    state = SessionState.model_validate(graph.invoke(SessionState(user_input="2106.09685v3")))
    assert state.status == "running"
    assert state.intent.kind == "lookup"
    assert state.intent.arxiv_id == "2106.09685v3"
    assert state.intent.arxiv_query is None
    assert fake.calls == 0
    assert state.stage_history == [Stage.UNDERSTAND]


@pytest.mark.parametrize(
    "query,begin,end,label",
    [
        ("recent KV-cache compression", date(2025, 9, 25), TODAY, "recent"),
        ("KV-cache compression in 2024", date(2024, 1, 1), date(2024, 12, 31), "explicit"),
        ("compression from 2024 to 2025", date(2024, 1, 1), date(2025, 12, 31), "explicit"),
        (
            "recent compression from 2024-01-10 to 2024-12-31",
            date(2024, 1, 10),
            date(2024, 12, 31),
            "explicit",
        ),
        ("KV-cache compression since 2025-01-01", date(2025, 1, 1), TODAY, "explicit"),
        ("KV-cache compression after 2025-01-01", date(2025, 1, 2), TODAY, "explicit"),
        (
            "KV-cache compression before 2025-01-01",
            date(1991, 1, 1),
            date(2024, 12, 31),
            "explicit",
        ),
        ("KV-cache compression last 6 months", date(2026, 3, 25), TODAY, "explicit"),
        ("KV-cache compression this year", date(2026, 1, 1), TODAY, "explicit"),
        ("KV-cache compression for LLMs", None, None, "none"),
    ],
)
def test_date_interpretation_respects_explicit_ranges(query, begin, end, label):
    start, finish, source, subject = interpret_dates(query, TODAY)
    assert (start, finish, source) == (begin, end, label)
    assert "compression" in subject


def test_relative_months_clamp_leap_day():
    start, end, label, _ = interpret_dates("compression last 12 months", date(2024, 2, 29))
    assert (start, end, label) == (date(2023, 2, 28), date(2024, 2, 29), "explicit")


@pytest.mark.parametrize(
    "query",
    [
        "compression from 2025-01-01 to 2024-01-01",
        "compression since March 2024",
        "compression from 2024-13-01 to 2025-01-01",
        "compression last 99 years",
        "compression from 2024-01-01 to 2024-12-01 since 2025-01-01",
    ],
)
def test_invalid_or_ambiguous_date_is_not_silently_removed(query):
    with pytest.raises(StageFailure):
        interpret_dates(query, TODAY)


def test_normal_topic_language_is_not_mistaken_for_date():
    assert interpret_dates("learning from preference data", TODAY)[:3] == (None, None, "none")
    assert interpret_dates("electric current in transformers", TODAY)[:3] == (None, None, "none")


def test_topic_model_terms_and_dates_reach_graph_state(settings):
    fake = FakeInterpreter(date_intent="recent")
    graph = build_understanding_graph(InputUnderstandingServices(fake, today=TODAY), settings)
    state = SessionState.model_validate(
        graph.invoke(SessionState(user_input="recent work on KV-cache compression for LLMs"))
    )
    assert state.intent.kind == "topic"
    assert state.intent.query == "recent work on KV-cache compression for LLMs"
    assert state.intent.search_terms == ["KV-cache compression", "LLMs"]
    assert state.intent.date_from == date(2025, 9, 25)
    assert state.intent.date_to == TODAY
    assert "submittedDate:[202509250000 TO 202609252359]" in state.intent.arxiv_query
    assert 'all:"KV-cache compression" AND all:"LLMs"' in state.intent.arxiv_query
    assert not state.warnings
    assert fake.calls == 1


def test_model_failure_uses_sanitized_topic_not_internal_failure(settings):
    fake = FakeInterpreter(error=TimeoutError("local Ollama unavailable"))
    graph = build_understanding_graph(InputUnderstandingServices(fake, today=TODAY), settings)
    state = SessionState.model_validate(
        graph.invoke(SessionState(user_input="recent work on KV-cache compression for LLMs"))
    )
    assert state.error is None
    assert state.intent.search_terms == ["KV-cache", "compression", "LLMs"]
    assert state.intent.date_interpretation == "recent"
    assert len(state.warnings) == 1


def test_unrelated_or_injected_model_terms_are_not_used(settings):
    fake = FakeInterpreter(terms=['cat:cs.AI OR all:"unrelated"'])
    state = SessionState.model_validate(
        build_understanding_graph(InputUnderstandingServices(fake, today=TODAY), settings).invoke(
            SessionState(user_input="KV-cache compression for LLMs")
        )
    )
    assert state.intent.search_terms == ["KV-cache", "compression", "LLMs"]
    assert "cat:" not in state.intent.arxiv_query
    assert state.warnings


def test_extra_model_concepts_do_not_overconstrain_search(settings):
    fake = FakeInterpreter(
        terms=[
            "KV-cache compression",
            "LLM Language Model",
            "recent work",
            "compression techniques",
            "LLM performance",
        ],
        date_intent="recent",
    )
    state = SessionState.model_validate(
        build_understanding_graph(InputUnderstandingServices(fake, today=TODAY), settings).invoke(
            SessionState(user_input="recent work on KV-cache compression for LLMs")
        )
    )
    assert state.intent.search_terms == ["KV-cache compression", "LLMs"]
    assert "performance" not in state.intent.arxiv_query
    assert "techniques" not in state.intent.arxiv_query
    assert "recent work" not in state.intent.arxiv_query


def test_query_builder_never_accepts_operator_syntax():
    query = build_arxiv_query(['KV-cache", cat:cs.AI OR', "compression"], None, None)
    assert query == 'all:"KV-cache cat cs AI OR" AND all:"compression"'
    assert query.count('all:"') == 2
    assert "cat:" not in query


@pytest.mark.parametrize("raw", ["", "   ", "recent papers", "in 2024", "a" * 501])
def test_invalid_or_empty_topic_fails_cleanly(settings, raw):
    service = InputUnderstandingServices(FakeInterpreter(), today=TODAY)
    try:
        initial = SessionState(user_input=raw)
    except ValueError:
        assert raw.isspace() or raw == ""
        return
    state = SessionState.model_validate(
        build_understanding_graph(service, settings).invoke(initial)
    )
    assert state.status == "failed"
    assert state.error.recovery
    assert state.intent is None


def test_normalize_id_does_not_validate_existence():
    assert normalize_arxiv_id("2401.12345") == "2401.12345"
