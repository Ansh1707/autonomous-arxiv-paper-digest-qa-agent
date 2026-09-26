import pytest

from arxiv_agent.contracts import SessionState, Stage
from arxiv_agent.graph import build_digest_graph, build_index_graph, build_qa_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.synthetic import SyntheticServices


def run_digest(settings, intent="lookup", scenario="success", services=None):
    return SessionState.model_validate(
        build_digest_graph(services or SyntheticServices(intent, scenario), settings).invoke(
            SessionState(user_input="synthetic input", execution_mode="synthetic"),
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    )


def test_index_graph_stops_before_briefing(settings):
    result = SessionState.model_validate(
        build_index_graph(SyntheticServices("lookup"), settings).invoke(
            SessionState(user_input="synthetic input", execution_mode="synthetic"),
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    )
    assert result.error is None
    assert result.index.chunk_count == 1
    assert result.stage_history[-1] == Stage.INDEX
    assert result.briefing is None


@pytest.mark.parametrize("intent", ["lookup", "topic"])
def test_routes_reach_ready_with_shared_state(settings, intent):
    result = run_digest(settings, intent)
    assert result.status == "ready"
    assert result.briefing.paper == result.selected_paper
    assert result.index.chunk_count == 1
    assert result.error is None
    if intent == "lookup":
        assert Stage.LOOKUP in result.stage_history
        assert Stage.SEARCH not in result.stage_history
        assert Stage.RANK not in result.stage_history
    else:
        assert Stage.SEARCH in result.stage_history
        assert Stage.RANK in result.stage_history
        assert Stage.LOOKUP not in result.stage_history
    assert result.stage_history.index(Stage.PARSE) < result.stage_history.index(Stage.INDEX)
    assert result.stage_history.index(Stage.INDEX) < result.stage_history.index(Stage.BRIEF)
    assert all(value >= 0 for value in result.timings_seconds.values())


def test_empty_search_broadens_once_and_stops(settings):
    result = run_digest(settings, "topic", "empty-search")
    assert result.status == "failed"
    assert result.error.code == "NO_RESULTS"
    assert result.error.recovery
    assert result.stage_history.count(Stage.SEARCH) == 2
    assert result.stage_history.count(Stage.BROADEN) == 1
    assert Stage.DOWNLOAD not in result.stage_history


def test_broader_search_can_recover(settings):
    result = run_digest(settings, "topic", "retry-search")
    assert result.status == "ready"
    assert result.search_broadened
    assert result.stage_history.count(Stage.SEARCH) == 2


def test_parse_failure_never_indexes_or_enters_qa(settings):
    result = run_digest(settings, scenario="parse-failure")
    assert result.status == "failed"
    assert result.error.code == "UNREADABLE_PDF"
    assert Stage.INDEX not in result.stage_history
    assert Stage.BRIEF not in result.stage_history
    assert Stage.READY not in result.stage_history
    assert result.briefing is None
    result.question = "Any question"
    qa = SessionState.model_validate(
        build_qa_graph(SyntheticServices("lookup"), settings).invoke(result)
    )
    assert qa.error.code == "QA_NOT_READY"
    assert Stage.RETRIEVE not in qa.stage_history


def test_transient_failure_retries_and_clears_error(settings):
    result = run_digest(settings, scenario="transient-download")
    assert result.status == "ready"
    assert result.stage_history.count(Stage.DOWNLOAD) == 2
    assert result.error is None


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_permanent_retryable_failure_has_strict_budget(settings, max_retries):
    class AlwaysFails(SyntheticServices):
        def run_stage(self, stage, state):
            if stage == Stage.DOWNLOAD:
                raise StageFailure("TIMEOUT", "Unavailable", "Try again later", retryable=True)
            return super().run_stage(stage, state)

    settings = type(settings)(**(settings.model_dump() | {"max_retries": max_retries}))
    result = run_digest(settings, services=AlwaysFails("lookup"))
    assert result.status == "failed"
    assert result.stage_history.count(Stage.DOWNLOAD) == max_retries + 1
    assert Stage.PARSE not in result.stage_history
    assert result.error.code == "TIMEOUT"


@pytest.mark.parametrize("bad_patch", [{"unknown_field": True}, {"index": {"chunk_count": -1}}])
def test_invalid_service_outputs_fail_closed(settings, bad_patch):
    class Invalid(SyntheticServices):
        def run_stage(self, stage, state):
            if stage == Stage.INDEX:
                return bad_patch
            return super().run_stage(stage, state)

    result = run_digest(settings, services=Invalid("lookup"))
    assert result.status == "failed"
    assert result.error.code == "INVALID_STATE"
    assert Stage.BRIEF not in result.stage_history


def test_missing_required_stage_outputs_do_not_proceed(settings):
    class Incomplete(SyntheticServices):
        def run_stage(self, stage, state):
            return {} if stage == Stage.PARSE else super().run_stage(stage, state)

    result = run_digest(settings, services=Incomplete("lookup"))
    assert result.error.code == "INCOMPLETE_STAGE"
    assert Stage.INDEX not in result.stage_history


def test_unexpected_exception_has_controlled_error(settings):
    class Broken(SyntheticServices):
        def run_stage(self, stage, state):
            raise RuntimeError("Internal implementation error")

    result = run_digest(settings, services=Broken("lookup"))
    assert result.error.code == "INTERNAL_ERROR"
    assert result.status == "failed"


def test_qa_keeps_prior_state_and_appends_turns(settings):
    services = SyntheticServices("lookup")
    state = run_digest(settings)
    identity, briefing, index = state.session_id, state.briefing, state.index
    graph = build_qa_graph(services, settings)
    for question in ["First question?", "And the next one?"]:
        state.question = question
        state = SessionState.model_validate(graph.invoke(state))
    assert state.session_id == identity
    assert state.briefing == briefing
    assert state.index == index
    assert len(state.conversation) == 2
    assert state.answer.citations[0].chunk_id == state.retrieved_chunks[0].chunk_id
    assert state.status == "ready"


def test_qa_has_fresh_retry_budget_each_turn(settings):
    class RetryEachTurn(SyntheticServices):
        def run_stage(self, stage, state):
            if stage == Stage.RETRIEVE and not state.retry_counts.get(stage.value):
                raise StageFailure("TEMP", "Temporary failure", "Retry", retryable=True)
            return super().run_stage(stage, state)

    state = run_digest(settings)
    graph = build_qa_graph(RetryEachTurn("lookup"), settings)
    for _ in range(2):
        state.question = "Question?"
        state = SessionState.model_validate(graph.invoke(state))
        assert state.status == "ready"
        assert state.retry_counts[Stage.RETRIEVE.value] == 1
    assert state.stage_history.count(Stage.RETRIEVE) == 4


def test_synthetic_provider_cannot_claim_live_processing(settings):
    result = build_digest_graph(SyntheticServices("lookup"), settings).invoke(
        SessionState(user_input="2106.09685")
    )
    state = SessionState.model_validate(result)
    assert state.status == "failed"
    assert state.error.code == "DEMO_ONLY"


def test_state_json_round_trip(settings):
    state = run_digest(settings)
    restored = SessionState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.selected_paper.published == state.selected_paper.published
