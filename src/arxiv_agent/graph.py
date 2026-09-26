"""Explicit LangGraph orchestration, independent of live or synthetic service implementations."""

import logging
from datetime import UTC, datetime
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from arxiv_agent.contracts import SessionState, Stage, WorkflowError
from arxiv_agent.services.base import StageFailure, WorkflowServices
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)


def _node(stage: Stage, services: WorkflowServices, settings: Settings):
    def execute(state: SessionState) -> dict:
        started = perf_counter()
        logger.info("stage=%s event=start session=%s", stage, state.session_id)
        patch = {}
        try:
            patch = services.run_stage(stage, state.model_copy(deep=True))
            # Validate unknown keys and nested domain records before LangGraph merges the update.
            merged = state.model_dump() | patch | {"error": None}
            validated = SessionState.model_validate(merged)
            required = {
                Stage.UNDERSTAND: (validated.intent,),
                Stage.LOOKUP: (validated.selected_paper,),
                Stage.RANK: (validated.selected_paper,),
                Stage.DOWNLOAD: (
                    validated.pdf_path,
                    validated.pdf_checksum,
                    validated.pdf_pages,
                    validated.pdf_size_bytes,
                ),
                Stage.PARSE: (
                    validated.parsed_path,
                    validated.sections,
                    validated.abstract_text,
                    validated.abstract_source,
                    validated.parsed_block_count,
                    validated.parsed_pages_with_text,
                ),
                Stage.INDEX: (validated.index,),
                Stage.EVIDENCE: (
                    validated.evidence_notes,
                    validated.evidence_path,
                    validated.limitations_evidence_status,
                ),
                Stage.BRIEF: (validated.briefing, validated.output_paths),
                Stage.READY: (validated.index, validated.briefing),
                Stage.ANSWER: (validated.answer,),
                Stage.VALIDATE: (validated.conversation,),
            }.get(stage, ())
            if not all(required):
                raise StageFailure(
                    "INCOMPLETE_STAGE",
                    f"Stage {stage} omitted required outputs.",
                    "Complete the stage outputs before proceeding.",
                )
            patch = {key: getattr(validated, key) for key in patch}
            patch["error"] = None
            if stage == Stage.BROADEN:
                patch["search_broadened"] = True
        except StageFailure as exc:
            patch = {
                "error": WorkflowError(
                    stage=stage,
                    code=exc.code,
                    message=str(exc),
                    recovery=exc.recovery,
                    retryable=exc.retryable,
                )
            }
        except ValidationError as exc:
            patch = {
                "error": WorkflowError(
                    stage=stage,
                    code="INVALID_STATE",
                    message=f"Invalid stage output: {exc}",
                    recovery="Check the stage implementation against the state contracts.",
                )
            }
        except Exception:
            logger.exception("stage=%s event=unexpected_failure", stage)
            patch = {
                "error": WorkflowError(
                    stage=stage,
                    code="INTERNAL_ERROR",
                    message="Unexpected stage failure.",
                    recovery="Inspect the local error log and correct the service implementation.",
                )
            }
        retry_counts = dict(state.retry_counts)
        if patch.get("error") and patch["error"].retryable:
            retry_counts[stage.value] = retry_counts.get(stage.value, 0) + 1
        elapsed = perf_counter() - started
        timings = dict(state.timings_seconds)
        timings[stage.value] = timings.get(stage.value, 0) + elapsed
        patch.update(
            stage=stage,
            updated_at=datetime.now(UTC),
            stage_history=[*state.stage_history, stage],
            retry_counts=retry_counts,
            timings_seconds=timings,
        )
        logger.info(
            "stage=%s event=%s elapsed_seconds=%.4f retries=%s/%s",
            stage,
            "error" if patch.get("error") else "complete",
            elapsed,
            retry_counts.get(stage.value, 0),
            settings.max_retries,
        )
        return patch

    return execute


def _handle_error(state: SessionState) -> dict:
    return {
        "status": "failed",
        "stage": Stage.ERROR,
        "error": state.error
        or WorkflowError(
            stage=state.stage,
            code="NO_RESULTS",
            message="No candidates after one broader search.",
            recovery="Refine the research topic or supply a specific arXiv ID.",
        ),
        "stage_history": [*state.stage_history, Stage.ERROR],
        "updated_at": datetime.now(UTC),
    }


def _route(state: SessionState, next_stage: str, settings: Settings) -> str:
    if state.error:
        if (
            state.error.retryable
            and state.retry_counts.get(state.stage.value, 0) <= settings.max_retries
        ):
            return state.stage.value
        return Stage.ERROR.value
    return next_stage


def build_digest_graph(services: WorkflowServices, settings: Settings):
    """Build the ingestion/briefing skeleton. No service makes real calls implicitly."""
    graph = StateGraph(SessionState)
    stages = [
        Stage.UNDERSTAND,
        Stage.LOOKUP,
        Stage.SEARCH,
        Stage.BROADEN,
        Stage.RANK,
        Stage.DOWNLOAD,
        Stage.PARSE,
        Stage.INDEX,
        Stage.EVIDENCE,
        Stage.BRIEF,
        Stage.READY,
    ]
    for stage in stages:
        graph.add_node(stage.value, _node(stage, services, settings))
    graph.add_node(Stage.ERROR.value, _handle_error)
    graph.add_edge(START, Stage.UNDERSTAND.value)

    def route_intent(state: SessionState) -> str:
        next_stage = (
            Stage.LOOKUP if state.intent and state.intent.kind == "lookup" else Stage.SEARCH
        )
        return _route(state, next_stage.value, settings)

    def route_search(state: SessionState) -> str:
        if state.error:
            return _route(state, Stage.RANK.value, settings)
        if state.candidates:
            return Stage.RANK.value
        return Stage.ERROR.value if state.search_broadened else Stage.BROADEN.value

    graph.add_conditional_edges(
        Stage.UNDERSTAND.value,
        route_intent,
        [
            Stage.LOOKUP.value,
            Stage.SEARCH.value,
            Stage.UNDERSTAND.value,
            Stage.ERROR.value,
        ],
    )
    graph.add_conditional_edges(
        Stage.SEARCH.value,
        route_search,
        [
            Stage.RANK.value,
            Stage.BROADEN.value,
            Stage.SEARCH.value,
            Stage.ERROR.value,
        ],
    )
    transitions = {
        Stage.LOOKUP: Stage.DOWNLOAD,
        Stage.BROADEN: Stage.SEARCH,
        Stage.RANK: Stage.DOWNLOAD,
        Stage.DOWNLOAD: Stage.PARSE,
        Stage.PARSE: Stage.INDEX,
        Stage.INDEX: Stage.EVIDENCE,
        Stage.EVIDENCE: Stage.BRIEF,
        Stage.BRIEF: Stage.READY,
    }
    for current, following in transitions.items():
        graph.add_conditional_edges(
            current.value,
            lambda state, target=following.value: _route(state, target, settings),
            [following.value, current.value, Stage.ERROR.value],
        )
    graph.add_conditional_edges(
        Stage.READY.value,
        lambda state: _route(state, END, settings),
        [END, Stage.READY.value, Stage.ERROR.value],
    )
    graph.add_edge(Stage.ERROR.value, END)
    return graph.compile()


def build_understanding_graph(services: WorkflowServices, settings: Settings):
    """Run only Step 5 without implying later retrieval stages are implemented."""
    graph = StateGraph(SessionState)
    graph.add_node(Stage.UNDERSTAND.value, _node(Stage.UNDERSTAND, services, settings))
    graph.add_node(Stage.ERROR.value, _handle_error)
    graph.add_edge(START, Stage.UNDERSTAND.value)
    graph.add_conditional_edges(
        Stage.UNDERSTAND.value,
        lambda state: _route(state, END, settings),
        [END, Stage.UNDERSTAND.value, Stage.ERROR.value],
    )
    graph.add_edge(Stage.ERROR.value, END)
    return graph.compile()


def _build_metadata_graph(
    services: WorkflowServices,
    settings: Settings,
    *,
    select: bool,
    download: bool = False,
    parse: bool = False,
    index: bool = False,
    evidence: bool = False,
    brief: bool = False,
):
    graph = StateGraph(SessionState)
    for stage in (Stage.UNDERSTAND, Stage.LOOKUP, Stage.SEARCH, Stage.BROADEN):
        graph.add_node(stage.value, _node(stage, services, settings))
    if select:
        graph.add_node(Stage.RANK.value, _node(Stage.RANK, services, settings))
    if download:
        graph.add_node(Stage.DOWNLOAD.value, _node(Stage.DOWNLOAD, services, settings))
    if parse:
        graph.add_node(Stage.PARSE.value, _node(Stage.PARSE, services, settings))
    if index:
        graph.add_node(Stage.INDEX.value, _node(Stage.INDEX, services, settings))
    if evidence:
        graph.add_node(Stage.EVIDENCE.value, _node(Stage.EVIDENCE, services, settings))
    if brief:
        graph.add_node(Stage.BRIEF.value, _node(Stage.BRIEF, services, settings))
    graph.add_node(Stage.ERROR.value, _handle_error)
    graph.add_edge(START, Stage.UNDERSTAND.value)
    graph.add_conditional_edges(
        Stage.UNDERSTAND.value,
        lambda state: _route(
            state,
            (
                Stage.LOOKUP if state.intent and state.intent.kind == "lookup" else Stage.SEARCH
            ).value,
            settings,
        ),
        [Stage.LOOKUP.value, Stage.SEARCH.value, Stage.UNDERSTAND.value, Stage.ERROR.value],
    )
    graph.add_conditional_edges(
        Stage.LOOKUP.value,
        lambda state: _route(state, Stage.DOWNLOAD.value if download else END, settings),
        [Stage.DOWNLOAD.value if download else END, Stage.LOOKUP.value, Stage.ERROR.value],
    )

    def after_search(state: SessionState) -> str:
        if state.error:
            return _route(state, END, settings)
        if state.candidates:
            return Stage.RANK.value if select else END
        return Stage.ERROR.value if state.search_broadened else Stage.BROADEN.value

    graph.add_conditional_edges(
        Stage.SEARCH.value,
        after_search,
        [
            Stage.RANK.value if select else END,
            Stage.BROADEN.value,
            Stage.SEARCH.value,
            Stage.ERROR.value,
        ],
    )
    graph.add_conditional_edges(
        Stage.BROADEN.value,
        lambda state: _route(state, Stage.SEARCH.value, settings),
        [Stage.SEARCH.value, Stage.BROADEN.value, Stage.ERROR.value],
    )
    graph.add_edge(Stage.ERROR.value, END)
    if select:
        graph.add_conditional_edges(
            Stage.RANK.value,
            lambda state: _route(state, Stage.DOWNLOAD.value if download else END, settings),
            [Stage.DOWNLOAD.value if download else END, Stage.RANK.value, Stage.ERROR.value],
        )
    if download:
        graph.add_conditional_edges(
            Stage.DOWNLOAD.value,
            lambda state: _route(state, Stage.PARSE.value if parse else END, settings),
            [Stage.PARSE.value if parse else END, Stage.DOWNLOAD.value, Stage.ERROR.value],
        )
    if parse:
        graph.add_conditional_edges(
            Stage.PARSE.value,
            lambda state: _route(state, Stage.INDEX.value if index else END, settings),
            [Stage.INDEX.value if index else END, Stage.PARSE.value, Stage.ERROR.value],
        )
    if index:
        graph.add_conditional_edges(
            Stage.INDEX.value,
            lambda state: _route(state, Stage.EVIDENCE.value if evidence else END, settings),
            [Stage.EVIDENCE.value if evidence else END, Stage.INDEX.value, Stage.ERROR.value],
        )
    if evidence:
        graph.add_conditional_edges(
            Stage.EVIDENCE.value,
            lambda state: _route(state, Stage.BRIEF.value if brief else END, settings),
            [Stage.BRIEF.value if brief else END, Stage.EVIDENCE.value, Stage.ERROR.value],
        )
    if brief:
        graph.add_conditional_edges(
            Stage.BRIEF.value,
            lambda state: _route(state, END, settings),
            [END, Stage.BRIEF.value, Stage.ERROR.value],
        )
    return graph.compile()


def build_discovery_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–6 and stop before ranking."""
    return _build_metadata_graph(services, settings, select=False)


def build_selection_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–7 and stop after choosing a versioned paper."""
    return _build_metadata_graph(services, settings, select=True)


def build_download_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–8 and stop after a validated versioned PDF is cached."""
    return _build_metadata_graph(services, settings, select=True, download=True)


def build_parse_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–9 and stop after page-aware PDF text extraction."""
    return _build_metadata_graph(services, settings, select=True, download=True, parse=True)


def build_index_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–10 and stop after persistent provenance-aware vector indexing."""
    return _build_metadata_graph(
        services, settings, select=True, download=True, parse=True, index=True
    )


def build_evidence_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–11 and stop after source-linked evidence extraction."""
    return _build_metadata_graph(
        services, settings, select=True, download=True, parse=True, index=True,
        evidence=True,
    )


def build_briefing_graph(services: WorkflowServices, settings: Settings):
    """Run Steps 5–12 and stop after a source-validated executive briefing."""
    return _build_metadata_graph(
        services, settings, select=True, download=True, parse=True, index=True,
        evidence=True, brief=True,
    )


def build_qa_graph(services: WorkflowServices, settings: Settings):
    """One QA turn per invocation. The future CLI owns the question/exit loop."""
    graph = StateGraph(SessionState)

    def check_ready(state: SessionState) -> dict:
        if (
            state.status != "ready" or not state.index or not state.briefing
            or not state.selected_paper or not state.question
        ):
            return {
                "error": WorkflowError(
                    stage=Stage.QA_CHECK,
                    code="QA_NOT_READY",
                    message="QA requires a successful briefing, index, and nonempty question.",
                    recovery="Complete paper ingestion and supply a question before entering QA.",
                ),
                "stage": Stage.QA_CHECK,
                "updated_at": datetime.now(UTC),
                "stage_history": [*state.stage_history, Stage.QA_CHECK],
            }
        return {
            "error": None,
            "answer": None,
            "answer_support_quotes": [],
            "retrieval_query": None,
            "retrieved_chunks": [],
            "stage": Stage.QA_CHECK,
            "updated_at": datetime.now(UTC),
            "retry_counts": {
                key: value
                for key, value in state.retry_counts.items()
                if key not in {Stage.RETRIEVE.value, Stage.ANSWER.value, Stage.VALIDATE.value}
            },
            "stage_history": [*state.stage_history, Stage.QA_CHECK],
        }

    graph.add_node(Stage.QA_CHECK.value, check_ready)
    graph.add_node(Stage.ERROR.value, _handle_error)
    stages = [Stage.RETRIEVE, Stage.ANSWER, Stage.VALIDATE]
    for stage in stages:
        graph.add_node(stage.value, _node(stage, services, settings))
    graph.add_edge(START, Stage.QA_CHECK.value)
    graph.add_conditional_edges(
        Stage.QA_CHECK.value,
        lambda state: Stage.ERROR.value if state.error else Stage.RETRIEVE.value,
        [Stage.ERROR.value, Stage.RETRIEVE.value],
    )
    for current, following in zip(
        stages, [Stage.ANSWER.value, Stage.VALIDATE.value, END], strict=True
    ):
        graph.add_conditional_edges(
            current.value,
            lambda state, target=following: _route(state, target, settings),
            [following, current.value, Stage.ERROR.value],
        )
    graph.add_edge(Stage.ERROR.value, END)
    return graph.compile()
