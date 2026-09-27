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


_INGESTION_ORDER = (
    Stage.UNDERSTAND, Stage.SEARCH, Stage.RANK, Stage.DOWNLOAD, Stage.PARSE,
    Stage.INDEX, Stage.EVIDENCE, Stage.BRIEF, Stage.READY,
)


def _build_ingestion_graph(
    services: WorkflowServices, settings: Settings, stop_after: Stage
):
    """Use one routing definition for the full workflow and every CLI checkpoint."""
    if stop_after not in _INGESTION_ORDER:
        raise ValueError(f"Unsupported ingestion stop stage: {stop_after}")
    enabled = _INGESTION_ORDER[:_INGESTION_ORDER.index(stop_after) + 1]
    graph = StateGraph(SessionState)
    graph.add_node(Stage.UNDERSTAND.value, _node(Stage.UNDERSTAND, services, settings))
    if stop_after != Stage.UNDERSTAND:
        for stage in (Stage.LOOKUP, Stage.SEARCH, Stage.BROADEN, *enabled[2:]):
            graph.add_node(stage.value, _node(stage, services, settings))
    graph.add_node(Stage.ERROR.value, _handle_error)
    graph.add_edge(START, Stage.UNDERSTAND.value)
    graph.add_edge(Stage.ERROR.value, END)

    def connect(current: Stage, following: Stage | None) -> None:
        target = following.value if following else END
        graph.add_conditional_edges(
            current.value,
            lambda state, next_node=target: _route(state, next_node, settings),
            [target, current.value, Stage.ERROR.value],
        )

    if stop_after == Stage.UNDERSTAND:
        connect(Stage.UNDERSTAND, None)
        return graph.compile()

    graph.add_conditional_edges(
        Stage.UNDERSTAND.value,
        lambda state: _route(
            state,
            (Stage.LOOKUP if state.intent and state.intent.kind == "lookup" else Stage.SEARCH)
            .value,
            settings,
        ),
        [Stage.LOOKUP.value, Stage.SEARCH.value, Stage.UNDERSTAND.value, Stage.ERROR.value],
    )
    after_selection = Stage.DOWNLOAD if Stage.DOWNLOAD in enabled else None
    connect(Stage.LOOKUP, after_selection)
    connect(Stage.BROADEN, Stage.SEARCH)

    def after_search(state: SessionState) -> str:
        if state.error:
            return _route(state, END, settings)
        if state.candidates:
            return Stage.RANK.value if Stage.RANK in enabled else END
        return Stage.ERROR.value if state.search_broadened else Stage.BROADEN.value

    graph.add_conditional_edges(
        Stage.SEARCH.value,
        after_search,
        [
            Stage.RANK.value if Stage.RANK in enabled else END,
            Stage.BROADEN.value, Stage.SEARCH.value, Stage.ERROR.value,
        ],
    )
    if Stage.RANK in enabled:
        connect(Stage.RANK, after_selection)
    linear = enabled[3:]
    for index, stage in enumerate(linear):
        connect(stage, linear[index + 1] if index + 1 < len(linear) else None)
    return graph.compile()


def build_digest_graph(services: WorkflowServices, settings: Settings):
    """Build the complete ingestion/briefing graph for synthetic demonstrations."""
    return _build_ingestion_graph(services, settings, Stage.READY)


def build_understanding_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.UNDERSTAND)


def build_discovery_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.SEARCH)


def build_selection_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.RANK)


def build_download_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.DOWNLOAD)


def build_parse_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.PARSE)


def build_index_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.INDEX)


def build_evidence_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.EVIDENCE)


def build_briefing_graph(services: WorkflowServices, settings: Settings):
    return _build_ingestion_graph(services, settings, Stage.BRIEF)


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
            "qa_retrieval_attempts": 0,
            "qa_retrieval_feedback": None,
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

    def retry_retrieval(state: SessionState) -> dict:
        return {
            "qa_retrieval_attempts": state.qa_retrieval_attempts + 1,
            "stage": Stage.REQUERY,
            "stage_history": [*state.stage_history, Stage.REQUERY],
            "updated_at": datetime.now(UTC),
        }

    graph.add_node(Stage.REQUERY.value, retry_retrieval)
    stages = [Stage.RETRIEVE, Stage.ANSWER, Stage.VALIDATE]
    for stage in stages:
        graph.add_node(stage.value, _node(stage, services, settings))
    graph.add_edge(START, Stage.QA_CHECK.value)
    graph.add_conditional_edges(
        Stage.QA_CHECK.value,
        lambda state: Stage.ERROR.value if state.error else Stage.RETRIEVE.value,
        [Stage.ERROR.value, Stage.RETRIEVE.value],
    )

    def after_retrieval(state: SessionState) -> str:
        if state.error:
            return _route(state, Stage.ANSWER.value, settings)
        if state.qa_retrieval_attempts and not state.qa_retrieval_feedback:
            return Stage.VALIDATE.value
        return Stage.ANSWER.value

    def after_answer(state: SessionState) -> str:
        if state.error:
            return _route(state, Stage.VALIDATE.value, settings)
        if (
            state.answer and state.answer.status == "insufficient_evidence"
            and state.qa_retrieval_feedback and not state.qa_retrieval_attempts
        ):
            return Stage.REQUERY.value
        return Stage.VALIDATE.value

    graph.add_conditional_edges(
        Stage.RETRIEVE.value, after_retrieval,
        [Stage.ANSWER.value, Stage.VALIDATE.value, Stage.RETRIEVE.value, Stage.ERROR.value],
    )
    graph.add_conditional_edges(
        Stage.ANSWER.value, after_answer,
        [Stage.REQUERY.value, Stage.VALIDATE.value, Stage.ANSWER.value, Stage.ERROR.value],
    )
    graph.add_edge(Stage.REQUERY.value, Stage.RETRIEVE.value)
    graph.add_conditional_edges(
        Stage.VALIDATE.value,
        lambda state: _route(state, END, settings),
        [END, Stage.VALIDATE.value, Stage.ERROR.value],
    )
    graph.add_edge(Stage.ERROR.value, END)
    return graph.compile()
