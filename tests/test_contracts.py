import pytest
from pydantic import ValidationError

from arxiv_agent.contracts import Briefing, Chunk, Intent, QAAnswer, SessionState
from arxiv_agent.services.synthetic import synthetic_chunk


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "lookup"},
        {"kind": "topic"},
        {"kind": "lookup", "arxiv_id": "1234.56789", "query": "extra topic"},
    ],
)
def test_intent_requires_exactly_its_own_input(data):
    with pytest.raises(ValidationError):
        Intent(**data)


@pytest.mark.parametrize(
    "patch",
    [
        {"page_start": 0},
        {"page_end": 0},
        {"page_start": 2, "page_end": 1},
        {"embedding_tokens": 225},
        {"text": "   "},
    ],
)
def test_chunk_rejects_bad_provenance_or_size(patch):
    with pytest.raises(ValidationError):
        Chunk.model_validate(synthetic_chunk().model_dump() | patch)


def test_answered_response_requires_citation():
    with pytest.raises(ValidationError):
        QAAnswer(status="answered", text="An unsupported fact")
    assert QAAnswer(status="insufficient_evidence", text="Not enough evidence").citations == []


def test_briefing_schema_requires_assessment_fields():
    required = set(Briefing.model_json_schema()["required"])
    assert {
        "paper",
        "plain_english_summary",
        "problem_statement",
        "method",
        "key_results",
        "limitations_status",
        "limitations",
        "limitations_note",
        "follow_up_questions",
    } <= required


def test_blank_session_input_and_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        SessionState(user_input=" ")
    with pytest.raises(ValidationError):
        SessionState(user_input="test", secrets="not allowed")
