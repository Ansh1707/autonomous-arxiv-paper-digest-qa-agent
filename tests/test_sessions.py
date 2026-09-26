import json
from uuid import uuid4

import pytest
from test_qa import state as qa_state

from arxiv_agent.contracts import ConversationTurn, QAAnswer, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.sessions import SessionRepository


def ready_state(settings):
    current = qa_state()
    current.stage = Stage.READY
    current.question = None
    current.output_paths = [
        str(settings.output_dir / "example.json"),
        str(settings.output_dir / "example.md"),
    ]
    current.evidence_path = str(settings.data_dir / "evidence" / "example.json")
    return current


def patched_sources(monkeypatch, current):
    monkeypatch.setattr(
        "arxiv_agent.services.sessions.load_qa_session",
        lambda settings, source: current,
    )
    monkeypatch.setattr(
        "arxiv_agent.services.sessions.ChromaIndexStore.validate_index",
        lambda self, index: None,
    )


def test_session_roundtrip_preserves_conversation_and_identity(settings, monkeypatch):
    current = ready_state(settings)
    current.stage = Stage.VALIDATE
    current.question = "What happened?"
    current.answer = QAAnswer(
        status="insufficient_evidence", text="No evidence was found."
    )
    current.conversation = [
        ConversationTurn(question=current.question, answer=current.answer)
    ]
    patched_sources(monkeypatch, ready_state(settings))
    first = SessionRepository(settings)
    path = first.save(current)
    assert path.stat().st_mode & 0o777 == 0o600
    restored = SessionRepository(settings).load(current.session_id)
    assert restored.session_id == current.session_id
    assert restored.conversation == current.conversation
    assert restored.selected_paper == current.selected_paper
    assert restored.index == current.index


def test_session_rejects_invalid_ids_and_corrupt_json(settings):
    repository = SessionRepository(settings)
    with pytest.raises(StageFailure, match="canonical UUID"):
        repository.load("../other")
    with pytest.raises(StageFailure, match="No saved QA session"):
        repository.load(str(uuid4()))
    current = ready_state(settings)
    path = repository.save(current)
    path.write_text("not json")
    with pytest.raises(StageFailure, match="unreadable or invalid"):
        repository.load(current.session_id)


def test_session_rejects_mismatched_briefing_and_index(settings, monkeypatch):
    current = ready_state(settings)
    repository = SessionRepository(settings)
    repository.save(current)
    other = ready_state(settings)
    other.index = other.index.model_copy(update={"fingerprint": "a" * 64})
    patched_sources(monkeypatch, other)
    with pytest.raises(StageFailure, match="does not match its briefing"):
        repository.load(current.session_id)
    patched_sources(monkeypatch, current)

    def missing_index(self, index):
        raise StageFailure("INDEX_MISSING", "The index is missing.", "Rebuild it.")

    monkeypatch.setattr(
        "arxiv_agent.services.sessions.ChromaIndexStore.validate_index", missing_index
    )
    with pytest.raises(StageFailure, match="index is missing"):
        repository.load(current.session_id)


def test_session_rejects_other_paper_citation(settings, monkeypatch):
    current = ready_state(settings)
    repository = SessionRepository(settings)
    path = repository.save(current)
    payload = json.loads(path.read_text())
    payload["question"] = "What happened?"
    payload["answer"] = {
        "status": "answered",
        "text": "Something happened.",
        "citations": [{
            "chunk_id": "x", "arxiv_id": "9999.99999", "version": 1,
            "page": 1, "section": "Method",
        }],
    }
    payload["conversation"] = [{"question": payload["question"], "answer": payload["answer"]}]
    path.write_text(json.dumps(payload))
    patched_sources(monkeypatch, current)
    with pytest.raises(StageFailure, match="different paper"):
        repository.load(current.session_id)


def test_failed_write_keeps_previous_snapshot(settings, monkeypatch):
    current = ready_state(settings)
    repository = SessionRepository(settings)
    path = repository.save(current)
    original = path.read_bytes()
    current.warnings.append("new warning")

    def fail_replace(source, destination):
        raise OSError("simulated failure")

    monkeypatch.setattr("arxiv_agent.services.sessions.os.replace", fail_replace)
    with pytest.raises(StageFailure, match="Could not save"):
        repository.save(current)
    assert path.read_bytes() == original
    assert list(repository.directory.glob("*.tmp")) == []
