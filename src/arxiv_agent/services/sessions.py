"""Atomic, validated local snapshots for one-paper QA sessions."""

import os
import tempfile
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from arxiv_agent.contracts import SessionState, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.indexing import ChromaIndexStore
from arxiv_agent.services.qa import load_qa_session
from arxiv_agent.settings import Settings


class SessionRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.directory = settings.data_dir / "sessions"

    def path(self, session_id: str) -> Path:
        try:
            parsed = UUID(session_id)
        except (ValueError, AttributeError) as exc:
            raise StageFailure(
                "SESSION_ID_INVALID", "Expected a canonical UUID session ID.",
                "Copy the session_id printed by ask-paper or chat.",
            ) from exc
        if str(parsed) != session_id:
            raise StageFailure(
                "SESSION_ID_INVALID", "Expected a canonical UUID session ID.",
                "Copy the session_id printed by ask-paper or chat.",
            )
        return self.directory / f"{session_id}.json"

    @staticmethod
    def _validate_ready(state: SessionState) -> None:
        if (
            state.execution_mode != "live"
            or state.status != "ready"
            or state.stage not in {Stage.READY, Stage.VALIDATE}
            or not state.selected_paper
            or not state.briefing
            or not state.index
            or not state.evidence_path
            or len(state.output_paths) != 2
            or state.error
            or (state.conversation and state.answer != state.conversation[-1].answer)
            or (state.conversation and state.question != state.conversation[-1].question)
            or (state.question and not state.conversation)
        ):
            raise StageFailure(
                "SESSION_INVALID", "The session is not a completed, ready QA state.",
                "Start a fresh QA session with ask-paper.",
            )
        for turn in state.conversation:
            if any(
                citation.arxiv_id != state.selected_paper.arxiv_id
                or citation.version != state.selected_paper.version
                for citation in turn.answer.citations
            ):
                raise StageFailure(
                    "SESSION_INVALID", "A saved answer cites a different paper version.",
                    "Start a fresh QA session with ask-paper.",
                )

    def save(self, state: SessionState) -> Path:
        self._validate_ready(state)
        destination = self.path(state.session_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory,
                prefix=f".{state.session_id}-", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                handle.write(state.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            return destination
        except OSError as exc:
            raise StageFailure(
                "SESSION_WRITE_ERROR", "Could not save the QA session.",
                "Check data/sessions permissions and free disk space, then retry.",
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, session_id: str) -> SessionState:
        path = self.path(session_id)
        try:
            state = SessionState.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StageFailure(
                "SESSION_MISSING", "No saved QA session has that ID.",
                "Use the session_id printed by ask-paper or start a new session.",
            ) from exc
        except (OSError, ValueError, ValidationError) as exc:
            raise StageFailure(
                "SESSION_INVALID", "The saved QA session is unreadable or invalid.",
                "Start a fresh QA session with ask-paper.",
            ) from exc
        self._validate_ready(state)
        if state.session_id != session_id:
            raise StageFailure(
                "SESSION_MISMATCH", "Saved session ID does not match its filename.",
                "Start a fresh QA session with ask-paper.",
            )
        # Revalidate the source artifact instead of trusting serialized copies of it.
        fresh = load_qa_session(self.settings, state.output_paths[0])
        if any((
            state.selected_paper != fresh.selected_paper,
            state.index != fresh.index,
            state.briefing != fresh.briefing,
            state.evidence_notes != fresh.evidence_notes,
            state.evidence_path != fresh.evidence_path,
            state.limitations_evidence_status != fresh.limitations_evidence_status,
            state.output_paths != fresh.output_paths,
            state.user_input != fresh.user_input,
        )):
            raise StageFailure(
                "SESSION_MISMATCH", "Saved session does not match its briefing or evidence.",
                "Start a fresh QA session with ask-paper after checking the artifacts.",
            )
        ChromaIndexStore(self.settings).validate_index(state.index)
        return state
