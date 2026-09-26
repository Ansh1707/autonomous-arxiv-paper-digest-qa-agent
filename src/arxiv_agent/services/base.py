from typing import Any, Protocol

from arxiv_agent.contracts import SessionState, Stage


class StageFailure(Exception):
    """Expected operational failure with explicit retry and recovery semantics."""

    def __init__(self, code: str, message: str, recovery: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.recovery = recovery
        self.retryable = retryable


class WorkflowServices(Protocol):
    def run_stage(self, stage: Stage, state: SessionState) -> dict[str, Any]:
        """Return only changed state fields; never mutate input state or prompt the user."""
        ...
