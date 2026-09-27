import pytest

from arxiv_agent.cli import main, parser
from arxiv_agent.diagnostics import Check


@pytest.mark.parametrize("intent", ["lookup", "topic"])
def test_demo_is_explicitly_synthetic(capsys, intent):
    assert main(["demo", "--intent", intent]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC DEMO" in output
    assert '"execution_mode": "synthetic"' in output
    assert '"status": "ready"' in output


def test_demo_failure_returns_nonzero(capsys):
    assert main(["demo", "--scenario", "parse-failure"]) == 1
    assert "UNREADABLE_PDF" in capsys.readouterr().out


def test_cli_lists_only_available_paper_commands():
    help_text = parser().format_help()
    assert "brief-paper" in help_text
    assert "ask-paper" in help_text
    assert "digest" not in help_text


def test_chat_rejects_invalid_session_id(capsys):
    assert main(["chat", "--session", "example", "What happened?"]) == 2
    assert "canonical UUID" in capsys.readouterr().err


def test_interactive_chat_sources_and_exit(monkeypatch, capsys, settings):
    from test_qa import state as qa_state

    from arxiv_agent.contracts import Citation, ConversationTurn, QAAnswer, Stage
    from arxiv_agent.services.sessions import SessionRepository

    current = qa_state()
    current.stage = Stage.VALIDATE
    current.question = "What happened?"
    current.answer = QAAnswer(
        status="answered", text="The method froze its base weights.",
        citations=[Citation(
            chunk_id="chunk-1", arxiv_id=current.selected_paper.arxiv_id,
            version=current.selected_paper.version, page=2, section="Method",
        )],
    )
    current.conversation = [ConversationTurn(question=current.question, answer=current.answer)]
    monkeypatch.setattr(SessionRepository, "load", lambda self, session_id: current)
    monkeypatch.setattr("arxiv_agent.cli.sys.stdin.isatty", lambda: True)
    commands = iter(["/sources", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    assert main(["chat", "--session", current.session_id]) == 0
    output = capsys.readouterr().out
    assert "#page=2" in output
    assert "Paper:" in output


def test_qa_command_requires_a_saved_briefing(capsys):
    assert main(["ask-paper", "2106.09685v1", "What does LoRA freeze?"]) == 2
    assert "QA setup error" in capsys.readouterr().err


def test_qa_parser_accepts_in_memory_followups():
    args = parser().parse_args([
        "ask-paper", "2106.09685v1", "What is the result?",
        "--follow-up", "How was it measured?",
    ])
    assert args.follow_up == ["How was it measured?"]


def test_invalid_demo_combination_is_clear(capsys):
    assert main(["demo", "--intent", "lookup", "--scenario", "empty-search"]) == 2
    assert "require topic" in capsys.readouterr().err


def test_invalid_config_returns_actionable_error(monkeypatch, capsys):
    monkeypatch.setenv("ARXIV_AGENT_CONTEXT_TOKENS", "-1")
    assert main(["demo"]) == 2
    assert "Configuration error" in capsys.readouterr().err


@pytest.mark.parametrize("passed,code", [(True, 0), (False, 1)])
def test_doctor_exit_status_and_no_download_default(monkeypatch, capsys, passed, code):
    def fake_doctor(settings, *, download_models, smoke):
        assert download_models is False
        assert smoke is False
        yield Check("fixture", passed, "diagnostic result")

    monkeypatch.setattr("arxiv_agent.diagnostics.run_doctor", fake_doctor)
    assert main(["doctor"]) == code
    assert "diagnostic result" in capsys.readouterr().out


def test_graph_prints_actual_compiled_graphs(capsys):
    assert main(["graph"]) == 0
    output = capsys.readouterr().out
    assert "lookup_metadata" in output
    assert "handle_error" in output
    assert "check_qa_ready" in output
    assert "Paper selection (live Steps 5–7)" in output
    assert "PDF parsing (live Steps 5–9)" in output


@pytest.mark.parametrize("rank", ["0", "-1", "abc"])
@pytest.mark.parametrize("command", ["select-paper", "fetch-paper", "parse-paper"])
def test_selection_rejects_impossible_rank_before_network(rank, command):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args([command, "graph networks", "--select", rank])
    assert exc.value.code == 2


def test_selection_accepts_rank_beyond_old_ten_candidate_limit():
    args = parser().parse_args(["select-paper", "quantized language models", "--select", "45"])
    assert args.select == 45
