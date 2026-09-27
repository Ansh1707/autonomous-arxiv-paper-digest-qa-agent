"""Terminal interaction stays outside graph nodes."""

import argparse
import json
import logging
import sys

from pydantic import ValidationError

from arxiv_agent.settings import Settings


def _answer_turn(state, question: str) -> dict:
    answer = state.answer
    return {
        "question": question,
        "retrieval_query": state.retrieval_query,
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in state.retrieved_chunks],
        "answer": answer.model_dump(mode="json"),
        "support_quotes": [quote.model_dump(mode="json") for quote in state.answer_support_quotes],
        "source_links": [
            f"{state.selected_paper.pdf_url}#page={citation.page}"
            for citation in answer.citations
        ],
    }


def _qa_turn(state, question: str, graph, settings, repository):
    from arxiv_agent.contracts import SessionState

    state.question = question
    last_stage = None
    for value in graph.stream(
        state, config={"recursion_limit": settings.graph_recursion_limit},
        stream_mode="values",
    ):
        current = SessionState.model_validate(value)
        if current.stage != last_stage:
            print(f"[QA] {current.stage.value}", file=sys.stderr, flush=True)
            last_stage = current.stage
        state = current
    if state.error:
        return state, None
    repository.save(state)
    return state, _answer_turn(state, question)


def _print_sources(state) -> None:
    if not state.conversation:
        print("No answer sources yet.")
        return
    citations = state.conversation[-1].answer.citations
    if not citations:
        print("The last answer abstained; it has no sources.")
        return
    for citation in citations:
        print(
            f"{citation.arxiv_id}v{citation.version} p.{citation.page} "
            f"({citation.section}) — {state.selected_paper.pdf_url}#page={citation.page}"
        )


def _selection_rank(value: str) -> int:
    try:
        rank = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RANK must be an integer from 1 to 10") from exc
    if not 1 <= rank <= 10:
        raise argparse.ArgumentTypeError("RANK must be an integer from 1 to 10")
    return rank


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="arXiv paper briefing and grounded QA agent")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check local prerequisites; no downloads by default")
    doctor.add_argument(
        "--download-models", action="store_true", help="Download MiniLM and tokenizer"
    )
    doctor.add_argument("--smoke", action="store_true", help="Run tiny real model/vector checks")
    demo = sub.add_parser("demo", help="Run the offline SYNTHETIC graph skeleton")
    demo.add_argument("--intent", choices=["lookup", "topic"], default="lookup")
    demo.add_argument(
        "--scenario",
        choices=["success", "empty-search", "retry-search", "parse-failure", "transient-download"],
        default="success",
    )
    sub.add_parser("graph", help="Print Mermaid diagrams for the graph skeleton")
    inspect_input = sub.add_parser(
        "inspect-input", help="Interpret an ID, arXiv URL, or topic without retrieving papers"
    )
    inspect_input.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    discover = sub.add_parser(
        "discover", help="Retrieve paper metadata or a bounded topic candidate pool from arXiv"
    )
    discover.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    select_paper = sub.add_parser(
        "select-paper", help="Rank topic candidates with local MiniLM and choose one paper"
    )
    select_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    select_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    fetch_paper = sub.add_parser(
        "fetch-paper", help="Select a paper and download its validated, versioned PDF"
    )
    fetch_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    fetch_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    parse_paper = sub.add_parser(
        "parse-paper", help="Extract page-aware sections, abstract, and references from a paper"
    )
    parse_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    parse_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    index_paper = sub.add_parser(
        "index-paper", help="Chunk, embed, and persist one parsed paper in Chroma"
    )
    index_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    index_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    index_paper.add_argument(
        "--probe", help="Retrieve matching chunks for this text after indexing"
    )
    evidence_paper = sub.add_parser(
        "evidence-paper", help="Select verbatim, source-linked briefing evidence"
    )
    evidence_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    evidence_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    brief_paper = sub.add_parser(
        "brief-paper", help="Generate a validated Qwen2.5:3b executive briefing"
    )
    brief_paper.add_argument("input", help="An arXiv ID/URL or natural-language topic")
    brief_paper.add_argument(
        "--select", type=_selection_rank, default=1, metavar="RANK",
        help="Choose this one-based rank after semantic ranking (default: 1)",
    )
    ask_paper = sub.add_parser(
        "ask-paper", help="Ask grounded questions using an existing validated briefing"
    )
    ask_paper.add_argument("source", help="Versioned arXiv ID or current briefing JSON path")
    ask_paper.add_argument("question", help="Question about the selected paper")
    ask_paper.add_argument(
        "--follow-up", action="append", default=[], metavar="QUESTION",
        help="Ask another question in the same saved session; repeat as needed",
    )
    digest = sub.add_parser("digest", help="Reserved for later steps; currently unavailable")
    digest.add_argument("input")
    digest.add_argument("--select", type=int, default=1)
    digest.add_argument("--no-chat", action="store_true")
    chat = sub.add_parser("chat", help="Reopen a saved QA session and ask more questions")
    chat.add_argument("--session", required=True)
    chat.add_argument("question", nargs="?", help="Question; omit for an interactive chat")
    chat.add_argument("--follow-up", action="append", default=[], metavar="QUESTION")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = Settings()
    except (ValidationError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if args.command == "digest":
        print(
            f"{args.command} is not implemented yet. "
            "Use 'brief-paper' for a briefing or 'ask-paper' for grounded QA.",
            file=sys.stderr,
        )
        return 2
    if args.command == "doctor":
        from arxiv_agent.diagnostics import run_doctor

        ok = True
        for check in run_doctor(settings, download_models=args.download_models, smoke=args.smoke):
            print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}", flush=True)
            ok = ok and check.ok
        return 0 if ok else 1

    from arxiv_agent.contracts import SessionState
    from arxiv_agent.graph import (
        build_briefing_graph,
        build_digest_graph,
        build_discovery_graph,
        build_download_graph,
        build_evidence_graph,
        build_index_graph,
        build_parse_graph,
        build_qa_graph,
        build_selection_graph,
        build_understanding_graph,
    )
    from arxiv_agent.services.synthetic import SyntheticServices

    if args.command in {"ask-paper", "chat"}:
        from arxiv_agent.services.base import StageFailure
        from arxiv_agent.services.qa import QAService, load_qa_session
        from arxiv_agent.services.sessions import SessionRepository

        try:
            repository = SessionRepository(settings)
            if args.command == "ask-paper":
                state = load_qa_session(settings, args.source)
                repository.save(state)
                print(f"[QA] saved session {state.session_id}", file=sys.stderr)
            else:
                state = repository.load(args.session)
                print(f"[QA] reopened session {state.session_id}", file=sys.stderr)
            graph = build_qa_graph(QAService(settings), settings)
            turns = []
            if args.command == "chat" and args.question is None:
                if args.follow_up:
                    print("Pass a first question before --follow-up.", file=sys.stderr)
                    return 2
                if not sys.stdin.isatty():
                    print(
                        "Interactive chat needs a terminal; pass a question after --session.",
                        file=sys.stderr,
                    )
                    return 2
                print(f"Paper: {state.selected_paper.title}")
                print("Enter a question, /sources for the last answer, or /exit to leave.")
                while True:
                    try:
                        question = input("qa> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if question == "/exit":
                        break
                    if question == "/sources":
                        _print_sources(state)
                        continue
                    if not question:
                        continue
                    state, turn = _qa_turn(state, question, graph, settings, repository)
                    if state.error:
                        print(
                            f"{state.error.code}: {state.error.message} "
                            f"{state.error.recovery}", file=sys.stderr,
                        )
                        return 2
                    print(turn["answer"]["text"])
                    _print_sources(state)
                return 0
            for question in [args.question, *args.follow_up]:
                if question in {"/exit", "/sources"}:
                    print("Use /exit and /sources only in interactive chat.", file=sys.stderr)
                    return 2
                state, turn = _qa_turn(state, question, graph, settings, repository)
                if state.error:
                    print(json.dumps({
                        "session_id": state.session_id,
                        "error": state.error.model_dump(mode="json"),
                        "turns": turns,
                    }, indent=2, ensure_ascii=False))
                    return 2
                turns.append(turn)
            print(json.dumps({
                "paper": {
                    "arxiv_id": (
                        f"{state.selected_paper.arxiv_id}v{state.selected_paper.version}"
                    ),
                    "title": state.selected_paper.title,
                    "pdf_url": str(state.selected_paper.pdf_url),
                },
                "briefing_path": state.output_paths[0],
                "session_id": state.session_id,
                "turns": turns,
            }, indent=2, ensure_ascii=False))
            return 0
        except (StageFailure, ValidationError, ValueError) as exc:
            print(
                f"QA setup error: {exc} {exc.recovery if isinstance(exc, StageFailure) else ''}",
                file=sys.stderr,
            )
            return 2

    if args.command in {
        "inspect-input", "discover", "select-paper", "fetch-paper", "parse-paper",
        "index-paper", "evidence-paper", "brief-paper",
    }:
        from arxiv_agent.services.input_understanding import (
            InputUnderstandingServices,
            OllamaTopicInterpreter,
        )

        try:
            initial = SessionState(user_input=args.input)
        except ValidationError as exc:
            print(f"Invalid input: {exc}", file=sys.stderr)
            return 2
        understanding = InputUnderstandingServices(OllamaTopicInterpreter(settings))
        if args.command == "brief-paper":
            from arxiv_agent.services.briefing import ArxivBriefingServices

            services = ArxivBriefingServices(
                settings, understanding, selection_rank=args.select
            )
            graph = build_briefing_graph(services, settings)
        elif args.command == "evidence-paper":
            from arxiv_agent.services.evidence import ArxivEvidenceServices

            services = ArxivEvidenceServices(
                settings, understanding, selection_rank=args.select
            )
            graph = build_evidence_graph(services, settings)
        elif args.command == "index-paper":
            from arxiv_agent.services.indexing import ArxivIndexServices

            services = ArxivIndexServices(settings, understanding, selection_rank=args.select)
            graph = build_index_graph(services, settings)
        elif args.command == "parse-paper":
            from arxiv_agent.services.pdf_parse import ArxivParseServices

            services = ArxivParseServices(settings, understanding, selection_rank=args.select)
            graph = build_parse_graph(services, settings)
        elif args.command == "fetch-paper":
            from arxiv_agent.services.pdf_download import ArxivPdfServices

            services = ArxivPdfServices(settings, understanding, selection_rank=args.select)
            graph = build_download_graph(services, settings)
        elif args.command == "select-paper":
            from arxiv_agent.services.selection import ArxivSelectionServices

            services = ArxivSelectionServices(settings, understanding, selection_rank=args.select)
            graph = build_selection_graph(services, settings)
        elif args.command == "discover":
            from arxiv_agent.services.discovery import ArxivDiscoveryServices

            services = ArxivDiscoveryServices(settings, understanding)
            graph = build_discovery_graph(services, settings)
        else:
            graph = build_understanding_graph(understanding, settings)
        result = graph.invoke(initial, config={"recursion_limit": settings.graph_recursion_limit})
        state = SessionState.model_validate(result)
        payload = {
            "intent": state.intent.model_dump(mode="json") if state.intent else None,
            "warnings": state.warnings,
            "error": state.error.model_dump(mode="json") if state.error else None,
        }
        if args.command in {
            "discover", "select-paper", "fetch-paper", "parse-paper", "index-paper",
            "evidence-paper",
            "brief-paper",
        }:
            payload.update(
                selected_paper=(
                    state.selected_paper.model_dump(mode="json") if state.selected_paper else None
                ),
                candidates=[item.model_dump(mode="json") for item in state.candidates],
                search_broadened=state.search_broadened,
                stage_history=state.stage_history,
            )
            if args.command in {
                "select-paper", "fetch-paper", "parse-paper", "index-paper",
                "evidence-paper", "brief-paper",
            }:
                payload.update(
                    selection_rank=state.selection_rank,
                    selection_source=state.selection_source,
                )
            if args.command in {
                "fetch-paper", "parse-paper", "index-paper", "evidence-paper", "brief-paper"
            }:
                payload.update(
                    pdf_path=state.pdf_path,
                    pdf_checksum=state.pdf_checksum,
                    pdf_pages=state.pdf_pages,
                    pdf_size_bytes=state.pdf_size_bytes,
                )
            if args.command in {"parse-paper", "index-paper", "evidence-paper", "brief-paper"}:
                payload.update(
                    parsed_path=state.parsed_path,
                    sections=state.sections,
                    abstract_text=state.abstract_text,
                    abstract_source=state.abstract_source,
                    references_found=state.references_found,
                    parsed_reference_count=state.parsed_reference_count,
                    parsed_block_count=state.parsed_block_count,
                    parsed_pages_with_text=state.parsed_pages_with_text,
                )
            if args.command in {"index-paper", "evidence-paper", "brief-paper"}:
                payload["index"] = state.index.model_dump(mode="json") if state.index else None
                if args.command == "index-paper" and state.index and args.probe:
                    from arxiv_agent.services.base import StageFailure

                    try:
                        payload["probe"] = [
                            {
                                "distance": round(distance, 6),
                                "chunk": chunk.model_dump(mode="json"),
                            }
                            for chunk, distance in services.store.query(
                                state.index, args.probe, limit=3
                            )
                        ]
                    except (StageFailure, ValueError) as exc:
                        payload["probe_error"] = str(exc)
            if args.command in {"evidence-paper", "brief-paper"}:
                payload.update(
                    evidence_path=state.evidence_path,
                    limitations_evidence_status=state.limitations_evidence_status,
                    evidence_notes=[note.model_dump(mode="json") for note in state.evidence_notes],
                )
            if args.command == "brief-paper":
                payload.update(
                    briefing=state.briefing.model_dump(mode="json") if state.briefing else None,
                    output_paths=state.output_paths,
                )
        print(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        return 2 if state.status == "failed" else 0
    if args.command == "graph":
        from arxiv_agent.services.briefing import ArxivBriefingServices
        from arxiv_agent.services.discovery import ArxivDiscoveryServices
        from arxiv_agent.services.evidence import ArxivEvidenceServices
        from arxiv_agent.services.indexing import ArxivIndexServices
        from arxiv_agent.services.input_understanding import (
            InputUnderstandingServices,
            OllamaTopicInterpreter,
        )
        from arxiv_agent.services.pdf_download import ArxivPdfServices
        from arxiv_agent.services.pdf_parse import ArxivParseServices
        from arxiv_agent.services.selection import ArxivSelectionServices

        services = SyntheticServices("lookup")
        graphs = [
            (
                "Input understanding (live Step 5)",
                build_understanding_graph(
                    InputUnderstandingServices(OllamaTopicInterpreter(settings)), settings
                ),
            ),
            (
                "Metadata discovery (live Steps 5–6)",
                build_discovery_graph(
                    ArxivDiscoveryServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "Paper selection (live Steps 5–7)",
                build_selection_graph(
                    ArxivSelectionServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "PDF acquisition (live Steps 5–8)",
                build_download_graph(
                    ArxivPdfServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "PDF parsing (live Steps 5–9)",
                build_parse_graph(
                    ArxivParseServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "Vector indexing (live Steps 5–10)",
                build_index_graph(
                    ArxivIndexServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "Evidence extraction (live Steps 5–11)",
                build_evidence_graph(
                    ArxivEvidenceServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            (
                "Executive briefing (live Steps 5–12)",
                build_briefing_graph(
                    ArxivBriefingServices(
                        settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
                    ),
                    settings,
                ),
            ),
            ("Digest skeleton", build_digest_graph(services, settings)),
            ("QA turn graph (live Step 15; synthetic provider in diagram)",
             build_qa_graph(services, settings)),
        ]
        for name, graph in graphs:
            print(f"\n{name}\n```mermaid\n{graph.get_graph().draw_mermaid()}\n```")
        return 0
    try:
        services = SyntheticServices(args.intent, args.scenario)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("SYNTHETIC DEMO — no real paper, download, embedding, briefing, or QA.", flush=True)
    state = SessionState(user_input="synthetic routing fixture", execution_mode="synthetic")
    config = {"recursion_limit": settings.graph_recursion_limit}
    result = build_digest_graph(services, settings).invoke(state, config=config)
    state = SessionState.model_validate(result)
    if state.status == "ready":
        state.question = "What does this synthetic fixture demonstrate?"
        state = SessionState.model_validate(
            build_qa_graph(services, settings).invoke(state, config=config)
        )
    print("Stage order: " + " -> ".join(state.stage_history))
    print(state.model_dump_json(indent=2))
    return 1 if state.status == "failed" else 0
