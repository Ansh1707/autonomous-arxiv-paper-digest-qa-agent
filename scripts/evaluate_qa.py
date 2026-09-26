"""Run the fixed paper-level QA set and record retrieval and answer checks separately."""

import argparse
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from arxiv_agent.contracts import SessionState
from arxiv_agent.graph import build_qa_graph
from arxiv_agent.services.qa import QAService, load_qa_session
from arxiv_agent.settings import Settings


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def evaluate_case(case: dict, graph, settings: Settings) -> dict:
    state = load_qa_session(settings, case["paper_id"])
    state.question = case["question"]
    started = perf_counter()
    state = SessionState.model_validate(
        graph.invoke(state, config={"recursion_limit": settings.graph_recursion_limit})
    )
    elapsed = round(perf_counter() - started, 3)
    result = {
        "id": case["id"], "split": case["split"], "paper_id": case["paper_id"],
        "question": case["question"], "expected_status": case["expected_status"],
        "elapsed_seconds": elapsed,
        "error": state.error.model_dump(mode="json") if state.error else None,
    }
    if state.error or state.answer is None:
        result.update(status_pass=False, retrieval_hit=False, fact_pass=False)
        return result
    answer = state.answer
    citations = answer.citations
    expected_page = case.get("expected_page")
    acceptable_sources = (
        [{"page": expected_page, "excerpt": case.get("source_excerpt", "")},
         *case.get("alternate_sources", [])]
        if expected_page else []
    )
    retrieval_hit = (
        any(
            chunk.page_start == source["page"]
            and normalized(source["excerpt"]) in normalized(chunk.text)
            for chunk in state.retrieved_chunks
            for source in acceptable_sources
        )
        if expected_page else None
    )
    status_pass = answer.status == case["expected_status"]
    support_text = " ".join(quote.source_quote for quote in state.answer_support_quotes)
    fact_pass = (
        all(re.search(pattern, answer.text, re.I) for pattern in case["expected_patterns"])
        if case["expected_status"] == "answered" else answer.status == "insufficient_evidence"
    )
    quote_fact_pass = (
        all(
            re.search(pattern, support_text, re.I)
            for pattern in case.get("expected_quote_patterns", case["expected_patterns"])
        )
        if case["expected_status"] == "answered" else not state.answer_support_quotes
    )
    citation_pass = (
        any(
            citation.arxiv_id == state.selected_paper.arxiv_id
            and citation.version == state.selected_paper.version
            and citation.page in {source["page"] for source in acceptable_sources}
            for citation in citations
        )
        if expected_page else not citations
    )
    result.update(
        answer=answer.model_dump(mode="json"),
        support_quotes=[quote.model_dump(mode="json") for quote in state.answer_support_quotes],
        retrieved_chunks=[
            {"chunk_id": chunk.chunk_id, "page": chunk.page_start, "section": chunk.section}
            for chunk in state.retrieved_chunks
        ],
        status_pass=status_pass, retrieval_hit=retrieval_hit,
        fact_pass=bool(fact_pass), quote_fact_pass=bool(quote_fact_pass),
        citation_pass=citation_pass,
        pass_automated=status_pass and bool(fact_pass) and bool(quote_fact_pass)
        and citation_pass,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "held_out", "all"], default="all")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--ids", nargs="+", help="Run only these case IDs")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/step17-live.json"))
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = [
        case for case in data["cases"]
        if (args.split == "all" or case["split"] == args.split)
        and (not args.ids or case["id"] in args.ids)
    ]
    if not cases or (args.ids and {case["id"] for case in cases} != set(args.ids)):
        parser.error("No matching cases, or an unknown case ID was supplied")
    settings = Settings()
    graph = build_qa_graph(QAService(settings), settings)
    report = {
        "schema_version": 1, "started_at": datetime.now(UTC).isoformat(),
        "model": settings.generation_model, "split": args.split, "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = evaluate_case(case, graph, settings)
        report["results"].append(result)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.output)
        print(
            f"{case['id']} status={result.get('answer', {}).get('status', 'error')} "
            f"retrieval={result['retrieval_hit']} pass={result.get('pass_automated', False)} "
            f"time={result['elapsed_seconds']}s",
            flush=True,
        )
    return 0 if all(item.get("pass_automated") for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
