"""Reproduce the fixed-paper QA challenge with live local models and arXiv PDFs."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_agent.contracts import SessionState
from arxiv_agent.graph import build_briefing_graph, build_qa_graph
from arxiv_agent.services.briefing import ArxivBriefingServices
from arxiv_agent.services.input_understanding import (
    InputUnderstandingServices,
    OllamaTopicInterpreter,
)
from arxiv_agent.services.qa import QAService, load_qa_session
from arxiv_agent.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--reuse-briefings", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    qa_graph = build_qa_graph(QAService(settings), settings)
    results = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.generation_model,
        "case_file": args.cases.name,
        "papers": [],
    }
    for paper in cases:
        paper_id = paper["paper_id"]
        row = {"paper_id": paper_id, "paper": paper["paper"], "questions": []}
        if not args.reuse_briefings:
            services = ArxivBriefingServices(
                settings, InputUnderstandingServices(OllamaTopicInterpreter(settings))
            )
            briefing_state = SessionState.model_validate(
                build_briefing_graph(services, settings).invoke(
                    SessionState(user_input=paper_id),
                    config={"recursion_limit": settings.graph_recursion_limit},
                )
            )
            if briefing_state.error:
                row["briefing_error"] = briefing_state.error.model_dump(mode="json")
                results["papers"].append(row)
                continue
            row["briefing_artifact"] = Path(briefing_state.output_paths[0]).name
            row["pdf_sha256"] = briefing_state.pdf_checksum
            row["index_fingerprint"] = briefing_state.index.fingerprint
        for case in paper["questions"]:
            state = load_qa_session(settings, paper_id)
            row.setdefault("briefing_artifact", Path(state.output_paths[0]).name)
            row.setdefault("index_fingerprint", state.index.fingerprint)
            state.question = case["question"]
            answered = SessionState.model_validate(
                qa_graph.invoke(state, config={"recursion_limit": settings.graph_recursion_limit})
            )
            answer = answered.answer
            quote_text = " ".join(item.source_quote for item in answered.answer_support_quotes)
            row["questions"].append({
                "question": case["question"],
                "expected_status": case["expect"],
                "expected_terms": case["terms"],
                "status": answer.status if answer else "error",
                "answer": answer.text if answer else None,
                "citations": [item.model_dump(mode="json") for item in answer.citations]
                if answer else [],
                "source_quotes": [
                    item.model_dump(mode="json") for item in answered.answer_support_quotes
                ],
                "status_match": bool(answer and answer.status == case["expect"]),
                "terms_present": all(
                    term.casefold() in (answer.text.casefold() if answer else "")
                    for term in case["terms"]
                ),
                "terms_in_support": all(
                    term.casefold() in quote_text.casefold() for term in case["terms"]
                ) if case["expect"] == "answered" else True,
                "error": answered.error.model_dump(mode="json") if answered.error else None,
            })
        results["papers"].append(row)
        print(f"Completed {paper_id}: {len(row['questions'])} questions", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}")
    return 0 if all(
        not paper.get("briefing_error") for paper in results["papers"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
