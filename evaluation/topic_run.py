"""Evaluate arXiv candidate recall and MiniLM topic ranking on fixed searches."""

import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_agent.contracts import Intent, SessionState
from arxiv_agent.services.discovery import ArxivDiscoveryServices
from arxiv_agent.services.selection import EmbeddingRanker, LocalMiniLMEncoder
from arxiv_agent.settings import Settings


def main() -> int:
    folder = Path(__file__).parent
    cases = json.loads((folder / "topic_cases.json").read_text(encoding="utf-8"))
    settings = Settings()
    discovery = ArxivDiscoveryServices(settings, understanding=None)
    ranker = EmbeddingRanker(LocalMiniLMEncoder(settings))
    rows = []
    for case in cases:
        intent = Intent(
            kind="topic", query=case["topic"], search_terms=case["terms"],
            arxiv_query=case["arxiv_query"],
        )
        found = discovery._search(SessionState(user_input=case["topic"], intent=intent))
        candidates = found["candidates"]
        ranked = ranker.rank(intent, candidates) if candidates else []
        expected = case["expected_id"]
        rank = next(
            (number for number, item in enumerate(ranked, 1) if item.paper.arxiv_id == expected),
            None,
        )
        rows.append({
            "topic": case["topic"], "expected_id": expected,
            "candidate_count": len(candidates), "expected_rank": rank,
            "top_five": [
                {"id": item.paper.arxiv_id, "title": item.paper.title, "score": item.score}
                for item in ranked[:5]
            ],
        })
        print(f"{case['topic']}: rank {rank} among {len(candidates)}", flush=True)
    result = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "pool_limits": {
            "relevance": settings.candidate_count,
            "oldest": settings.candidate_oldest_count,
            "phrase": settings.candidate_phrase_count,
        },
        "queries": rows,
    }
    (folder / "topic_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
