import json
import re
from collections import Counter
from pathlib import Path


def test_fixed_paper_evaluation_has_balanced_independent_cases():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 15
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({(case["paper_id"], case["question"]) for case in cases}) == len(cases)
    assert Counter(case["expected_status"] for case in cases) == {
        "answered": 9, "insufficient_evidence": 6,
    }
    assert Counter(case["paper_id"] for case in cases) == {
        "2106.09685v1": 5, "1706.03762v1": 5, "2305.14314v1": 5,
    }
    assert {case["paper_id"] for case in cases if case["split"] == "development"} == {
        "2106.09685v1"
    }
    for case in cases:
        if case["expected_status"] == "answered":
            assert case["expected_page"] > 0 and case["source_excerpt"]
            assert all(re.compile(pattern) for pattern in case["expected_patterns"])
        else:
            assert case["reason_absent"]
