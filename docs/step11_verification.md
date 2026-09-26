# Step 11 verification: source-linked briefing evidence

`evidence-paper` runs the live graph through `extract_evidence` and stops before briefing
generation or QA. It reads the selected paper's persistent Chroma collection, retrieves
candidates for problem, method, and results, and prefers the main paper's matching sections
over appendix or reference matches. It saves an `EvidenceBundle` JSON artifact in
`data/evidence`, with verbatim chunk text, chunk IDs, PDF pages, section labels, and original
PDF block IDs. No model-written scientific claim is made in this stage.

Limitations are handled separately. The extractor only marks them `reported` when it finds an
explicit limitation passage outside the references and generic problem/background sections.
Otherwise it records `not_found` and a warning. That status means this heuristic found no
clear passage; a human may still notice a limitation elsewhere in the paper.

Verification on 2026-09-25:

- Offline tests cover facet selection, exact source text/provenance, exclusions for references,
  missing limitations, mismatched indexes, and the graph stopping before `summarize`.
- On LoRA `2106.09685v1`, six notes were selected: the page 2 problem statement, two page 3
  method passages, GPT-3 and GPT-2 results on pages 6 and 7, and the stated limitation on
  page 4 about batching different tasks. Status: `reported`.
- On *Attention Is All You Need* `1706.03762v1`, five notes were selected: the page 1
  introduction, architecture/application passages on pages 2 and 4, and machine translation
  results on pages 7 and 8. No explicit limitation matched; status: `not_found`.
- `python -m pytest -q`, `python -m ruff check .`, and `python -m pip check` pass in the
  project's Python 3.11 environment.

Manual review: run `python -m arxiv_agent evidence-paper 2106.09685v1` and inspect the saved
JSON beside the cached PDF. Check that each selected passage actually supports its assigned
facet, particularly numerical results and limitations. The source text is unedited, so PDF
extraction artifacts may remain. A future briefing stage must paraphrase carefully and
validate every generated claim against this evidence.
