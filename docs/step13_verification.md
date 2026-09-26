# Step 13 verification: bounded full-body evidence extraction

The original plan labels Step 13 “Extract evidence before writing the briefing.” Earlier
milestones implemented a first evidence selector as Step 11 and briefing generation as Step
12. This milestone closes the remaining Step 13 evidence requirements; it does not begin QA.

`EvidenceExtractor` now walks each eligible main-body index chunk in page order. Reference
chunks are excluded. It loads the pinned local Qwen tokenizer and creates batches capped at
2,048 tokens, with at least 512 context tokens reserved for prompt/response overhead. The
artifact records every processed chunk ID once, the Qwen-token count and page range of each
batch, candidate evidence by facet, and the reduced notes used by the briefing. This pass is
extractive and deterministic: it copies source chunks instead of using Qwen to paraphrase
technical results. The subsequent briefing still uses Qwen2.5:3b. Candidate and final notes
retain PDF page, section, source block IDs, and chunk IDs. Numerical wording remains verbatim;
the extractor makes no unsupported mapping between flattened table rows and metrics.

Checks completed on 2026-09-26:

- LoRA `2106.09685v1`: 56 eligible body chunks, pages 1–10, in 8 batches; 20 candidate
  notes reduced to 6 briefing notes. The reduced notes include results on pages 6 and 7 and
  the explicit batching limitation on page 4.
- *Attention Is All You Need* `1706.03762v1`: 53 eligible body chunks, pages 1–10, in 6
  batches; 19 candidate notes reduced to 5. Results on pages 7 and 8 remain represented;
  explicit limitations were not detected.
- A fixture with a result in the last body batch confirms that it remains a cited result
  after reduction, that every body chunk appears exactly once in the batch audit, that
  references are excluded, and that dataset, baseline, metric, and unit wording stays in
  the copied passage.
- The full suite passes with 177 tests, and Ruff lint passes. The saved `e4` artifacts
  validate against the `EvidenceBundle` contract.

Manual review is recommended for the two saved evidence artifacts. In particular, compare
the page 7/8 table-heavy result chunks with the PDF before using a numerical claim. Text
extraction can flatten rows and headings. `not_found` for limitations is a detection result,
not a claim that the paper has no limitation. No extra model download or setup is needed on
the current machine.
