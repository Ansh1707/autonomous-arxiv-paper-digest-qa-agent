# Step 14 verification: executive briefing

The original plan labels Step 14 “Generate and validate the executive briefing.” A first
briefing implementation was delivered earlier as Step 12. This milestone completes the
original Step 14 details using the full-body `e4` evidence from Step 13.

The live `brief-paper` graph generates a `b3` JSON audit and Markdown briefing. Title,
authors, versioned arXiv link, and publication date come from validated arXiv metadata. The
Qwen2.5:3b generator drafts concise problem and method content, while source-linked
comparisons can be copied from the paper to preserve precise qualifiers. Every scientific
claim has a cited chunk and a copied source quote. The builder rejects unknown/wrong-facet
citations, quotes absent from the source, unsupported numeric values, invented limitations,
and omission of a detected limitation. It retries an invalid draft once and saves no briefing
if validation still fails. The output has exactly three distinct follow-up questions.

Paper limitations and processing caveats occupy separate fields and Markdown sections. When
the evidence extractor reports no explicit limitation, the briefing says none was *detected*
in the processed main-body passages; it does not assert that the paper has none. Table-heavy
passages trigger a separate warning about flattened PDF text.

Verification on 2026-09-26:

- The public `brief-paper 2106.09685v1` command completed every stage through `summarize`.
  Its briefing has seven cited claims, three questions, an explicit LoRA batching limitation,
  and a separate table-extraction caveat.
- The public `brief-paper 1706.03762v1` command completed successfully. Its briefing has
  six cited claims, three questions, a clear `not_found` limitation status, and a separate
  processing caveat. The English-to-French comparison retains the paper's “single models”
  qualifier verbatim.
- Both `b3` JSON records were reopened and checked against their `e4` evidence: metadata
  matches the paper, every claim citation resolves, every quote is copied from its chunk,
  and every number in a claim appears in the cited quote. Markdown contains the required
  fields and page links.
- The full suite passes with 179 tests. Ruff lint and the package install pass.

Read the final [LoRA Markdown](../outputs/2106.09685v1-8a55beb68c48-b3.md) and
[Transformer Markdown](../outputs/1706.03762v1-081e7b04db5d-b3.md), with their adjacent
JSON audit files. Structural citation and number checks establish traceability but cannot
prove full semantic entailment. Before sharing a scientific briefing, a human should compare
the paraphrased claims and any numerical comparison with the linked PDF page. No other manual
setup is needed on this machine. Interactive follow-up QA remains Step 15.
