# Step 9 verification — page-aware PDF parsing

`parse-paper` runs Steps 5–9 and stops before chunking and embedding. It checks that the
downloaded PDF still matches its recorded SHA-256, extracts selectable text with PyMuPDF,
identifies section boundaries and numbered references, and writes a schema-validated JSON
record to `data/parsed`. Each text block retains its page, section, kind, and bounding box.

## Checks completed

- Live `parse-paper "2106.09685v1"` parsed the cached 20-page LoRA v1 PDF in the order
  `understand → lookup_metadata → download → parse`. All 20 pages contained selectable text.
  The artifact has **494 page-aware blocks**, **34 section spans**, a **PDF-derived abstract**,
  and **40 numbered references**, with no parser warnings.
- The abstract, first introduction paragraph, and first reference were spot-checked against
  the source PDF. The section list includes Introduction, Problem Statement, Our Method,
  Conclusion and Future Work, References, and appendices.
- A second live paper, `1706.03762v1` (*Attention Is All You Need*), parsed all 15 text-bearing
  pages into 511 blocks, 25 section spans, a PDF-derived abstract, and 36 numbered references
  without warnings. Its introduction and first reference were spot-checked as well.
- Offline fixtures cover abstract/section/reference recovery, page provenance, two-column
  order, full-width overlap, metadata abstract fallback, missing references, changed checksums,
  corrupted parsed cache, and clear failure for image-only PDFs.

## Reproduce

```sh
python -m arxiv_agent parse-paper "2106.09685v1"
python -m pytest tests/test_pdf_parse.py
```

The JSON path appears as `parsed_path` in command output. Parsing a cached ID PDF needs no
internet or model inference. Uncached paper lookup/download needs internet; topic selection
also needs local MiniLM and preferably Qwen2.5:3b via Ollama. OCR for scanned PDFs is not in
scope yet; such a paper produces `UNREADABLE_PDF`, so choose a text-based paper. A manual
spot-check is advisable for papers with complex figures, equations, or unusual layouts before
using their parsed text as evidence in later steps.
