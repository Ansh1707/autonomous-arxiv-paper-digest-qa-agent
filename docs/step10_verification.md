# Step 10 verification: chunking, embedding, and vector storage

`index-paper` runs the live graph through `chunk_embed` and stops before evidence extraction,
briefing generation, and QA. It requires the pinned MiniLM snapshot in `.cache/huggingface`;
the command never downloads a model. The paper PDF and metadata can be reused from earlier
steps. Topic input still uses local Qwen2.5:3b for search-term interpretation.

The indexer reads the validated `data/parsed` artifact for the exact selected version and PDF
checksum. It groups blocks by page and section, then applies the MiniLM fast tokenizer to
create 200-token target windows with 30-token overlap. Every final chunk is checked against
the 224-token ceiling, including special tokens, and stores page, section, source block IDs,
text, and stable chunk ID. It does not merge pages, which keeps later citations unambiguous.
The embedding is a normalized 384-dimensional MiniLM vector stored in persistent Chroma with
cosine distance. The index fingerprint binds the paper/version, PDF checksum, parsed blocks,
embedding revision, and chunk settings. A repeat run reuses a complete index.

Verification on 2026-09-25:

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: all tests pass, including chunk limits,
  overlap, provenance, Chroma reopen/query, idempotent rerun, index identity, and bad vectors.
- `2106.09685v1` (LoRA): 117 chunks across the parsed 20-page paper; largest input 204
  MiniLM tokens. `index-paper 2106.09685v1 --probe "What is low rank adaptation?"` completed
  all five live stages and returned source-linked chunks after reopening the stored collection.
- `1706.03762v1` (Attention Is All You Need): 69 chunks across the parsed 15-page paper;
  largest input 200 tokens. A query about scaled dot-product attention returned page 4 in
  that section as its closest match.

For a manual check, run `python -m arxiv_agent index-paper 2106.09685v1 --probe "How does LoRA
adapt a model?"` from the project virtual environment. Check that the returned page and
section agree with the cached PDF and that the quoted text supports the intended question.
If the local model is missing, run `python -m arxiv_agent doctor --download-models` once with
internet access. A scanned PDF still requires a different text extraction path; OCR is outside
this step. Retrieval relevance across many papers and answer grounding remain later work.
