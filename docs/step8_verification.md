# Step 8 verification — versioned PDF acquisition

`fetch-paper` runs live Steps 5–8 and stops before text extraction. It uses the paper selected
by direct ID lookup or topic ranking, downloads its exact version, and records path, SHA-256,
byte size, and page count in graph state. The ignored `data/pdfs` folder holds the PDF and a
small manifest; a cached file is reused only after its checksum and PDF structure pass again.

## Checks completed

- Live `fetch-paper "2106.09685v1"` downloaded LoRA v1 from arxiv.org to
  `data/pdfs/2106.09685v1.pdf`. It is **1,502,998 bytes**, **20 pages**, and the computed
  SHA-256 matches graph state. The first page renders legibly with the expected LoRA title,
  authors, and `arXiv:2106.09685v1` marking.
- Offline fixtures verify streamed download, exact versioned URLs and separate cache files,
  manifest reuse, corrupted-cache refetch, HTML rejection, size checks against both declared
  and actual bytes, page limit, safe redirects, 404 handling, and at most two retries on 503.
- The graph stage order for direct lookup is `understand → lookup_metadata → download`; no
  parse/index/briefing node runs.

## Reproduce

```sh
python -m arxiv_agent fetch-paper "2106.09685v1"
python -m arxiv_agent fetch-paper "low rank adaptation of language models" --select 2
python -m pytest tests/test_pdf_download.py
```

Uncached downloads need internet access. Topic input also needs the local MiniLM model for
ranking; Qwen2.5:3b provides topic interpretation when Ollama is available. No manual action
is required for direct-ID PDF download after setup. The user may open the cached PDF to verify
it is the intended paper. Extracting sections, abstract, and references is a later step.
