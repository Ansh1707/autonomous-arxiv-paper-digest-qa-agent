# Step 7 verification — ranking and paper selection

`select-paper` runs real Steps 5–7 and stops before PDF download. Topic candidates are ranked
with the pinned local `all-MiniLM-L6-v2` snapshot on CPU. The score is `0.6 × cosine(topic,
title) + 0.4 × cosine(topic, abstract)`. A stable tie uses the arXiv API order. Direct IDs/URLs
select their one resolved version without loading MiniLM.

## Checks completed

- Offline fixtures verify semantic reordering, score calculation, stable ties, top-rank default,
  explicit rank override, invalid rank rejection, direct lookup bypass, and invalid embedding
  output rejection.
- Live `select-paper "low rank adaptation of language models" --select 2` used local
  `qwen2.5:3b` for topic terms, cached arXiv metadata for ten papers, and local MiniLM for
  ranking. It selected rank 2, **DenseLoRA: Dense Low-Rank Adaptation of Large Language Models**
  (`2505.23808v1`), with `selection_source: "override"` and stages
  `understand → search → rank_select`.

## Reproduce

```sh
python -m arxiv_agent select-paper "low rank adaptation of language models"
python -m arxiv_agent select-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent select-paper "2106.09685v1"
python -m pytest tests/test_selection.py
```

An internet connection is needed for uncached arXiv searches. Ollama with `qwen2.5:3b` gives
the intended topic interpretation; sanitized input terms are a fallback. The local MiniLM
files are required for ranking and can be obtained with `doctor --download-models` during setup.
The user may inspect the candidate titles and choose a different rank; no manual selection is
required because rank 1 is the default. PDF parsing and briefing begin in later steps.
