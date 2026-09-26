# Autonomous arXiv Paper Digest and QA Agent

Local Python application built with LangGraph, Ollama Qwen2.5:3b, MiniLM embeddings,
and Chroma. The assessment pipeline is implemented: requirements, environment checks, project
structure, validated records, graph skeleton, input understanding, arXiv metadata discovery,
local embedding-based paper selection, validated PDF acquisition, page-aware text extraction,
and persistent vector indexing, source-linked evidence extraction, executive briefing,
grounded follow-up QA, durable session reopening, and a three-paper evaluation.
Step 13 strengthens the evidence extractor originally added in Step 11.

The `demo` command uses explicitly synthetic fixtures to verify graph routing. It never
claims to have downloaded, read, indexed, or summarized a real paper.

## Setup

Target: Apple M1 MacBook Air, 8 GB RAM, Python **3.11**. Internet is required to install
dependencies and download models initially. Python's packages and models stay local;
no paid API key or LangSmith account is needed.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation .
```

`requirements.lock` pins the tested Python 3.11 dependency environment, including developer
tools. It is not a promise of validation on other operating systems. The project does not
modify the system Python. Reinstall the package with the final command after changing source
files; this Python installation does not load editable `.pth` installs. Optional overrides are
documented in `.env.example`.

Install/start [Ollama](https://ollama.com/download), then obtain the exact model if missing:

```sh
ollama pull qwen2.5:3b
```

If the Ollama desktop app is not already serving requests, run `ollama serve` in a separate
terminal. Do not start a second server when the app is already running.

## Foundation commands

```sh
python -m arxiv_agent --help
python -m arxiv_agent doctor
python -m arxiv_agent doctor --download-models
python -m arxiv_agent doctor --smoke
python -m arxiv_agent inspect-input "2106.09685v2"
python -m arxiv_agent inspect-input "https://arxiv.org/pdf/2106.09685.pdf"
python -m arxiv_agent inspect-input "recent work on KV-cache compression for LLMs"
python -m arxiv_agent inspect-input "transformer inference from 2024-01-01 to 2024-12-31"
python -m arxiv_agent discover "2106.09685v1"
python -m arxiv_agent discover "low rank adaptation of language models"
python -m arxiv_agent select-paper "low rank adaptation of language models"
python -m arxiv_agent select-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent fetch-paper "2106.09685v1"
python -m arxiv_agent fetch-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent parse-paper "2106.09685v1"
python -m arxiv_agent parse-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent index-paper "2106.09685v1" --probe "How does LoRA adapt a model?"
python -m arxiv_agent index-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent evidence-paper "2106.09685v1"
python -m arxiv_agent evidence-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent brief-paper "2106.09685v1"
python -m arxiv_agent brief-paper "low rank adaptation of language models" --select 2
python -m arxiv_agent ask-paper 2106.09685v1 "What does LoRA freeze during training?"
python -m arxiv_agent ask-paper 2106.09685v1 "What limitation does LoRA report about batching tasks?" --follow-up "Why is that difficult?"
python -m arxiv_agent chat --session SESSION_ID "What else does the paper report?"
python -m arxiv_agent chat --session SESSION_ID
python scripts/evaluate_qa.py --output evaluation/results/local-evaluation.json
python -m arxiv_agent demo --intent lookup
python -m arxiv_agent demo --intent topic
python -m arxiv_agent demo --intent topic --scenario parse-failure
python -m arxiv_agent graph
python -m pytest
python -m ruff check .
```

`doctor` checks Python, required imports, Ollama connectivity/model availability, and cached
MiniLM/Qwen tokenizer files. It does not download assets by default. `--download-models`
explicitly downloads MiniLM and tokenizer files only (not another set of Qwen weights).
`--smoke` performs a tiny real generation, embedding, and temporary Chroma reopen/query check.
Checks produce actionable errors and a nonzero exit code on failure. Model inference may take
time on an 8 GB machine, particularly on the first load.

The graph demo is offline and deterministic. It prints `SYNTHETIC DEMO`, stage order, and
validated final state. Supported scenarios: success, empty-search, retry-search,
parse-failure, and transient-download. Failure scenarios deliberately exit with code 1.
The `inspect-input` command runs only the input-understanding node. It recognizes modern and
legacy IDs, direct arxiv.org abstract/PDF URLs, and research topics. For topics, Qwen proposes
up to three search phrases in a validated JSON response. Python constructs an escaped arXiv
search expression; the model cannot supply raw query syntax. If Qwen is unreachable or its
output is invalid, sanitized terms from the input are used and the command prints a warning.
It performs no arXiv lookup. The `discover` command runs input understanding and retrieves
metadata from the official arXiv API. For an ID, it prints the resolved paper and version;
for a topic, it prints at most ten metadata candidates. A supplied `vN` is verified exactly.
Unversioned IDs resolve to the version returned by arXiv. It stores successful metadata
responses for 24 hours in `data/metadata`, enforces a minimum three seconds between API
attempts, uses a 20-second request timeout, and retries transient failures at most twice.
An empty search broadens terms once while retaining any date filter. API errors, missing IDs,
and unavailable versions produce structured errors and a nonzero exit. `digest` remains
unavailable; use `brief-paper` followed by `ask-paper` or `chat`.

`select-paper` continues through the live `rank_select` graph node. For topic searches,
MiniLM embeds the interpreted topic terms, each candidate title, and each abstract on CPU.
The score is 60% title cosine similarity plus 40% abstract cosine similarity. Candidates are
shown in descending score order; equal scores retain arXiv API order. The default selects rank
1, while `--select N` chooses a different one-based rank from the displayed list. Direct ID/URL
lookups select their one resolved paper without ranking. The command records the chosen paper,
its rank, and whether it was an override, then stops before PDF download. The local MiniLM
snapshot must already be present; run `doctor --download-models` once if it is missing.

`fetch-paper` continues through the live `download` graph node. It fetches the selected
version's arXiv PDF to `data/pdfs`, checks the PDF header and structure, rejects encrypted,
oversized, or overlong files, records SHA-256, page count, and byte size, and reuses a cached
copy only when its manifest and checksum still match. Downloads are streamed to a temporary
file and moved into place only after validation. The default limits are 50 MB, 60 pages, and
60 seconds; use `ARXIV_AGENT_MAX_PDF_MB`, `ARXIV_AGENT_MAX_PDF_PAGES`, and
`ARXIV_AGENT_PDF_TIMEOUT_SECONDS` to adjust them. The command stops before extracting text.

`parse-paper` continues through the live `parse` node and writes a validated JSON artifact in
`data/parsed`. It keeps page numbers, bounding boxes, reading order, and section labels for
each text block, plus the recovered abstract and numbered references. It handles common
numbered headings, unnumbered Abstract/References headings, and two-column pages. A missing
PDF abstract is explicitly sourced from arXiv metadata with a warning; missing numbered
references are reported without inventing them. Image-only or text-poor PDFs fail with an
`UNREADABLE_PDF` error because OCR is not implemented. The command stops before chunking or
embedding.

`index-paper` continues through the live `chunk_embed` node. It groups parsed blocks by PDF
page and section, makes overlapping MiniLM-tokenized windows (200-token target, 224-token
ceiling, 30-token overlap by default), and records source block IDs for every chunk. It embeds
chunks with the pinned local MiniLM model on CPU and persists normalized 384-dimensional
vectors, text, and provenance in `data/vectors` using Chroma cosine distance. A fingerprint
includes paper version, PDF checksum, parsed block content, model revision, and chunk settings;
rerunning the same input reuses the complete collection. `--probe` prints the three closest
chunks, their distances, pages, sections, and source blocks for manual inspection. This is
retrieval only; it does not generate a briefing or answer.

`evidence-paper` continues through `extract_evidence`. It scans every eligible main-body
chunk in page order, excluding references, and records bounded batches measured with the
pinned Qwen tokenizer. Vector retrieval and section/signal matching identify candidate
passages for the problem, method, and results within those batches. A reduction step selects
briefing notes while retaining candidate notes, batch coverage, verbatim text, chunk IDs, PDF
pages, sections, and source block IDs in the JSON artifact. Verbatim passages retain any
dataset, metric, baseline, and unit wording present in the PDF chunk; no numeric relationship
is inferred from a flattened table. It checks for an explicit passage about the paper's own
limitations. `limitations_status: not_found` means the heuristic found no clear passage, not
that the paper has none. This stage does not paraphrase claims or generate the briefing.

`brief-paper` continues through `summarize` and writes a JSON audit record plus a readable
Markdown executive briefing in `outputs`. Bibliographic fields come directly from arXiv
metadata. Qwen2.5:3b drafts concise source-bound text field by field. Each claim is tied to
a Step 13 chunk, and the program checks that its source quote is present and that any numeric
value occurs in the cited passage. Dense PDF tables are handled conservatively: the briefing
uses explicit comparison prose from nearby passages or captions instead of guessing which row
belongs to a metric. The output includes title, authors, ID, date, link, why-it-matters
summary, problem, method bullets, key results, explicit limitations status, and follow-up
questions (exactly three). Result sentences stated clearly in the source can be copied directly
to keep qualifiers such as “single models.” Processing caveats are listed separately from
limitations reported by the paper. Citations link to the exact PDF page. This stage ends before
interactive QA.

`ask-paper` opens the current validated `b3` briefing and matching `e4` evidence for an
exact versioned arXiv ID, or accepts the briefing JSON path. It does not regenerate the
briefing. Each question retrieves 12 candidates from the selected paper, combines semantic
and exact-term matches, excludes references by default, removes near-duplicates, and fits
up to six passages within the Qwen context budget. Bibliography questions can include
references. A simple metric question naming a task uses its direct prose passage when
available, avoiding neighboring task columns in flattened tables. Qwen2.5:3b supplies a
concise answer and source IDs; the validator checks the paper/version, cited passages,
copied support text, numbers, and key actions. One invalid answer repair is allowed.
Insufficient evidence or persistent invalid output returns the exact abstention without
citations. Repeat `--follow-up` for more turns in one command. Only earlier *user questions*
help resolve references such as “that method”; every turn retrieves fresh paper evidence.
The JSON response includes PDF page links, support quotes, and a `session_id`. A successful
`ask-paper` or `chat` turn atomically saves the complete state to `data/sessions/SESSION_ID.json`.
Run `chat --session SESSION_ID "question"` from another process to continue, or omit the
question in a terminal for an interactive loop. Interactive `/sources` prints the latest
answer's PDF page links; `/exit` closes the loop. Earlier questions remain available for
referential follow-ups, but every turn retrieves new evidence. The CLI prints graph-stage
progress to stderr and keeps JSON answers on stdout for scripted commands.

On reopen, the CLI checks the saved state against the versioned briefing and evidence, then
checks the Chroma collection fingerprint, embedding model revision, and chunk count. It does
not download a PDF or re-embed the paper. A missing session, artifact, index, tokenizer, or
Ollama service produces recovery guidance. A failed answer does not replace the last saved
ready state; rerun `chat --session SESSION_ID` after correcting the cause.

## Example: one paper, briefing, and follow-up QA

Run this from a configured environment with Ollama serving `qwen2.5:3b`:

```sh
python -m arxiv_agent brief-paper 2106.09685v1
python -m arxiv_agent ask-paper 2106.09685v1 "What does LoRA freeze during training?"
python -m arxiv_agent ask-paper 2106.09685v1 "For GPT-3, by how much can LoRA reduce trainable parameters compared with full fine-tuning?"
python -m arxiv_agent ask-paper 2106.09685v1 "What batching limitation does LoRA explicitly report?"
```

The recorded briefing is titled **LoRA: Low-Rank Adaptation of Large Language Models**
([2106.09685v1](https://arxiv.org/abs/2106.09685v1)). Its method section says the pretrained
weights are frozen while low-rank matrices are trained; its limitations section notes that
batching different tasks with different A/B matrices in one forward pass is difficult.
The generated Markdown includes the authors, publication date, paper link, summary,
problem, method bullets, results, limitations, three follow-up questions, and PDF-page
citations. A complete, reviewed example is in [examples/lora_briefing.md](examples/lora_briefing.md).

Three recorded QA exchanges from the fixed evaluation are below. Answers can vary between
model runs; each citation points to the exact paper version and PDF page.

| Question | Recorded answer | Source |
| --- | --- | --- |
| What does LoRA freeze during training? | LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, reducing the number of trainable parameters for downstream tasks. | [LoRA PDF p. 1](https://arxiv.org/pdf/2106.09685v1#page=1) |
| For GPT-3, by how much can LoRA reduce trainable parameters compared with full fine-tuning? | LoRA can reduce the number of trainable parameters by 10,000 times compared to full fine-tuning for GPT-3. | [LoRA PDF p. 1](https://arxiv.org/pdf/2106.09685v1#page=1) |
| What batching limitation does LoRA explicitly report? | LoRA explicitly reports a limitation where it is not straightforward to batch inputs to different tasks with different A and B in a single forward pass. | [LoRA PDF p. 4](https://arxiv.org/pdf/2106.09685v1#page=4) |

For an unsupported question such as “What is the first author's favorite food?”, the
recorded answer is “I couldn’t find enough evidence in the retrieved paper text to answer
that.” and has no citation. The [source-verified evaluation](evaluation/results/step17-final-source-verified.json)
records answers, copied quotes, retrieval pages, and failures as well as successes.

Dates in the user text control filters. `recent`, `latest`, and `newest` mean the prior
12 calendar months, ending today in UTC. Supported explicit forms include `in 2024`,
`from 2024 to 2025`, `from YYYY-MM-DD to YYYY-MM-DD`, `since YYYY-MM-DD`, `before` or
`after YYYY-MM-DD`, `last N months/years`, and `this year`. A valid explicit range overrides
`recent`. Unrecognized date requests fail with a clear correction instead of silently
dropping the date constraint. The generated filter uses arXiv's UTC `submittedDate` field;
it does not use date words in the embedding ranking query. See the
[official arXiv API manual](https://info.arxiv.org/help/api/user-manual.html).

## Architecture and requirements

See [acceptance checklist](docs/acceptance.md), [architecture](docs/architecture.md), and
[foundation verification](docs/verification.md). Step 5 verification is recorded in
[docs/step5_verification.md](docs/step5_verification.md), Step 6 in
[docs/step6_verification.md](docs/step6_verification.md), and Step 7 in
[docs/step7_verification.md](docs/step7_verification.md), and Step 8 in
[docs/step8_verification.md](docs/step8_verification.md), and Step 9 in
[docs/step9_verification.md](docs/step9_verification.md), and Step 10 in
[docs/step10_verification.md](docs/step10_verification.md), and Step 11 in
[docs/step11_verification.md](docs/step11_verification.md), and Step 12 in
[docs/step12_verification.md](docs/step12_verification.md), and Step 13 in
[docs/step13_verification.md](docs/step13_verification.md), and Step 14 in
[docs/step14_verification.md](docs/step14_verification.md).
[Step 15 verification](docs/step15_verification.md) records the live QA checks.
[Step 16 verification](docs/step16_verification.md) records session reopening checks.
[Step 17 verification](docs/step17_verification.md) records the three-paper evaluation.
[Step 18 submission checks](docs/step18_submission.md) record packaging and manual handoff;
the [reflection guide](docs/step18_reflection_guide.md) helps keep the required video under
four minutes.

- `src/arxiv_agent/`: CLI, settings, contracts, graph, and diagnostics.
- `src/arxiv_agent/services/`: injectable workflow service boundary and synthetic implementation.
- `src/arxiv_agent/services/input_understanding.py`: live Step 5 input interpreter and query builder.
- `src/arxiv_agent/services/discovery.py`: live Step 6 arXiv metadata service and cache.
- `src/arxiv_agent/services/selection.py`: live Step 7 MiniLM ranking and selection.
- `src/arxiv_agent/services/pdf_download.py`: live Step 8 bounded PDF download and validation.
- `src/arxiv_agent/services/pdf_parse.py`: live Step 9 page-aware text extraction.
- `src/arxiv_agent/services/indexing.py`: live Step 10 token-aware chunking and Chroma storage.
- `src/arxiv_agent/services/evidence.py`: live Step 11 source-linked evidence selection.
- `src/arxiv_agent/services/briefing.py`: live Step 12 local Qwen briefing and validation.
- `src/arxiv_agent/services/qa.py`: selected-paper QA and citation validation.
- `src/arxiv_agent/services/sessions.py`: atomic session snapshots and reopen validation.
- `src/arxiv_agent/prompts/`: reserved for later QA prompts.
- `tests/`: offline contract, configuration, CLI, and graph behavior tests.
- Ignored runtime folders: `data/metadata`, `data/pdfs`, `data/parsed`, `data/vectors`,
  `data/evidence`, `data/sessions`,
  `outputs`, and `.cache/huggingface`.

LangGraph holds typed state in memory across nodes. Pydantic validates service updates and
serializable records. Clients are injected into nodes rather than stored in state. The CLI
owns user interaction. The briefing graph stops before QA readiness. Durable
sessions, real QA, and the scored evaluation are implemented.

## Design Decisions & Tradeoffs

One selected paper per session keeps the assessment focused on grounding. Topic searches rank
up to ten candidates. A CLI is sufficient; UI polish is explicitly outside the assessment.
Qwen2.5 3B generates text; MiniLM serves the separate embedding role. Default generation context
is 4,096 tokens and concurrency is one to control memory use. MiniLM chunk inputs are capped at
224 wordpieces, including special tokens, below its default 256-token limit.

The workflow uses explicit graph stages and a typed shared state so paper selection, PDF
processing, briefing, and QA can fail or resume at clear boundaries. Briefing and evidence
artifacts are versioned with paper and model inputs, and QA sessions save the selected paper,
index fingerprint, and conversation turns. Reopening a session validates those artifacts
instead of silently continuing against changed evidence. Downloads are bounded and checked
before parsing; scanned or nearly textless PDFs stop with a recovery message rather than
producing a misleading briefing.

Grounding takes priority over fluent output. Each answer is generated from the selected
paper's retrieved chunks, then checked for its cited paper/version, copied source quote,
numeric values, requested benchmark or model size, and key action. Unsupported or invalid
answers abstain. This conservative policy can reject a true answer when retrieval or PDF
extraction is weak. The separate evaluation records retrieval misses and answer failures,
including the first-pass scores, so the final tuned regression is not mistaken for a blind
accuracy estimate.

Synthetic services isolate workflow correctness from network/model variability. They use a
fictional document and cannot establish real retrieval or answer quality. Live QA checks
source quotes and citations, but subtle semantic entailment still needs manual review and the
broader evaluation on new papers and questions.

With more time, I would test on a wider, unseen paper set, add OCR for scanned PDFs,
improve table/equation structure recovery, and compare the local 3B model with a stronger
local model under the same citation checks. These improvements would be measured with
human-reviewed source evidence rather than by answer fluency alone.

No frontend, REST API, deployment, authentication, non-arXiv ingestion, fine-tuning, or scheduled
monitoring is included. LoRA, Transformer, and QLoRA briefing examples are available in `outputs`.
The fixed 15-question evaluation passed after corrections; it is a small regression set,
not a general accuracy estimate. Two separate fresh 10-question first passes scored 6/10
and 8/10; their raw results and later fixes are in the Step 17 report. No reflection video is claimed at this
milestone. These remain visible in the acceptance checklist.

## Updating the dependency lock

After changing dependency bounds, resolve and retest in Python 3.11:

```sh
python -m piptools compile --extra dev --allow-unsafe --strip-extras --resolver backtracking --output-file requirements.lock pyproject.toml
python -m pip install -r requirements.lock
python -m pip check
```

The tested model tag/digest and dependency versions are recorded in the foundation verification
report; Step 5 behavior is recorded in the Step 5 verification report.
Downloaded papers, vectors, models, environment files, and logs must not be committed.
