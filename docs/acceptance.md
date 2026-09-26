# Assessment acceptance checklist

Source: `AI-Intern-Assessment_E2CF564601.pdf`, all four pages. The user confirmed that this PDF,
not the unrelated support-ticket DOCX or any CSV, governs the work. The user subsequently
authorized Steps 5–18. A skeleton passing does not mean that the final assessment is complete.

## Completed milestones

| Step | Acceptance condition | Evidence | Status |
| --- | --- | --- | --- |
| 1 | Every assessment requirement maps to implementation and verification | Table below | Implemented |
| 2 | Isolated Python 3.11, exact dependency lock, local Qwen availability, embedding/tokenizer assets, tiny real generation/embedding/Chroma smoke | `doctor`, lock, verification report | Verified |
| 3 | Modular CLI/package, validated settings, ignored runtime folders, stage timing logs | CLI/config tests | Implemented |
| 4 | Typed state/records, explicit nodes/edges, both input routes, bounded retry and parsing failure path | Offline graph and contract tests | Implemented |
| 5 | Normalize arXiv IDs/URLs; interpret topics/dates with Qwen; construct safe arXiv expressions; deterministic fallback | Input graph, tests, live local model check | Verified |
| 6 | Official API metadata lookup/topic search; exact versions; bounded candidates; dates; timeout, retry, broadening | Discovery graph, offline fixtures, live API checks | Verified |
| 7 | Embed topic and candidate metadata with local MiniLM; rank at most ten; select top or user override | Selection graph, ranking fixtures, live model check | Verified |
| 8 | Download selected versioned PDF; bound size/time/pages; verify SHA-256 and cache integrity; stop on corrupt/HTML responses | Download graph, failure fixtures, live LoRA PDF | Verified |
| 9 | Extract page-aware PDF text, abstract, sections, references; preserve source positions; detect text-poor/scanned papers | Parser fixtures, live LoRA review | Verified |
| 10 | Split page/section text into bounded overlapping chunks, embed locally, and persist a versioned provenance-aware Chroma index | Chunk/storage tests and live LoRA/Transformer retrieval | Verified |
| 11 | Select verbatim, source-linked passages for problem, method, results, and stated limitations; mark missing limitations explicitly | Evidence fixture tests and two live paper reviews | Verified |
| 12 | Generate a complete executive briefing with Qwen2.5:3b, validated source quotes/numbers, explicit limitations, and Markdown/JSON artifacts | Contract tests and live LoRA/Transformer briefings | Verified |
| 13 | Audit full main-body evidence coverage in Qwen-token-bounded batches; retain candidate and reduced notes with source citations, including later results | Coverage fixture and two live indexed papers | Verified |
| 14 | Publish metadata-derived, source-checked JSON and Markdown briefings with required limitations, separate processing caveats, and three follow-up questions | Live LoRA/Transformer `b3` artifacts, quote audit, full tests | Verified |
| 15 | Retrieve fresh selected-paper evidence for each QA turn; answer with checked citations or the exact insufficient-evidence response | QA fixtures and live LoRA/Transformer answer, follow-up, and abstention checks | Verified |
| 16 | Save and reopen QA sessions with artifact/index checks, stage progress, and interactive `/exit` and `/sources` | Snapshot failure tests and separate-process LoRA QA | Verified |
| 17 | Run the fixed 15-question evaluation, fix unsupported claims and retrieval misses, and document remaining limitations | First-pass, two fresh challenge sets, final source-verified 15/15 JSON, source review, 230 offline tests | Verified as a small regression set |

## Final application requirements

| ID | Requirement | Planned component | Final verification | Current status |
| --- | --- | --- | --- | --- |
| R01 | Natural-language topic input | Intent and arXiv search services | Live topic search | Metadata search and selection verified |
| R02 | Specific arXiv ID or URL input | Input normalization and metadata lookup | Modern/legacy/versioned IDs, abs/pdf URLs | Metadata lookup verified; further corpus evaluation later |
| R03 | Official API metadata: title, authors, abstract, PDF, categories, date | arXiv service, PaperMetadata | Live API record plus fixtures | Verified; selected version's PDF fetched |
| R04 | Explicit stateful graph | LangGraph builder and SessionState | Routes, state updates, trace, diagram | Live through QA with durable session snapshots |
| R05 | Topic selection/ranking | Embedding-based candidate ranker | Relevance examples and override | Live topic-to-briefing run and MiniLM ranking verified; broader evaluation later |
| R06 | Fetch/parse sections, abstract, references | Downloader and parser | Visual comparison and failure fixtures | Live LoRA PDF parsed; OCR remains unsupported |
| R07 | Chunk/embed/store in vector DB | Chunker, MiniLM, Chroma | Token/provenance and retrieval tests | Implemented and verified on three papers; broader relevance review later |
| R08 | Briefing bibliographic fields and one-paragraph summary | Briefing schema and renderer | Schema and manual source review | Live Markdown/JSON examples; human source review recommended |
| R09 | Problem, method bullets, key claims, explicit limitations, follow-up questions | Evidence extraction and briefing | All fields, numerical claim review | Three questions, source citations, and separate processing notes verified |
| R10 | Follow-up QA grounded in vector retrieval | QA graph and citation validator | Nine answerable questions | Final fixed set: 9/9 with expected fact and citation; untouched evaluation still needed |
| R11 | Say when paper evidence is insufficient | QA abstention | Six unsupported questions | Final fixed set: 6/6 abstentions; untouched evaluation still needed |
| R12 | State survives transitions from briefing to QA | Shared state and session snapshots | State test and process reopen | Separate-process reopen verified with preserved paper, index, and turns |
| R13 | At least one realistic graceful failure | Error node, bounded retries | Empty search and parse failure | Metadata and unreadable-PDF failure paths verified offline |
| R14 | Local run without paid API keys, Python preferred | Local stack and setup instructions | Fresh setup, real smoke | See verification report |
| R15 | Public GitHub repository with setup | Submission packaging | Public URL and committed-export check | Verified: [public repository](https://github.com/Ansh1707/autonomous-arxiv-paper-digest-qa-agent), remote `main`, README, wheel build, 230 tests |
| R16 | README graph, state shape, setup, example and 2–3 QA exchanges | README/docs | Actual recorded runs | Verified: graph/state docs, setup, real briefing and three QA examples |
| R17 | Design decisions/tradeoffs, roughly half to one page | README | Final review | Verified: choices, limits, and next improvements |
| R18 | Reflection video, no more than four minutes | Video | Playback and duration check | Recording by the user is required; timed guide prepared |

## Scope and priorities

- Default: one selected paper per session; retrieve/rank up to ten candidates.
- Required: all briefing fields, vector retrieval, abstention, explainable graph, graceful failure.
- Optional after required work: better ranking presentation and additional parsing heuristics.
- Excluded: UI beyond CLI/notebook, auth/deployment/production infra, non-arXiv sources,
  fine-tuning. REST APIs and scheduling are not needed.
- No CSV or training dataset is required. Metadata collection through the official API is implemented.
- Time box: roughly 2–3 days for the whole assessment; reserve 2–3 hours for README/video.

## Rubric alignment

| Area | Weight | Priority |
| --- | --- | --- |
| Graph/state design | 25% | Explicit routes and state; show design clearly |
| Correctness/grounding | 25% | Evidence-supported briefing/QA and abstention |
| Retrieval/parsing | 20% | Correct text order, provenance, sensible chunks |
| Code quality | 15% | Modular contracts, controlled failures, tests |
| Communication | 15% | Reproducible README, real examples, reflection |

The Step 17 fixed evaluation uses pinned v1 versions of Attention Is All You Need
(1706.03762), LoRA (2106.09685), and QLoRA (2305.14314). It contains five reviewed
questions per paper: three answerable with source evidence and two unsupported. Because
the first held-out run informed corrections, the final 15/15 is a regression result.
Two later untouched 10-question first passes scored 6/10 and 8/10 before their failures
were investigated. Broader evaluation on new papers is still needed for a generalization estimate.
