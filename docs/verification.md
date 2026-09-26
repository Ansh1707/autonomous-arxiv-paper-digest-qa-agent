# Steps 1–4 verification

Verified on 2026-09-25 on the target Apple M1 MacBook Air with 8 GB memory. The workspace
started empty. The only Python environment used for this project is `.venv` with Python
3.11.12. All commands below were run in the project directory. The installed dependencies
are exactly the 127 pinned packages in `requirements.lock`.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Dependency consistency | PASS | `python -m pip check`: no broken requirements |
| Lock consistency | PASS | All 127 pinned installed versions match `requirements.lock` |
| Offline tests | PASS | `python -m pytest -q`: 53 passed |
| Lint | PASS | `python -m ruff check .`: all checks passed |
| Local model detection | PASS | `qwen2.5:3b` in Ollama, digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| Structured generation | PASS | Qwen returned `{"ok": true}` with 4,096-token context |
| Embedding | PASS | MiniLM produced a normalized 384-dimensional CPU vector |
| Qwen tokenizer | PASS | Eight tokens for the tiny diagnostic text |
| Vector persistence | PASS | Chroma data written by a subprocess was reopened and queried successfully |
| Graph routing | PASS | Lookup and topic routes, one broadening pass, QA loop, parse failure, strict retry budget, invalid state output, and failure gating exercised by tests |

Pinned downloaded asset revisions:

| Asset | Revision |
| --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Qwen2.5 3B Instruct tokenizer files | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

The generation weights come from the already installed Ollama `qwen2.5:3b` model. No second
copy of Qwen weights was downloaded. `doctor --smoke` ran against the final pinned model
revisions and exited successfully.

## Boundaries of this verification

The synthetic graph fixtures test control flow and state only. They do not demonstrate live
arXiv metadata retrieval, PDF extraction quality, paper chunking, evidence-supported briefing
content, or substantive QA. Those remain pending under the acceptance checklist. The tiny
generation check proves the local model works and respects a JSON schema for one trivial case;
it cannot establish reliability on long scientific papers.

The environment was installed and checked on this Mac. A fresh-environment setup rehearsal is
reserved for the final delivery step. The assessment does not require manual user action for
this milestone; the user can optionally run `source .venv/bin/activate` and
`python -m arxiv_agent doctor --smoke` in their normal terminal to see the same checks.

The Codex command sandbox initially denied package registry and localhost access, so the
dependency/model download and Ollama check were rerun with the required execution permission.
Both then completed successfully. An interrupted editable-install command was finished and
verified before this report was written.
