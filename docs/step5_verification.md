# Step 5 input understanding verification

Verified on 2026-09-25 with Python 3.11.12 and the existing local Ollama
`qwen2.5:3b` model. Step 5 stops at a normalized lookup intent or a topic query plan;
it makes **no arXiv API request** and generates no briefing.

## Behavior implemented

- Modern and legacy IDs, optional version suffixes, `arXiv:` prefixes, direct `/abs/` and
  `/pdf/` links, and bare `arxiv.org` links normalize to one identifier.
- Invalid ID formats, unrelated hosts, URL credentials, ports, query strings, fragments,
  and unrelated arXiv paths fail with recovery instructions.
- Topic search obtains a validated Qwen JSON interpretation. At most three useful phrases
  whose substantive words occur in the user topic are retained. Missing topic words can be
  filled from sanitized input text. If local Qwen fails or its output is invalid, the
  sanitized fallback remains usable and a warning is shown.
- Date filters come from explicit user wording rather than unverified model dates. Bare
  `recent`/`latest`/`newest` means the prior 12 calendar months, using the UTC date when
  the command runs. Explicit date ranges override those words. Unsupported or reversed
  date requests produce an error instead of silently dropping the constraint.
- Python quotes each sanitized phrase under the arXiv `all:` field and adds an inclusive
  GMT `submittedDate` range where applicable. The next milestone will send this expression
  through the official API client and handle empty results.

## Checks

| Check | Result |
| --- | --- |
| Full offline test suite | 109 passed |
| Ruff lint | Passed |
| Modern, legacy, versioned IDs and paper URLs | Passed |
| Unsupported URLs and malformed IDs | Rejected in tests |
| Date precedence, month rollover and leap-day clamp | Passed |
| Model unavailable/invalid output fallback | Passed using injected failures |
| Query operator injection through model terms | Rejected/sanitized in tests |
| Live local Qwen topic interpretation | Passed; ~9 seconds for the final example |

Live example input: `recent work on KV-cache compression for LLMs`.

Observed output: terms `KV-cache compression` and `LLMs`, date range
`2025-09-25` through `2026-09-25`, and query
`all:"KV-cache compression" AND all:"LLMs" AND submittedDate:[202509250000 TO 202609252359]`.
Qwen's first attempt proposed unrelated terms including `LLM performance`; the validation
rule was tightened, and the final live run no longer included them.

No manual action is required. To inspect Step 5 interactively in a normal terminal:

```sh
source .venv/bin/activate
python -m arxiv_agent inspect-input "2106.09685v2"
python -m arxiv_agent inspect-input "recent work on KV-cache compression for LLMs"
```

The second command needs the Ollama app/server running; if it is unavailable, the command
returns a sanitized fallback and prints a warning. A complete paper digest still requires
the later retrieval, parsing, indexing, and QA steps.
