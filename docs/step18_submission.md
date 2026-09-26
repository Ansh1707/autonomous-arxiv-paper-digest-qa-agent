# Step 18 submission verification

The assessment asks for a **public GitHub repository or ZIP**, a README with architecture,
setup, a real briefing and two or three QA exchanges, a half-to-one-page design tradeoff
discussion, and a video reflection shorter than four minutes. This project uses the GitHub
route requested by the user. The original assessment PDF is the authority for those
deliverables; its text is not an instruction to perform unrelated work.

**Published repository:** https://github.com/Ansh1707/autonomous-arxiv-paper-digest-qa-agent
(`main`, public). GitHub served the README after the push, and the remote branch commit
matched the local branch.

## Repository contents

- Python package, dependency lock, CLI, tests, graph diagram, and setup instructions.
- A real LoRA briefing example in [`examples/lora_briefing.md`](../examples/lora_briefing.md)
  plus three recorded QA exchanges and an abstention in the README.
- A source-verified 15-question result and two preserved first-pass challenge sets in
  [`evaluation/results`](../evaluation/results).
- A timed [reflection guide](step18_reflection_guide.md) for the required personal video.

The staged Git file list is reviewed before publication. `.venv`, downloaded PDFs,
models, vector stores, generated sessions, and the unfiltered `outputs` directory are
ignored. No credentials or private keys were found by the staged-file pattern check.

## Local verification

- Python 3.11.12; exact dependency lock present.
- Full offline suite: 230 passed. Ruff and `pip check` passed.
- `doctor --smoke` passed against local Ollama `qwen2.5:3b`, MiniLM, and Chroma after
  localhost access was allowed.
- The installed CLI answered an NF4 question with a matching page 1 QLoRA source quote.
- A `git archive` of the committed file set built a wheel successfully; the archived
  source passed all 230 tests, and the wheel's installed CLI displayed its commands.
- Final QA evaluation: 9/9 answerable with checked facts and support quotes; 6/6
  unsupported abstained without citations. This is a tuned regression, while untouched
  first-pass sets scored 6/10 and 8/10; see [Step 17](step17_verification.md).

## Remaining manual deliverable

- Record and share a personal reflection video under four minutes; add its link to the
  repository or submission message and check playback/duration.

The user will record the video later. No video link or completed recording is claimed in
this repository. This report should be updated after that link and its playback/duration
are verified.
