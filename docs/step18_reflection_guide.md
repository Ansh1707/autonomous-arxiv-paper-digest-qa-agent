# Step 18 reflection video guide (maximum four minutes)

The assessment requires a short video reflection on the approach. Record this in your own
voice and words; the outline below is a prompt, not a claim that a video already exists.
Aim for **3:15–3:40** so the recording remains below the four-minute cutoff. Show the
repository README, the architecture diagram, one real briefing, and one cited QA answer.
The sample briefing lives at [`examples/lora_briefing.md`](../examples/lora_briefing.md).

| Time | Screen | Points to cover |
| --- | --- | --- |
| 0:00–0:25 | README title and assessment goal | Topic or arXiv ID enters a local paper-digest and QA workflow. State the choice of Qwen2.5:3b and local tools. |
| 0:25–1:05 | Architecture diagram | Walk through intent parsing, arXiv metadata/search, ranking, PDF fetch/parse, chunk/embed/index, evidence/briefing, and QA loop. Explain typed shared state and saved sessions. |
| 1:05–1:45 | Example briefing | Show title/authors/date, plain-English summary, method, key results, explicit limitations, three follow-up questions, and source-page links. |
| 1:45–2:25 | README QA example or a live `ask-paper` run | Show an answer with copied source quote and page citation, then an unsupported question that abstains. Explain that the answer is constrained to one selected paper's retrieved chunks. |
| 2:25–3:05 | Verification report | Explain the 15-question, three-paper regression: nine answerable and six unsupported after fixes. State honestly that fresh first-pass sets scored 6/10 and 8/10, so broader evaluation is still needed. |
| 3:05–3:40 | README tradeoffs | Note the local 3B model/8 GB memory tradeoff, parser and OCR limits, and what you would improve next. |

Suggested opening, to adapt rather than read verbatim: “I built a local, stateful arXiv
paper briefing and question-answering agent. It takes either a topic or an arXiv ID,
selects and reads a paper, then writes a source-linked briefing. I chose LangGraph for
explicit stages, MiniLM plus Chroma for retrieval, and Qwen2.5:3b through Ollama so the
project runs without a paid key.”

Before sharing, check that the video plays with audio, the text is legible, personal or
secret desktop content is hidden, and the duration is **under 4:00**. Put a shareable video
link in the README's submission section, or provide a small video file for the repository.
The GitHub repository itself is the code submission; this video remains a separate required
deliverable until a playable link or file is supplied.
