# Step 16 verification — saved QA sessions

Step 16 adds atomic snapshots in `data/sessions/<session_id>.json`. `ask-paper` creates a
ready snapshot before the first question and updates it after each successful answer.
`chat --session ID "question"` reopens it in a new process. Omitting the question in a
terminal starts an interactive loop; `/sources` shows the latest answer's PDF page links,
and `/exit` ends the loop. Stage names appear on stderr. Scripted answers remain JSON on
stdout. A failed QA turn leaves the prior ready snapshot available for retry.

Reopening checks the session ID and schema, completed ready status, selected paper/version,
briefing and evidence artifact content, index fingerprint, embedding model revision, and
chunk count. It uses the existing Chroma index; it does not download or embed the paper.
Missing or incompatible files return an error with recovery advice. Snapshots use a
temporary file and atomic replacement, with owner-only file permissions.

Live check on 2026-09-26:

1. `ask-paper 2106.09685v1 "What does LoRA freeze during training?"` created session
   `8837226f-80eb-4a9a-bdbd-1e7a2886e4c7`. Qwen2.5:3b answered that LoRA freezes the
   pre-trained model weights, with a source quote from page 1.
2. A separate CLI process ran `chat --session 8837226f-80eb-4a9a-bdbd-1e7a2886e4c7
   "What limitation does LoRA report about batching tasks?"`. It reopened the same session
   and answered with a page 4 source quote.
3. A third process loaded the saved snapshot and confirmed two conversation turns and the
   same `paper-8a55beb68c48109c3632fabc1b8a5b669b69fae8` Chroma collection.
4. An installed-package terminal session ran `chat --session ID`; `/sources` printed the
   page 4 link from the previous turn and `/exit` closed with status 0.

The offline suite covers snapshot round-trip, corrupt JSON, invalid/missing ID, mismatched
briefing/index, cross-paper citation, interrupted atomic replacement, and interactive
`/sources` and `/exit`. The full suite passed with 198 tests, Ruff passed, and `pip check`
found no broken requirements. Manual check: open the cited PDF pages and confirm the quoted
text and answer wording. The first live attempt inside the restricted sandbox could
not reach localhost Ollama; the approved local-network run succeeded.
