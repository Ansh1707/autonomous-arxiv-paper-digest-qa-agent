# Step 15 verification: grounded follow-up QA

`ask-paper` implements the original plan's grounded QA step. It opens a checked `b3`
briefing and matching `e4` evidence for one exact paper version, then runs a LangGraph QA
turn for each question. Retrieval uses the selected paper's Chroma index. It queries 12
semantic candidates and supplements them with exact-term matches, excludes references by
default, removes near-duplicates, and sends at most six Qwen-token-bounded passages to local
`qwen2.5:3b`. Bibliography questions can include references. A short window of previous user
questions resolves phrases such as “that”; previous assistant answers are never source evidence.

Qwen supplies an answer and source IDs. The validator checks that IDs belong to the supplied
passages and selected paper/version, selects exact support sentences from those chunks, and
rejects unsupported numbers or a key action absent from the source. A malformed or unsupported
draft gets one correction attempt; persistent failure returns exactly:

> I couldn’t find enough evidence in the retrieved paper text to answer that.

The output includes source chunk IDs, PDF page links, and copied support sentences.
Conversation turns exist in memory for one command. Durable session snapshots are Step 16.

Live checks on 2026-09-26:

- LoRA: “What does LoRA freeze during training?” answered that the pre-trained model weights
  are frozen, citing the page 1 abstract. An earlier model draft wrongly said trainable
  matrices were frozen; the action support check rejected it, and exact-term retrieval
  supplied the correct source.
- LoRA: “What limitation does LoRA report about batching tasks?” and “Why is that difficult?”
  ran in one session. Both answers cited the explicit page 4 limitation; the second retrieval
  query used the previous *question* to resolve “that.”
- Transformer: “What BLEU score did Transformer big achieve on WMT 2014 English-to-German
  translation?” answered **28.4 BLEU** from page 7. A mixed-task draft of **41.0** was
  rejected; the final retrieval restricted this simple metric question to the direct
  English-to-German prose passage.
- Unsupported questions about an author's favorite food and a Mars-colony dataset in 2030
  returned the exact abstention with no citations.
- Offline tests cover source/paper mismatch, references, duplicates, history isolation,
  numeric and action checks, bad citation repair, abstention, and audited briefing reopening.
  The full suite passes with 192 tests; Ruff lint and `pip check` also pass.

These are focused functional checks, not a measured accuracy claim. A broader scored set of
answerable and unsupported questions remains for final evaluation. Manual review of answer
meaning against the linked PDF is recommended before sharing scientific claims; code cannot
prove every semantic relationship merely by matching citation text. No additional setup is
needed on the current machine.
