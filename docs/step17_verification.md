# Step 17 verification — three-paper evaluation and failure review

The fixed [15-question set](../evaluation/questions.json) uses exact v1 PDFs for LoRA
(`2106.09685v1`), Transformer (`1706.03762v1`), and QLoRA (`2305.14314v1`). Each paper has
three answerable and two unsupported questions. LoRA was the development paper. Transformer
and QLoRA were held out for the first scored run. Expected facts and PDF pages were checked
against the downloaded source papers before running the model. Unsupported cases include
absent personal facts, missing ImageNet and translation results, a HELM/Vicuna benchmark
switch, and an external hardware price. The QLoRA PDF is 26 pages and parsed with text on
the needed pages, including its explicit limitations section on PDF page 15.

The [evaluation runner](../scripts/evaluate_qa.py) opens each existing briefing and index,
starts an independent QA state, invokes the graph, and records status, answer, source quotes,
retrieved pages, elapsed time, and separate retrieval and answer checks. Results are saved
after each case so an interrupted run retains completed observations. Run it from the project
root after installing the package and starting Ollama:

```sh
python scripts/evaluate_qa.py --output evaluation/results/step17-final.json
```

| Run | Answerable with expected fact and citation | Unsupported abstentions | Notes |
| --- | ---: | ---: | --- |
| Development first pass, LoRA | 3/3 | 2/2 | All three source pages retrieved. |
| Held-out first pass, Transformer + QLoRA | 4/6 | 2/4 | Four failures exposed retrieval/grounding gaps. |
| Earlier 15-case rerun | 9/9 | 6/6 | First successful regression after the held-out fixes. |
| Final source-verified 15-case run | 9/9 | 6/6 | [Result JSON](../evaluation/results/step17-final-source-verified.json); all nine acceptable source passages retrieved; 243.0 seconds total QA time. |
| First untouched challenge, 10 new cases | 4/6 | 2/4 | [Raw first pass](../evaluation/results/step17-challenge.json); 6/10 overall. |
| Second fresh validation, 10 new cases | 4/6 | 4/4 | [Raw first pass](../evaluation/results/step17-fresh-validation.json); 8/10 overall. |

The initial held-out pass had **two of six answerable failures** (Transformer
training time/hardware and QLoRA's unevaluated benchmark list) and **two of four unsupported
failures** (English-to-Spanish BLEU and HELM percentage), so it scored **4/6 answerable and
2/4 unsupported**. The raw
[first-pass held-out results](../evaluation/results/step17-held-out.json) preserve every
answer and citation. These papers were used to diagnose the failures after the first run.
The final rerun is therefore a regression result on a small fixed set, **not** an untouched
generalization estimate.

Corrections, in priority order:

1. Metric answers now require support for the requested language pair or benchmark. If
   that scope is absent from retrieved paper text, QA abstains before calling Qwen. This
   prevents moving the 41.0 BLEU English-to-French result to English-to-Spanish and the
   Vicuna 99.3% result to HELM.
2. Citation quotes may include up to three adjacent source sentences when an answer needs
   linked facts, such as 8 P100 GPUs and 12 hours on PDF page 7. Unsupported numbers still
   fail validation.
3. An explicit “did not evaluate” benchmark question focuses on its limitation passage.
   When the passage itself enumerates the benchmarks, a narrow extractive answer copies the
   names with a citation. This handled QLoRA's BigBench, RAFT, and HELM list on page 15.
4. A QLoRA briefing claim falsely presented a *conditional* possibility of scoring above
   ChatGPT as an observed result. The briefing validator now rejects that comparison, and
   the regenerated artifact instead quotes the observed ordering effect. Final QA citations
   also discard redundant passages that contribute no material answer support.
5. The first untouched challenge revealed missed residual-dropout evidence, confusion over
   Double Quantization, a counterfactual attention-to-LSTM BLEU answer, and an unsupported
   claim that Guanaco beat GPT-4. Its raw 6/10 result remains preserved. Narrow direct
   extraction and checks for model comparison and counterfactual replacements resolved
   those observed cases; those reruns are regressions, not independent validation.
6. The second fresh set exposed a missed “8 instruction datasets” passage and an answer
   that borrowed 65B hardware facts for a 33B question. Exact subject-phrase retrieval,
   sentence-level model-size/hardware matching, and checks after citation pruning addressed
   these errors. The two failures passed targeted live regressions, but the 8/10 first-pass
   score remains the independent observation.
7. A subsequent whole-set regression identified one overly literal “replace” validator:
   the Transformer abstract says “dispensing with recurrence and convolutions.” The
   validator accepts that paraphrase while still requiring evidence for a proposed
   replacement *with* a named component. Two other reported misses were evaluator rubric
   errors: the correct BLEU fact is stated on both pages 1 and 7, and the 65B/48GB fact
   on both pages 1 and 2. The fixed rubric lists those alternative source passages
   explicitly, and the complete final run passed under it.
8. Manual review of an earlier automated 15/15 run found that the NF4 answer's citations
   did not quote the crucial “optimal for normally distributed weights” statement. That
   earlier score was a false positive of the answer-only fact check. The final evaluator
   also checks expected facts in copied support quotes; NF4 now uses a direct page 1 quote.
   The earlier [automated run](../evaluation/results/step17-final-verified.json) is retained
   as an audit trail, not presented as the final source-verified result.

Manual review compared each of the nine final answers and its cited source text with the
source PDF pages. The numerical statements checked include LoRA's 10,000-times trainable
parameter reduction, Transformer big's 28.4 English-to-German BLEU, Transformer base's
12-hour/8-P100 training, and QLoRA's 65B model on a 48GB GPU. The final QLoRA briefing's
numeric model-evaluation claim has its matching values in the page 10 quote. No unsupported
numeric claim was found in the three final demo briefings. All final citations point to the
correct paper version. The six final unsupported answers have no citations.

The full offline suite passes with **230 tests**. It covers the original mandatory areas:
input variants, search/retry/rate limits, PDF validation/interruption, one- and two-column
parsing plus a headingless document, chunk/index identity and reopening, malformed/timeout
generation, answer grounding and follow-ups, graph error routing, and session corruption.
Ruff and `pip check` pass. Live ID-to-briefing and topic-to-briefing runs both completed for
QLoRA v1; cached indexing was reused on the topic route. A saved LoRA QA session reopened
in another process in Step 16 without fetching or embedding the paper again. No paid key was
used. After reinstalling the current package, the installed `ask-paper` CLI answered the
Transformer base residual-dropout question as `0.1` with a page 7 source quote.

Limits: the 15-question final result is a **tuned regression**, not a blind accuracy
estimate. The two untouched 10-case first passes scored 6/10 and 8/10, and their discovered
failures informed later corrections. The evaluation uses three English, text-based papers;
the deterministic checks look for expected facts and source-page citations. Manual review
improves confidence but cannot prove every paraphrase. Qwen can vary across runs. OCR for
scanned PDFs, complex equation/table meaning, non-arXiv sources, and broader paper-level
accuracy are outside the demonstrated scope.
