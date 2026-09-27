import hashlib

import pytest

from arxiv_agent.contracts import (
    Briefing,
    Chunk,
    ConversationTurn,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceNote,
    ParsedBlock,
    ParsedPaper,
    ParsedSection,
    QAAnswer,
    RetrievalIndex,
    SessionState,
    Stage,
)
from arxiv_agent.graph import build_qa_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.briefing import BRIEFING_VERSION, BriefingArtifact, QuoteAudit
from arxiv_agent.services.evidence import EVIDENCE_VERSION
from arxiv_agent.services.indexing import ChromaIndexStore
from arxiv_agent.services.qa import (
    ABSTENTION,
    AnswerDraft,
    OllamaAnswerGenerator,
    QAService,
    _current_briefing_path,
    _direct_source_answer,
    _essential_citations,
    _support_quote,
    load_qa_session,
)
from arxiv_agent.services.synthetic import synthetic_paper


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class Store:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def query(self, index, question, limit):
        self.queries.append((question, limit))
        return self.hits[:limit]

    def all_chunks(self, index):
        return [chunk for chunk, _ in self.hits]


class Generator:
    def __init__(self, drafts):
        self.drafts = drafts
        self.calls = []

    def generate(self, question, context_questions, passages, feedback=""):
        self.calls.append((question, context_questions, passages, feedback))
        return self.drafts[min(len(self.calls) - 1, len(self.drafts) - 1)]


def chunk(number, text, section="Method", *, arxiv_id="0000.00000", page=2):
    return Chunk(
        chunk_id=f"chunk-{number}",
        arxiv_id=arxiv_id,
        version=1,
        section=section,
        page_start=page,
        page_end=page,
        text=text,
        embedding_tokens=30,
        source_block_ids=[f"block-{number}"],
    )


def state(question="What does the method freeze?"):
    paper = synthetic_paper()
    claim = EvidenceClaim(text="The paper reports a result.", chunk_ids=["chunk-1"])
    briefing = Briefing(
        paper=paper,
        plain_english_summary=claim,
        problem_statement=claim,
        method=[claim],
        key_results=[claim],
        limitations_status="not_found",
        limitations=[],
        limitations_note="No explicit limitation was found.",
        follow_up_questions=["What is the problem?", "How does it work?", "What changed?"],
    )
    return SessionState(
        user_input="0000.00000v1",
        selected_paper=paper,
        index=RetrievalIndex(collection="fake", fingerprint="f" * 64, chunk_count=3),
        briefing=briefing,
        status="ready",
        question=question,
    )


def answered(source_id="C1", text="The method freezes the original weights."):
    return AnswerDraft(
        status="answered",
        text=text,
        source_ids=[source_id],
    )


def service(settings, chunks, drafts):
    store = Store([(item, 0.3) for item in chunks])
    generator = Generator(drafts)
    return (
        QAService(settings, store=store, generator=generator, tokenizer=Tokenizer()),
        store,
        generator,
    )


def test_retrieval_filters_references_duplicates_and_other_papers(settings):
    original = chunk(1, "The method freezes the original weights and trains an adapter.")
    duplicate = chunk(2, original.text)
    reference = chunk(3, "A cited paper freezes another model's weights.", "References")
    qa, store, _ = service(settings, [original, duplicate, reference], [answered()])
    selected = qa.run_stage(Stage.RETRIEVE, state())
    assert [item.chunk_id for item in selected["retrieved_chunks"]] == ["chunk-1"]
    assert store.queries[0][1] == 12
    bibliography = state("Which cited paper discusses freezing weights?")
    assert reference in qa.run_stage(Stage.RETRIEVE, bibliography)["retrieved_chunks"]
    bad = chunk(4, "Another paper has different results.", arxiv_id="9999.99999")
    qa, _, _ = service(settings, [bad], [answered()])
    with pytest.raises(StageFailure, match="another paper/version"):
        qa.run_stage(Stage.RETRIEVE, state())


def test_weak_lexical_overlap_cannot_bypass_dense_distance_cutoff(settings):
    weak = chunk(1, "Training uses an adapter in each layer.")
    store = Store([(weak, 0.96)])
    qa = QAService(settings, store=store, generator=Generator([]), tokenizer=Tokenizer())
    chosen = qa.run_stage(
        Stage.RETRIEVE, state("What causes training memory to decrease?")
    )["retrieved_chunks"]
    assert chosen == []


def test_multi_term_lexical_rescue_has_separate_relevance_gate(settings):
    direct = chunk(1, "Training with the adapter reduces memory use.")
    store = Store([(direct, 0.96)])
    qa = QAService(settings, store=store, generator=Generator([]), tokenizer=Tokenizer())
    chosen = qa.run_stage(
        Stage.RETRIEVE, state("How does training reduce memory use?")
    )["retrieved_chunks"]
    assert chosen == [direct]


def test_fresh_retrieval_uses_questions_only_for_referential_followup(settings):
    passage = chunk(1, "The method freezes the original weights and trains an adapter.")
    qa, store, _ = service(settings, [passage], [answered()])
    current = state("How does it train them?")
    current.conversation = [
        ConversationTurn(
            question="What does the method freeze?",
            answer=QAAnswer(status="insufficient_evidence", text="Invented assistant text"),
        )
    ]
    retrieved = qa.run_stage(Stage.RETRIEVE, current)
    assert retrieved["retrieval_query"] == "What does the method freeze? How does it train them?"
    assert "Invented assistant text" not in store.queries[0][0]


def test_named_task_metric_prefers_its_direct_prose_over_neighboring_table(settings):
    german = chunk(
        1,
        "On the WMT 2014 English-to-German task, the model achieved 28.4 BLEU.",
        section="Results",
    )
    french = chunk(
        2,
        "English-to-French results: 28.4 41.0 BLEU in a flattened table.",
        section="Results",
    )
    qa, _, _ = service(settings, [german, french], [answered()])
    selected = qa.run_stage(
        Stage.RETRIEVE, state("What BLEU score did the model achieve on English-to-German?")
    )
    assert selected["retrieved_chunks"] == [german]


def test_answer_validation_and_graph_append_turn(settings):
    passage = chunk(1, "The method freezes the original weights.")
    qa, _, generator = service(settings, [passage], [answered()])
    current = SessionState.model_validate(build_qa_graph(qa, settings).invoke(state()))
    assert current.error is None and current.status == "ready"
    assert current.answer.status == "answered"
    assert current.answer.citations[0].chunk_id == passage.chunk_id
    assert current.answer_support_quotes[0].source_quote == passage.text
    assert len(current.conversation) == 1
    assert generator.calls[0][2][0][0] == "C1"


@pytest.mark.parametrize(
    "bad",
    [
        answered(source_id="C99"),
        AnswerDraft(
            status="answered",
            text="The method freezes the original weights.",
            source_ids=["C1", "C1"],
        ),
        answered(text="The method freezes 92 parameters."),
        answered(text="The method freezes the original weights. It also adds an adapter."),
    ],
)
def test_invalid_citation_or_number_repairs_then_abstains(settings, bad):
    passage = chunk(1, "The method freezes the original weights.")
    qa, _, generator = service(settings, [passage], [bad, bad])
    current = SessionState.model_validate(build_qa_graph(qa, settings).invoke(state()))
    assert current.answer.status == "insufficient_evidence"
    assert current.answer.text == ABSTENTION
    assert not current.answer.citations
    assert len(generator.calls) == 2 and generator.calls[1][3]
    assert current.stage_history.count(Stage.REQUERY) == 1
    assert current.stage_history.count(Stage.RETRIEVE) == 2
    assert current.qa_retrieval_attempts == 1


def test_failed_draft_retrieves_new_evidence_and_answers(settings):
    settings = type(settings)(**(
        settings.model_dump() | {"evidence_chunks": 1}
    ))
    first = chunk(1, "Residual dropout is mentioned without a value.")
    second = chunk(2, "The residual dropout value is 0.1.")
    bad = answered(text="The residual dropout value is 92.")
    good = answered(text="The residual dropout value is 0.1.")
    qa, store, generator = service(settings, [first, second], [bad, bad, good])
    current = SessionState.model_validate(
        build_qa_graph(qa, settings).invoke(state("What is the residual dropout value?"))
    )
    assert current.answer.status == "answered"
    assert current.answer.citations[0].chunk_id == second.chunk_id
    assert current.stage_history.count(Stage.REQUERY) == 1
    assert current.stage_history.count(Stage.RETRIEVE) == 2
    assert current.qa_retrieval_attempts == 1
    assert current.qa_retrieval_feedback is None
    assert len(generator.calls) == 3
    assert store.queries[0][0] != store.queries[1][0]
    assert store.queries[1][1] == 2 * store.queries[0][1]


def test_model_abstention_can_recover_from_a_new_passage(settings):
    settings = type(settings)(**(settings.model_dump() | {"evidence_chunks": 1}))
    first = chunk(1, "Residual dropout is discussed without a rate.")
    second = chunk(2, "The residual dropout value is 0.1.")
    abstention = AnswerDraft(status="insufficient_evidence", text="Not stated.")
    qa, _, generator = service(settings, [first, second], [abstention, answered(
        text="The residual dropout value is 0.1."
    )])
    current = SessionState.model_validate(
        build_qa_graph(qa, settings).invoke(state("What is the residual dropout value?"))
    )
    assert current.answer.status == "answered"
    assert current.answer.citations[0].chunk_id == second.chunk_id
    assert current.stage_history.count(Stage.REQUERY) == 1
    assert len(generator.calls) == 2


def test_retrieval_recovery_abstains_after_one_new_passage(settings):
    settings = type(settings)(**(
        settings.model_dump() | {"evidence_chunks": 1}
    ))
    first = chunk(1, "Residual dropout is mentioned without a value.")
    second = chunk(2, "The residual dropout value is 0.1.")
    bad = answered(text="The residual dropout value is 92.")
    qa, store, generator = service(settings, [first, second], [bad])
    current = SessionState.model_validate(
        build_qa_graph(qa, settings).invoke(state("What is the residual dropout value?"))
    )
    assert current.answer.status == "insufficient_evidence"
    assert current.answer.citations == []
    assert current.stage_history.count(Stage.REQUERY) == 1
    assert current.stage_history.count(Stage.RETRIEVE) == 2
    assert len(store.queries) == 2
    assert len(generator.calls) == 4


def test_retrieval_recovery_budget_resets_on_next_question(settings):
    passage = chunk(1, "The method freezes the original weights.")
    bad = answered(text="The method freezes 92 parameters.")
    qa, _, generator = service(settings, [passage], [bad])
    graph = build_qa_graph(qa, settings)
    current = state()
    for question in ["What does the method freeze?", "How many parameters does it freeze?"]:
        current.question = question
        current = SessionState.model_validate(graph.invoke(current))
        assert current.answer.status == "insufficient_evidence"
        assert current.qa_retrieval_attempts == 1
        assert current.qa_retrieval_feedback is None
    assert current.stage_history.count(Stage.REQUERY) == 2
    assert len(current.conversation) == 2
    assert len(generator.calls) == 4


def test_unsupported_question_abstains_without_model_when_no_relevant_chunks(settings):
    passage = chunk(1, "The method freezes the original weights.")
    store = Store([(passage, 0.92)])
    generator = Generator([answered()])
    qa = QAService(settings, store=store, generator=generator, tokenizer=Tokenizer())
    current = SessionState.model_validate(
        build_qa_graph(qa, settings).invoke(state("What is the author's favorite food?"))
    )
    assert current.answer.text == ABSTENTION
    assert not generator.calls


def test_answer_cannot_claim_freezing_from_trainable_matrix_quote(settings):
    passage = chunk(1, "LoRA adds trainable pairs of rank decomposition matrices.")
    bad = answered(
        text="LoRA freezes the trainable pairs of rank decomposition matrices.",
    )
    qa, _, generator = service(settings, [passage], [bad, bad])
    current = SessionState.model_validate(build_qa_graph(qa, settings).invoke(state()))
    assert current.answer.text == ABSTENTION
    assert len(generator.calls) == 2


def test_qa_cannot_swap_effects_between_named_methods():
    passage = chunk(1, "Method A improves accuracy. Method B reduces memory.")
    with pytest.raises(ValueError, match="one cited source sentence"):
        QAService._check_draft(
            answered(text="Method A reduces memory."),
            [passage],
            "What does Method A reduce?",
        )


def test_uncertain_paraphrase_is_replaced_with_direct_paper_sentence():
    passage = chunk(
        1,
        "DPO avoids fitting an explicit, standalone reward model while using human "
        "preferences to optimize the policy.",
    )
    answer, quotes = QAService._check_draft(
        answered(text="DPO avoids training a separate reward model."),
        [passage], "What model does DPO avoid training?",
    )
    assert answer.text == passage.text
    assert quotes[0].source_quote == passage.text


def test_direct_source_fallback_does_not_guess_yes_no_training_phase():
    passage = chunk(
        1,
        "The pipeline samples completions to build an offline preference dataset.",
    )
    assert _direct_source_answer(
        "Does DPO require sampling during fine-tuning?", [passage]
    ) is None


def test_direct_source_fallback_requires_actual_numeric_value():
    passage = chunk(1, "Residual dropout is discussed without a rate.")
    assert _direct_source_answer(
        "What is the residual dropout value?", [passage]
    ) is None


def test_direct_source_fallback_requires_named_benchmark():
    generic = chunk(1, "Arithmetic reasoning is a task where language models struggle.")
    assert _direct_source_answer(
        "Which arithmetic reasoning benchmark is named in the paper?", [generic]
    ) is None


def test_model_cannot_answer_named_benchmark_with_generic_topic_sentence():
    generic = chunk(1, "Though simple, arithmetic reasoning can exhibit flat scaling.")
    with pytest.raises(ValueError, match="does not name"):
        QAService._check_draft(
            answered(text=generic.text), [generic],
            "Which arithmetic reasoning benchmark is named in the paper?",
        )


def test_direct_source_fallback_requires_model_scale():
    generic = chunk(1, "Chain of thought prompting works with GPT-3 models.")
    assert _direct_source_answer(
        "What scale of language models does chain of thought prompting work with?", [generic]
    ) is None


def test_qa_cannot_reverse_an_explicit_causal_relation():
    passage = chunk(1, "X causes Y.")
    with pytest.raises(ValueError, match="one cited source sentence"):
        QAService._check_draft(
            answered(text="Y causes X."), [passage], "What causes X?"
        )


def test_qa_cannot_reverse_a_reported_comparison():
    passage = chunk(1, "Method A has lower accuracy than Method B.")
    with pytest.raises(ValueError, match="one cited source sentence"):
        QAService._check_draft(
            answered(text="Method A has higher accuracy than Method B."),
            [passage], "How does Method A compare with Method B?",
        )


def test_adjacent_source_sentences_support_hardware_and_training_time():
    passage = chunk(
        1,
        "We trained our models on one machine with 8 NVIDIA P100 GPUs. "
        "For base models, each step took about 0.4 seconds. "
        "We trained the base models for 100,000 steps or 12 hours.",
        page=7,
    )
    quote = _support_quote(
        passage,
        "How long were the base models trained, and on what hardware?",
        "The base models trained for 12 hours on 8 NVIDIA P100 GPUs.",
    )
    assert "8 NVIDIA P100 GPUs" in quote and "12 hours" in quote


@pytest.mark.parametrize(
    "question,source,answer",
    [
        (
            "What BLEU did English-to-Spanish achieve?",
            "English-to-French achieved 41.0 BLEU.",
            "English-to-Spanish achieved 41.0 BLEU.",
        ),
        (
            "What percentage of ChatGPT performance was achieved on the HELM benchmark?",
            "On the Vicuna benchmark, Guanaco reached 99.3% of ChatGPT performance.",
            "Guanaco reached 99.3% of ChatGPT performance on Vicuna.",
        ),
    ],
)
def test_metric_cannot_be_moved_to_another_task_or_benchmark(question, source, answer):
    passage = chunk(1, source)
    with pytest.raises(ValueError, match="exact requested task or benchmark"):
        QAService._check_draft(answered(text=answer), [passage], question)


def test_explicit_non_evaluation_question_focuses_on_limitation_passage(settings):
    limitation = chunk(
        1,
        "We did not evaluate on other benchmarks such as BigBench, RAFT, and HELM.",
        section="Limitations",
    )
    distractor = chunk(2, "We evaluated MMLU, Vicuna, and OA benchmarks.")
    qa, _, _ = service(settings, [limitation, distractor], [answered()])
    current = state("Which benchmarks did the paper say it did not evaluate?")
    selected = qa.run_stage(Stage.RETRIEVE, current)["retrieved_chunks"]
    assert selected == [limitation]


@pytest.mark.parametrize(
    "question,source",
    [
        ("What BLEU did English-to-Spanish achieve?", "English-to-French achieved 41.0 BLEU."),
        (
            "What percentage of ChatGPT performance was achieved on the HELM benchmark?",
            "On Vicuna, the model reached 99.3% of ChatGPT performance.",
        ),
    ],
)
def test_missing_requested_metric_scope_abstains_before_model(settings, question, source):
    qa, _, generator = service(settings, [chunk(1, source)], [answered()])
    current = state(question)
    current.retrieved_chunks = [chunk(1, source)]
    answer = qa.run_stage(Stage.ANSWER, current)["answer"]
    assert answer.status == "insufficient_evidence"
    assert generator.calls == []


def test_explicit_unevaluated_benchmark_list_is_copied_and_cited(settings):
    passage = chunk(
        1,
        "We did not evaluate on other benchmarks such as BigBench, RAFT, and HELM, "
        "and it is not ensured that our evaluations generalize to these benchmarks.",
        section="Limitations",
        page=15,
    )
    qa, _, generator = service(settings, [passage], [answered()])
    current = state("Which three benchmarks did the paper say it did not evaluate?")
    current.retrieved_chunks = [passage]
    result = qa.run_stage(Stage.ANSWER, current)
    assert result["answer"].status == "answered"
    assert all(name in result["answer"].text for name in ["BigBench", "RAFT", "HELM"])
    assert result["answer"].citations[0].page == 15
    assert generator.calls == []


def test_residual_dropout_passage_is_prioritized_over_generic_matches(settings):
    direct = chunk(
        1,
        "Residual Dropout is applied to each sub-layer. For the base model, "
        "we use a rate of Pdrop = 0.1.",
        section="Regularization",
        page=7,
    )
    generic = chunk(2, "The base model uses dropout and many training rates.")
    qa, _, _ = service(settings, [generic, direct], [answered()])
    selected = qa.run_stage(
        Stage.RETRIEVE, state("What residual dropout rate is used for the base model?")
    )["retrieved_chunks"]
    assert selected[0] == direct


def test_double_quantization_target_is_copied_from_source(settings):
    direct = chunk(
        1,
        "Double Quantization reduces the average memory footprint by quantizing the "
        "quantization constants.",
    )
    qa, _, generator = service(settings, [direct], [answered()])
    current = state("What does QLoRA's Double Quantization quantize?")
    current.retrieved_chunks = [direct]
    answer = qa.run_stage(Stage.ANSWER, current)["answer"]
    assert "quantization constants" in answer.text
    assert answer.citations[0].chunk_id == direct.chunk_id
    assert generator.calls == []


def test_nf4_purpose_cites_its_direct_definition(settings):
    direct = chunk(
        1,
        "QLoRA introduces 4-bit NormalFloat (NF4), a new data type that is "
        "information theoretically optimal for normally distributed weights.",
        page=1,
    )
    qa, _, generator = service(settings, [direct], [answered()])
    current = state("What is the purpose of QLoRA's 4-bit NormalFloat (NF4) data type?")
    current.retrieved_chunks = [direct]
    result = qa.run_stage(Stage.ANSWER, current)
    assert "normally distributed weights" in result["answer"].text
    assert "information theoretically optimal" in result["answer_support_quotes"][0].source_quote
    assert generator.calls == []


def test_nf4_optimality_claim_requires_matching_quote():
    irrelevant = chunk(1, "The 4-bit NormalFloat data type reduces memory use.")
    with pytest.raises(ValueError, match="core claim"):
        QAService._check_draft(
            answered(
                text=(
                    "NF4 is information theoretically optimal for normally distributed "
                    "weights."
                )
            ),
            [irrelevant],
            "What is NF4 used for?",
        )


def test_counterfactual_replacement_cannot_borrow_original_bleu():
    passage = chunk(1, "The Transformer big model achieved 41.0 BLEU in translation.")
    with pytest.raises(ValueError, match="replace"):
        QAService._check_draft(
            answered(text="Transformer big achieved 41.0 BLEU."),
            [passage],
            "What BLEU did Transformer big achieve after replacing attention with LSTMs?",
        )


def test_dispensing_with_recurrence_supports_architecture_answer():
    passage = chunk(
        1,
        "The Transformer is based solely on attention mechanisms, dispensing with "
        "recurrence and convolutions entirely.",
    )
    answer, quotes = QAService._check_draft(
        answered(text="Attention mechanisms replace recurrence and convolutions."),
        [passage],
        "What mechanisms replace recurrence and convolutions?",
    )
    assert answer.status == "answered"
    assert "dispensing with" in quotes[0].source_quote


def test_evaluator_name_does_not_prove_beating_that_model():
    passage = chunk(
        1,
        "The Vicuna benchmark scores are percentages of ChatGPT evaluated by GPT-4. "
        "Guanaco has a 95% confidence interval of 4.4%.",
    )
    with pytest.raises(ValueError, match="does not report outperforming GPT-4"):
        QAService._check_draft(
            answered(text="Guanaco beat GPT-4 by 4.4 percentage points."),
            [passage],
            "By how many percentage points did Guanaco beat GPT-4 on the Vicuna benchmark?",
        )


def test_big_model_dropout_quote_cannot_support_base_model_answer():
    passage = chunk(
        1,
        "The Transformer big model used dropout rate Pdrop = 0.1, instead of 0.3.",
    )
    with pytest.raises(ValueError, match="base model result"):
        QAService._check_draft(
            answered(text="The Transformer base model used residual dropout rate 0.1."),
            [passage],
            "What residual dropout rate is used for the Transformer base model?",
        )


def test_dropout_quote_keeps_heading_and_rate_together():
    passage = chunk(
        1,
        "Residual Dropout We apply dropout to each sub-layer before normalization. "
        "It is also applied to positional encodings. "
        "For the base model, we use a rate of Pdrop = 0.1.",
        page=7,
    )
    quote = _support_quote(
        passage,
        "What residual dropout rate is used for the Transformer base model?",
        "The Transformer base model uses residual dropout rate 0.1.",
    )
    assert "Residual Dropout" in quote and "Pdrop = 0.1" in quote


def test_base_residual_dropout_uses_direct_passage_without_model_guess(settings):
    direct = chunk(
        1,
        "Residual Dropout We apply dropout to each sub-layer before normalization. "
        "It is also applied to positional encodings. "
        "For the base model, we use a rate of Pdrop = 0.1.",
        page=7,
    )
    qa, _, generator = service(settings, [direct], [answered()])
    current = state("What residual dropout rate is used for the Transformer base model?")
    current.retrieved_chunks = [direct]
    result = qa.run_stage(Stage.ANSWER, current)
    assert result["answer"].status == "answered"
    assert "0.1" in result["answer"].text
    assert "Residual Dropout" in result["answer_support_quotes"][0].source_quote
    assert generator.calls == []


def test_exact_subject_phrase_precedes_broad_dense_match(settings):
    broad = chunk(1, "Guanaco 65B can train on a GPU with 48GB of memory.")
    direct = chunk(2, "Guanaco 33B can train on 24 GB consumer GPUs in less than 12 hours.")
    qa, _, _ = service(settings, [broad, direct], [answered()])
    selected = qa.run_stage(
        Stage.RETRIEVE,
        state("What GPU memory and training time suffice for Guanaco 33B?"),
    )["retrieved_chunks"]
    assert selected[0] == direct


def test_answer_must_address_requested_model_size():
    wrong = chunk(1, "Guanaco 65B can train on a 48GB GPU. Guanaco 33B is also studied.")
    with pytest.raises(ValueError, match="model size named|omits a number"):
        QAService._check_draft(
            answered(text="Guanaco 65B uses a 48GB GPU."),
            [wrong],
            "What GPU memory suffices for Guanaco 33B?",
        )


def test_citation_pruning_cannot_drop_requested_model_size():
    other = chunk(1, "Guanaco 65B uses a 48GB GPU.")
    target = chunk(2, "Guanaco 33B is discussed separately.")
    with pytest.raises(ValueError, match="model size named|Final QA citations"):
        QAService._check_draft(
            AnswerDraft(
                status="answered",
                text="Guanaco 65B uses a 48GB GPU.",
                source_ids=["C1", "C2"],
            ),
            [other, target],
            "What GPU memory suffices for Guanaco 33B?",
        )


def test_model_size_memory_and_time_passage_is_prioritized(settings):
    mixed = chunk(1, "Guanaco 33B has 21 GB of weights; the largest model takes 24 hours.")
    direct = chunk(2, "Our 33B Guanaco can be trained on 24 GB GPUs in less than 12 hours.")
    qa, _, _ = service(settings, [mixed, direct], [answered()])
    selected = qa.run_stage(
        Stage.RETRIEVE,
        state("What GPU memory and training time suffice for Guanaco 33B?"),
    )["retrieved_chunks"]
    assert selected[0] == direct


def test_time_from_another_model_cannot_support_requested_size():
    generic_time = chunk(1, "The largest Guanaco model trains in 24 hours.")
    target_memory = chunk(2, "Guanaco 33B uses 24 GB of GPU memory.")
    with pytest.raises(ValueError, match="hardware or time fact"):
        QAService._check_draft(
            AnswerDraft(
                status="answered",
                text="Guanaco 33B uses 24 GB and trains in 24 hours.",
                source_ids=["C1", "C2"],
            ),
            [generic_time, target_memory],
            "What GPU memory and training time suffice for Guanaco 33B?",
        )


def test_direct_model_size_quote_discards_broad_extra_citation():
    direct = chunk(1, "Our 33B Guanaco can be trained on 24 GB GPUs in less than 12 hours.")
    broad = chunk(2, "The Guanaco family has a large model trained in 24 hours.")
    answer, quotes = QAService._check_draft(
        AnswerDraft(
            status="answered",
            text="Guanaco 33B trains on 24 GB GPUs in less than 12 hours.",
            source_ids=["C1", "C2"],
        ),
        [direct, broad],
        "What GPU memory and training time suffice for Guanaco 33B?",
    )
    assert [citation.chunk_id for citation in answer.citations] == [direct.chunk_id]
    assert len(quotes) == 1


def test_redundant_model_citations_are_removed():
    from arxiv_agent.contracts import Citation, QAQuote

    citations = [
        Citation(
            chunk_id=f"chunk-{i}", arxiv_id="0000.00000", version=1,
            page=i, section="Results",
        )
        for i in (1, 2, 3)
    ]
    quotes = [
        QAQuote(
            chunk_id="chunk-1",
            source_quote="For GPT-3, LoRA reduces trainable parameters by 10,000 times.",
        ),
        QAQuote(chunk_id="chunk-2", source_quote="GPT-3 experiments used AdamW."),
        QAQuote(chunk_id="chunk-3", source_quote="Other GPT-3 tasks were compared."),
    ]
    kept, support = _essential_citations(
        "For GPT-3, LoRA reduces trainable parameters by 10,000 times.",
        citations, quotes,
    )
    assert [item.chunk_id for item in kept] == ["chunk-1"]
    assert [item.chunk_id for item in support] == ["chunk-1"]


def test_truncated_model_json_repairs_then_abstains(settings):
    class TruncatedGenerator:
        calls = 0

        def generate(self, question, context_questions, passages, feedback=""):
            self.calls += 1
            return AnswerDraft.model_validate_json('{"status":"answered"')

    generator = TruncatedGenerator()
    qa = QAService(
        settings,
        store=Store([]),
        generator=generator,
        tokenizer=Tokenizer(),
    )
    current = state()
    current.retrieved_chunks = [
        chunk(1, "The method freezes its original weights and trains adapters.")
    ]
    assert qa.run_stage(Stage.ANSWER, current)["answer"].status == "insufficient_evidence"
    assert generator.calls == 2


def test_local_model_timeout_has_recovery_guidance(settings, monkeypatch):
    import ollama

    class BrokenClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            raise TimeoutError("model timed out")

    monkeypatch.setattr(ollama, "Client", BrokenClient)
    with pytest.raises(StageFailure, match="QA call failed") as exc:
        OllamaAnswerGenerator(settings).generate("Question?", [], [])
    assert exc.value.code == "MODEL_UNAVAILABLE"
    assert "Start Ollama" in exc.value.recovery


def test_saved_briefing_reopens_only_with_matching_evidence(settings):
    paper = synthetic_paper().model_copy(update={"arxiv_id": "2106.09685"})
    index = RetrievalIndex(collection="fake", fingerprint="f" * 64, chunk_count=1)
    note = EvidenceNote(
        facet="problem",
        text="The paper studies a costly training problem.",
        chunk_id="p",
        page=1,
        section="Introduction",
        source_block_ids=["b1"],
    )
    notes = [note]
    for facet, cid, text in [
        ("method", "m", "The method freezes weights during training."),
        ("result", "r", "The result improves accuracy on Dataset A."),
    ]:
        notes.append(note.model_copy(update={"facet": facet, "chunk_id": cid, "text": text}))
    evidence = EvidenceBundle(paper=paper, index=index, notes=notes, limitations_status="not_found")
    safe = "2106.09685v1-ffffffffffff"
    evidence_path = settings.data_dir / "evidence" / f"{safe}-e{EVIDENCE_VERSION}.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(evidence.model_dump_json())
    claim = EvidenceClaim(text="The paper studies a costly training problem.", chunk_ids=["p"])
    briefing = Briefing(
        paper=paper,
        plain_english_summary=claim,
        problem_statement=claim,
        method=[EvidenceClaim(text=notes[1].text, chunk_ids=["m"])],
        key_results=[EvidenceClaim(text=notes[2].text, chunk_ids=["r"])],
        limitations_status="not_found",
        limitations=[],
        limitations_note="None detected.",
        follow_up_questions=["Why?", "How?", "What result?"],
    )
    audits = [
        QuoteAudit(field=field, chunk_id=cid, page=1, section="Introduction", source_quote=text)
        for field, cid, text in [
            ("plain_english_summary", "p", note.text),
            ("problem_statement", "p", note.text),
            ("method", "m", notes[1].text),
            ("key_results", "r", notes[2].text),
        ]
    ]
    path = settings.output_dir / f"{safe}-b{BRIEFING_VERSION}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        BriefingArtifact(
            evidence_path=str(evidence_path.resolve()),
            briefing=briefing,
            quote_audit=audits,
        ).model_dump_json()
    )
    path.with_suffix(".md").write_text("# Sample\n")
    restored = load_qa_session(settings, "2106.09685v1")
    assert restored.status == "ready" and restored.briefing == briefing
    assert restored.index == index
    path.with_suffix(".md").unlink()
    with pytest.raises(StageFailure, match="identity does not match"):
        load_qa_session(settings, str(path))


def test_reopening_prefers_briefing_for_current_parsed_pdf(settings):
    safe_id = "2106.09685v1"
    pdf_path = settings.data_dir / "pdfs" / f"{safe_id}.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"fixture PDF checksum")
    checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    parsed = ParsedPaper(
        paper=synthetic_paper().model_copy(update={"arxiv_id": "2106.09685"}),
        pdf_checksum=checksum, page_count=1, pages_with_text=1,
        abstract="A paper abstract.", abstract_source="metadata",
        sections=[ParsedSection(title="Introduction", page_start=1, page_end=1)],
        blocks=[ParsedBlock(
            block_id="p1-b1", page=1, section="Introduction", kind="body",
            text="The method reports a measured result.", bbox=(72, 72, 300, 90),
        )],
    )
    parsed_path = settings.data_dir / "parsed" / f"{safe_id}-{checksum[:12]}.json"
    parsed_path.parent.mkdir(parents=True)
    parsed_path.write_text(parsed.model_dump_json())
    _, fingerprint = ChromaIndexStore(settings)._identity(parsed)
    current = settings.output_dir / f"{safe_id}-{fingerprint[:12]}-b{BRIEFING_VERSION}.json"
    stale = settings.output_dir / f"{safe_id}-000000000000-b{BRIEFING_VERSION}.json"
    assert _current_briefing_path(settings, safe_id, [stale, current]) == current.resolve()
