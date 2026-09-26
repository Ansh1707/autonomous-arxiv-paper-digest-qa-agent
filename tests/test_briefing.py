import pytest

from arxiv_agent.contracts import (
    EvidenceBundle,
    EvidenceNote,
    RetrievalIndex,
    SessionState,
    Stage,
)
from arxiv_agent.graph import build_briefing_graph
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.briefing import (
    BriefingArtifact,
    BriefingBuilder,
    BriefingDraft,
    DraftClaim,
    _direct_method_sentence,
    _direct_result_sentence,
    _focused_problem_passage,
    _table_result_sentence,
)
from arxiv_agent.services.evidence import EVIDENCE_VERSION
from arxiv_agent.services.synthetic import SyntheticServices, synthetic_paper


def evidence(limitation=True):
    records = [
        ("problem", 1, "Problem", "Full tuning needs a costly model copy for every task."),
        ("method", 2, "Method", "The method freezes base weights and trains small adapters."),
        ("result", 3, "Results", "The study reports 91% accuracy with fewer parameters."),
    ]
    if limitation:
        records.append(
            (
                "limitation",
                4,
                "Discussion",
                "The method cannot batch different tasks in a single forward pass.",
            )
        )
    notes = [
        EvidenceNote(
            facet=facet,
            text=text,
            chunk_id=f"c{number}",
            page=page,
            section=section,
            source_block_ids=[f"p{page}-b1"],
        )
        for number, (facet, page, section, text) in enumerate(records)
    ]
    return EvidenceBundle(
        paper=synthetic_paper(),
        index=RetrievalIndex(collection="fixture", fingerprint="f" * 64, chunk_count=4),
        notes=notes,
        limitations_status="reported" if limitation else "not_found",
    )


def test_conditional_comparison_cannot_become_reported_result():
    source = (
        "The relative score can be higher than 100% if the model achieves a "
        "higher absolute score than ChatGPT."
    )
    claim = DraftClaim(
        text="The model scores higher than ChatGPT.",
        chunk_id="c2",
        source_quote=source,
    )
    note = EvidenceNote(
        facet="result", text=source, chunk_id="c2", page=9,
        section="Evaluation", source_block_ids=["b1"],
    )
    with pytest.raises(ValueError, match="conditional comparison"):
        BriefingBuilder._validate_claim("key_results", claim, {"c2"}, {"c2": note})


def test_direct_result_uses_observed_effect_instead_of_hypothetical_comparison():
    source = (
        "The relative score can be higher than 100% if the model achieves a higher "
        "score than ChatGPT. We find a significant ordering effect with GPT-4 increasing "
        "the score of the response occurring earlier in the prompt."
    )
    assert _direct_result_sentence(source) == (
        "We find a significant ordering effect with GPT-4 increasing the score of the "
        "response occurring earlier in the prompt."
    )


def draft(limitation=True, *, bad_number=False):
    problem = DraftClaim(
        text="Full tuning needs a separate costly copy for every task.",
        chunk_id="c0",
        source_quote="Full tuning needs a costly model copy for every task.",
    )
    method = DraftClaim(
        text="The method freezes base weights and trains small adapters.",
        chunk_id="c1",
        source_quote="freezes base weights and trains small adapters",
    )
    result = DraftClaim(
        text=(
            "The study reports 92% accuracy."
            if bad_number
            else "The study reports 91% accuracy with fewer parameters."
        ),
        chunk_id="c2",
        source_quote="reports 91% accuracy with fewer parameters",
    )
    limitations = (
        [
            DraftClaim(
                text="Different tasks cannot be batched in one forward pass.",
                chunk_id="c3",
                source_quote="cannot batch different tasks in a single forward pass",
            )
        ]
        if limitation
        else []
    )
    return BriefingDraft(
        plain_english_summary=problem,
        problem_statement=problem,
        method=[method],
        key_results=[result],
        limitations=limitations,
        follow_up_questions=[
            "What problem does the paper address?",
            "How does the adapter reduce training cost?",
            "How was the result measured?",
        ],
    )


def prepared_state(settings, bundle):
    path = (
        settings.data_dir
        / "evidence"
        / f"0000.00000v1-{bundle.index.fingerprint[:12]}-e{EVIDENCE_VERSION}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return SessionState(
        user_input="0000.00000v1",
        selected_paper=bundle.paper,
        index=bundle.index,
        evidence_path=str(path),
        evidence_notes=bundle.notes,
        limitations_evidence_status=bundle.limitations_status,
    )


class FakeGenerator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def generate(self, evidence, feedback=""):
        self.calls += 1
        return self.result


@pytest.mark.parametrize("limitation", [True, False])
def test_saves_grounded_briefing_and_page_citations(settings, limitation):
    bundle = evidence(limitation)
    generator = FakeGenerator(draft(limitation))
    briefing, paths = BriefingBuilder(settings, generator).build(prepared_state(settings, bundle))
    assert briefing.paper == bundle.paper
    assert briefing.limitations_status == ("reported" if limitation else "not_found")
    assert len(briefing.follow_up_questions) == 3
    assert bool(briefing.limitations) == limitation
    assert all(
        claim.chunk_ids
        for claim in [
            briefing.plain_english_summary,
            briefing.problem_statement,
            *briefing.method,
            *briefing.key_results,
            *briefing.limitations,
        ]
    )
    assert len(paths) == 2
    saved = BriefingArtifact.model_validate_json(open(paths[0], encoding="utf-8").read())
    assert saved.briefing == briefing
    assert all(audit.source_quote for audit in saved.quote_audit)
    markdown = open(paths[1], encoding="utf-8").read()
    assert "## Why this paper matters" in markdown
    assert "## Follow-up questions" in markdown
    assert markdown.count("?") == 3
    assert "#page=3" in markdown
    assert ("No explicit limitation was detected" in markdown) != limitation
    assert generator.calls == 1


def test_rejects_unsupported_number_without_saving(settings):
    bundle = evidence()
    generator = FakeGenerator(draft(bad_number=True))
    with pytest.raises(StageFailure, match="source-linked briefing"):
        BriefingBuilder(settings, generator).build(prepared_state(settings, bundle))
    assert generator.calls == 2
    assert list(settings.output_dir.glob("*")) == []


def test_rejects_wrong_quote_or_facet(settings):
    bundle = evidence()
    builder = BriefingBuilder(settings, FakeGenerator(draft()))
    wrong_quote = draft().model_copy(deep=True)
    wrong_quote.method[0].source_quote = "This quote is not present in the paper."
    with pytest.raises(ValueError, match="not copied"):
        builder._assemble(bundle, wrong_quote)
    wrong_facet = draft().model_copy(deep=True)
    wrong_facet.method[0].chunk_id = "c2"
    with pytest.raises(ValueError, match="outside its evidence facet"):
        builder._assemble(bundle, wrong_facet)


def test_dense_table_uses_only_an_explicit_comparison_sentence():
    text = (
        "(LoRA) 4.7M 73.4 91.3 52.1/28.3/44.3. "
        "Table 1: Validation results. LoRA outperforms several baselines "
        "with fewer trainable parameters."
    )
    assert _table_result_sentence(text) == (
        "LoRA outperforms several baselines with fewer trainable parameters."
    )


def test_direct_result_preserves_single_model_qualifier_after_flattened_table():
    source = (
        "27.3 38.1 3.3 · 1018 Transformer (big) 28.4 41.0 2.3 · 1019 "
        "On the WMT 2014 English-to-French translation task, our big model "
        "achieves a BLEU score of 41.0, outperforming all of the previously "
        "published single models, at less than 1/4 the training cost of the "
        "previous state-of-the-art model. The next sentence is unrelated."
    )
    result = _direct_result_sentence(source)
    assert result.startswith("On the WMT 2014 English-to-French")
    assert "single models" in result
    assert "all previous models" not in result


def test_processing_caveats_are_separate_from_paper_limitations(settings):
    bundle = evidence(limitation=False)
    builder = BriefingBuilder(settings, FakeGenerator(draft(False)))
    briefing, _ = builder._assemble(
        bundle,
        draft(False),
        [
            "Parser skipped a figure caption.",
            "No explicit limitation passage was detected; report not_found.",
        ],
    )
    assert briefing.limitations_status == "not_found"
    assert briefing.processing_notes == ["Parser skipped a figure caption."]
    assert "Parser" not in briefing.limitations_note


def test_complete_method_sentence_is_reused_verbatim():
    source = (
        "We describe the approach. During training, W0 is frozen while A and B "
        "contain trainable parameters. The next sentence is unrelated."
    )
    assert _direct_method_sentence(source) == (
        "During training, W0 is frozen while A and B contain trainable parameters."
    )


def test_problem_prompt_excludes_unrelated_prior_work():
    source = (
        "Recurrent models are common. Their sequential nature precludes parallelization. "
        "Prior work used factorization tricks to improve efficiency."
    )
    assert _focused_problem_passage(source).startswith(
        "Their sequential nature precludes parallelization."
    )


def test_missing_or_altered_evidence_fails_before_model(settings):
    bundle = evidence()
    generator = FakeGenerator(draft())
    builder = BriefingBuilder(settings, generator)
    state = prepared_state(settings, bundle)
    state.evidence_notes = bundle.notes[:-1]
    with pytest.raises(StageFailure, match="differs"):
        builder.build(state)
    assert generator.calls == 0


def test_briefing_graph_stops_before_qa(settings):
    result = SessionState.model_validate(
        build_briefing_graph(SyntheticServices("lookup"), settings).invoke(
            SessionState(user_input="synthetic input", execution_mode="synthetic"),
            config={"recursion_limit": settings.graph_recursion_limit},
        )
    )
    assert result.error is None
    assert result.stage_history[-1] == Stage.BRIEF
    assert result.briefing is not None
    assert Stage.READY not in result.stage_history
