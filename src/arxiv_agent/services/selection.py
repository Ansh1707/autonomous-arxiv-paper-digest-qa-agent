"""Local MiniLM ranking and explicit paper selection after arXiv discovery."""

import logging
import math
from typing import Protocol

import arxiv
import numpy as np
from huggingface_hub import snapshot_download

from arxiv_agent.contracts import Candidate, Intent, SessionState, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.discovery import ArxivDiscoveryServices
from arxiv_agent.services.input_understanding import InputUnderstandingServices
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalMiniLMEncoder:
    """Load the pinned, already-downloaded model on demand; never download during selection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            try:
                path = snapshot_download(
                    self.settings.embedding_model,
                    revision=self.settings.embedding_revision,
                    cache_dir=str(self.settings.model_cache_dir.resolve()),
                    local_files_only=True,
                )
                torch.set_num_threads(2)
                self._model = SentenceTransformer(path, device="cpu", local_files_only=True)
            except (OSError, ValueError) as exc:
                raise StageFailure(
                    "EMBEDDING_MODEL_MISSING",
                    "The local MiniLM embedding model is unavailable or incomplete.",
                    "Run 'python -m arxiv_agent doctor --download-models' "
                    "once with internet access.",
                ) from exc
        return self._model.encode(
            texts, batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )


class EmbeddingRanker:
    """Score title and abstract independently to keep the title from being truncated away."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def rank(self, intent: Intent, candidates: list[Candidate]) -> list[Candidate]:
        if intent.kind != "topic" or not intent.search_terms:
            raise StageFailure("INVALID_INTENT", "Ranking needs topic terms.", "Supply a topic.")
        if not candidates:
            raise StageFailure("NO_CANDIDATES", "No papers to rank.", "Refine the topic.")
        texts = [" ".join(intent.search_terms)]
        texts.extend(candidate.paper.title for candidate in candidates)
        texts.extend(candidate.paper.abstract for candidate in candidates)
        try:
            vectors = np.asarray(self.encoder.encode(texts), dtype=np.float64)
        except StageFailure:
            raise
        except Exception as exc:
            logger.exception("MiniLM candidate ranking failed")
            raise StageFailure(
                "EMBEDDING_FAILED", "Could not embed the topic and candidate papers.",
                "Check the local MiniLM cache and available memory, then retry.",
            ) from exc
        count = len(candidates)
        if vectors.ndim != 2 or vectors.shape[0] != 1 + count * 2 or vectors.shape[1] == 0:
            raise StageFailure(
                "INVALID_EMBEDDINGS", "MiniLM returned an unexpected embedding shape.",
                "Check the local embedding model installation.",
            )
        if not np.isfinite(vectors).all():
            raise StageFailure(
                "INVALID_EMBEDDINGS", "MiniLM returned a non-finite embedding.",
                "Retry after checking the local embedding model.",
            )
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise StageFailure(
                "INVALID_EMBEDDINGS", "MiniLM returned an empty embedding.",
                "Retry after checking the local embedding model.",
            )
        unit = vectors / norms[:, None]
        ranked = []
        for index, candidate in enumerate(candidates):
            title_similarity = float(np.dot(unit[0], unit[1 + index]))
            abstract_similarity = float(np.dot(unit[0], unit[1 + count + index]))
            score = 0.6 * title_similarity + 0.4 * abstract_similarity
            if not math.isfinite(score):
                raise StageFailure(
                    "INVALID_EMBEDDINGS", "Could not calculate a finite relevance score.",
                    "Retry after checking the local embedding model.",
                )
            scored = candidate.model_copy(update={"score": round(max(-1, min(1, score)), 6)})
            ranked.append((index, scored))
        ranked.sort(key=lambda item: (-item[1].score, item[0]))
        return [candidate for _, candidate in ranked]


class ArxivSelectionServices(ArxivDiscoveryServices):
    def __init__(
        self,
        settings: Settings,
        understanding: InputUnderstandingServices,
        *,
        selection_rank: int = 1,
        ranker: EmbeddingRanker | None = None,
        client: arxiv.Client | None = None,
    ):
        super().__init__(settings, understanding, client)
        self.selection_rank = selection_rank
        self.ranker = ranker or EmbeddingRanker(LocalMiniLMEncoder(settings))

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage == Stage.RANK:
            ranked = self.ranker.rank(state.intent, state.candidates)
            if self.selection_rank < 1 or self.selection_rank > len(ranked):
                raise StageFailure(
                    "INVALID_SELECTION",
                    f"Selection {self.selection_rank} is outside the "
                    f"1–{len(ranked)} ranked papers.",
                    "Choose a rank shown in the candidate list.",
                )
            return {
                "candidates": ranked,
                "selected_paper": ranked[self.selection_rank - 1].paper,
                "selection_rank": self.selection_rank,
                "selection_source": "top_ranked" if self.selection_rank == 1 else "override",
            }
        if stage == Stage.LOOKUP and self.selection_rank != 1:
            raise StageFailure(
                "INVALID_SELECTION", "An arXiv ID resolves to one paper, so only rank 1 is valid.",
                "Omit --select or use --select 1.",
            )
        patch = super().run_stage(stage, state)
        if stage == Stage.LOOKUP:
            patch.update(selection_rank=1, selection_source="lookup")
        return patch
