"""Page-bounded MiniLM chunks and a reusable, versioned Chroma index."""

import hashlib
import json
import logging
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from huggingface_hub import snapshot_download
from pydantic import ValidationError
from transformers import AutoTokenizer

from arxiv_agent.contracts import Chunk, ParsedPaper, RetrievalIndex, SessionState, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import InputUnderstandingServices
from arxiv_agent.services.pdf_parse import ArxivParseServices
from arxiv_agent.services.selection import Encoder, LocalMiniLMEncoder
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)
CHUNKER_VERSION = 2


class TokenAwareChunker:
    def __init__(self, settings: Settings, tokenizer=None):
        self.settings = settings
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                path = snapshot_download(
                    self.settings.embedding_model,
                    revision=self.settings.embedding_revision,
                    cache_dir=str(self.settings.model_cache_dir.resolve()),
                    local_files_only=True,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    path, use_fast=True, local_files_only=True
                )
                # Full page/section groups are tokenized only to locate bounded windows.
                self._tokenizer.model_max_length = 1_000_000
            except (OSError, ValueError) as exc:
                raise StageFailure(
                    "EMBEDDING_MODEL_MISSING", "The pinned local MiniLM tokenizer is unavailable.",
                    "Run 'python -m arxiv_agent doctor --download-models' "
                    "once with internet access.",
                ) from exc
        if not self._tokenizer.is_fast:
            raise StageFailure(
                "TOKENIZER_UNSUPPORTED", "MiniLM needs a fast tokenizer for source offsets.",
                "Check the pinned MiniLM model cache.",
            )
        return self._tokenizer

    def chunk(self, parsed: ParsedPaper) -> list[Chunk]:
        tokenizer = self.tokenizer
        special = tokenizer.num_special_tokens_to_add(pair=False)
        budget = self.settings.chunk_target_tokens - special
        if budget < 1 or self.settings.chunk_overlap_tokens >= budget:
            raise StageFailure(
                "INVALID_CHUNK_LIMIT", "Chunk target leaves no room for content and overlap.",
                "Increase ARXIV_AGENT_CHUNK_TARGET_TOKENS or reduce overlap.",
            )
        groups: list[tuple[tuple[int, str], list]] = []
        for block in parsed.blocks:
            key = (block.page, block.section)
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
            groups[-1][1].append(block)
        result = []
        for group_number, ((page, section), blocks) in enumerate(groups):
            content = "\n".join(block.text for block in blocks)
            spans = []
            offset = 0
            for block in blocks:
                spans.append((offset, offset + len(block.text), block.block_id))
                offset += len(block.text) + 1
            encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            start = 0
            ordinal = 0
            while start < len(offsets):
                end = min(start + budget, len(offsets))
                # Word boundaries avoid embedding a detached WordPiece at the edge.
                first = offsets[start][0]
                while first > 0 and not content[first - 1].isspace():
                    first -= 1
                last = offsets[end - 1][1]
                if end < len(offsets):
                    while last > first and not content[last - 1].isspace():
                        last -= 1
                    if last <= first:
                        last = offsets[end - 1][1]
                chunk_text = content[first:last].strip()
                tokens = len(tokenizer.encode(chunk_text, add_special_tokens=True))
                while tokens > self.settings.chunk_max_tokens and end > start + 1:
                    end -= 1
                    last = offsets[end - 1][1]
                    chunk_text = content[first:last].strip()
                    tokens = len(tokenizer.encode(chunk_text, add_special_tokens=True))
                if not chunk_text or tokens > self.settings.chunk_max_tokens:
                    raise StageFailure(
                        "CHUNKING_FAILED", "A source span exceeds the MiniLM token ceiling.",
                        "Inspect the parsed text or increase the configured ceiling.",
                    )
                source_ids = [
                    block_id for lower, upper, block_id in spans
                    if lower < last and upper > first
                ]
                if not source_ids:
                    raise StageFailure(
                        "CHUNKING_FAILED", "Chunk lost its PDF source block.",
                        "Inspect the parsed block offsets.",
                    )
                digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
                section_digest = hashlib.sha256(section.encode("utf-8")).hexdigest()[:8]
                chunk_id = (
                    f"{parsed.paper.arxiv_id}v{parsed.paper.version}"
                    f"-{parsed.pdf_checksum[:12]}-p{page}-s{section_digest}"
                    f"-g{group_number}-c{ordinal}-{digest}"
                )
                result.append(Chunk(
                    chunk_id=chunk_id, arxiv_id=parsed.paper.arxiv_id,
                    version=parsed.paper.version, section=section,
                    page_start=page, page_end=page, text=chunk_text,
                    embedding_tokens=tokens, source_block_ids=source_ids,
                ))
                ordinal += 1
                if end == len(offsets):
                    break
                start = max(start + 1, end - self.settings.chunk_overlap_tokens)
        if not result:
            raise StageFailure(
                "NO_CHUNKS", "No text chunks were recovered from the parsed PDF.",
                "Inspect the parsed paper artifact.",
            )
        return result


class ChromaIndexStore:
    def __init__(self, settings: Settings, encoder: Encoder | None = None):
        self.settings = settings
        self.encoder = encoder or LocalMiniLMEncoder(settings)

    def _client(self):
        path = self.settings.data_dir / "vectors"
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(path.resolve()),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _collection(self, index: RetrievalIndex):
        collection = self._client().get_collection(index.collection)
        if (
            collection.metadata.get("fingerprint") != index.fingerprint
            or collection.metadata.get("embedding_model") != self.settings.embedding_model
            or collection.metadata.get("embedding_revision") != self.settings.embedding_revision
            or collection.count() != index.chunk_count
        ):
            raise StageFailure(
                "INDEX_MISMATCH", "Stored index is incomplete or belongs to another paper.",
                "Rebuild the selected paper index.",
            )
        return collection

    def validate_index(self, index: RetrievalIndex) -> None:
        """Check a saved index without loading the encoder or embedding anything."""
        try:
            self._collection(index)
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure(
                "INDEX_MISSING", "The saved paper index is unavailable.",
                "Rebuild the selected paper index, then retry reopening the session.",
            ) from exc

    @staticmethod
    def _chunk(chunk_id: str, content: str, meta: dict) -> Chunk:
        return Chunk(
            chunk_id=chunk_id, arxiv_id=meta["arxiv_id"],
            version=meta["version"], section=meta["section"],
            page_start=meta["page_start"], page_end=meta["page_end"],
            text=content, embedding_tokens=meta["embedding_tokens"],
            source_block_ids=json.loads(meta["source_block_ids"]),
        )

    def _identity(self, parsed: ParsedPaper) -> tuple[str, str]:
        identity = {
            "arxiv_id": parsed.paper.arxiv_id, "version": parsed.paper.version,
            "pdf_checksum": parsed.pdf_checksum, "parsed_schema": parsed.schema_version,
            "parsed_text_checksum": hashlib.sha256(
                json.dumps(
                    [block.model_dump() for block in parsed.blocks], sort_keys=True
                ).encode("utf-8")
            ).hexdigest(),
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.settings.embedding_revision,
            "chunker_version": CHUNKER_VERSION,
            "target": self.settings.chunk_target_tokens,
            "maximum": self.settings.chunk_max_tokens,
            "overlap": self.settings.chunk_overlap_tokens,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return "paper-" + fingerprint[:40], fingerprint

    @staticmethod
    def _vectors(encoder: Encoder, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(encoder.encode(texts), dtype=np.float32)
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise StageFailure(
                "INVALID_EMBEDDINGS", "MiniLM did not return finite 384-dimensional vectors.",
                "Check the pinned embedding model and retry.",
            )
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise StageFailure(
                "INVALID_EMBEDDINGS", "MiniLM returned an empty vector.",
                "Check the embedding model and retry.",
            )
        return vectors / norms[:, None]

    def build(self, parsed: ParsedPaper, chunks: list[Chunk]) -> RetrievalIndex:
        collection_name, fingerprint = self._identity(parsed)
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise StageFailure(
                "DUPLICATE_CHUNKS", "Chunk IDs are not unique.",
                "Inspect the parsed paper and chunker.",
            )
        try:
            collection = self._client().get_or_create_collection(
                name=collection_name,
                metadata={
                    "fingerprint": fingerprint, "hnsw:space": "cosine",
                    "embedding_model": self.settings.embedding_model,
                    "embedding_revision": self.settings.embedding_revision,
                },
            )
            if collection.metadata.get("fingerprint") != fingerprint:
                raise StageFailure(
                    "INDEX_MISMATCH", "Stored index identity differs from this paper.",
                    "Inspect or remove the mismatched collection before retrying.",
                )
            existing = set(collection.get(include=[])["ids"])
            expected = set(ids)
            if existing != expected:
                if existing - expected:
                    collection.delete(ids=list(existing - expected))
                missing = [chunk for chunk in chunks if chunk.chunk_id not in existing]
                for offset in range(0, len(missing), self.settings.embedding_batch_size):
                    batch = missing[offset:offset + self.settings.embedding_batch_size]
                    vectors = self._vectors(self.encoder, [chunk.text for chunk in batch])
                    collection.upsert(
                        ids=[chunk.chunk_id for chunk in batch],
                        documents=[chunk.text for chunk in batch],
                        embeddings=vectors.tolist(),
                        metadatas=[{
                            "arxiv_id": chunk.arxiv_id, "version": chunk.version,
                            "section": chunk.section, "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "embedding_tokens": chunk.embedding_tokens,
                            "source_block_ids": json.dumps(chunk.source_block_ids),
                            "pdf_checksum": parsed.pdf_checksum,
                        } for chunk in batch],
                    )
            if set(collection.get(include=[])["ids"]) != expected:
                raise StageFailure(
                    "INDEX_INCOMPLETE", "Chroma did not retain every paper chunk.",
                    "Retry indexing after checking disk space.",
                )
            return RetrievalIndex(
                collection=collection_name, fingerprint=fingerprint, chunk_count=len(chunks)
            )
        except StageFailure:
            raise
        except Exception as exc:
            logger.exception("Chroma indexing failed")
            raise StageFailure(
                "INDEX_STORAGE_ERROR", "Could not persist the paper vector index.",
                "Check disk space and data/vectors permissions, then retry.",
            ) from exc

    def query(
        self, index: RetrievalIndex, question: str, limit: int = 3
    ) -> list[tuple[Chunk, float]]:
        if not question.strip() or limit < 1:
            raise ValueError("A nonempty question and positive result limit are required")
        try:
            collection = self._collection(index)
            vector = self._vectors(self.encoder, [question])[0]
            rows = collection.query(
                query_embeddings=[vector.tolist()], n_results=min(limit, index.chunk_count),
                include=["documents", "metadatas", "distances"],
            )
            results = []
            for chunk_id, content, meta, distance in zip(
                rows["ids"][0], rows["documents"][0], rows["metadatas"][0],
                rows["distances"][0], strict=True,
            ):
                results.append((self._chunk(chunk_id, content, meta), float(distance)))
            return results
        except StageFailure:
            raise
        except Exception as exc:
            logger.exception("Chroma query failed")
            raise StageFailure(
                "INDEX_QUERY_ERROR", "Could not query the stored paper index.",
                "Check the local MiniLM cache and Chroma storage, then retry.",
            ) from exc

    def all_chunks(self, index: RetrievalIndex) -> list[Chunk]:
        """Read stored source text without loading vectors into memory."""
        try:
            rows = self._collection(index).get(include=["documents", "metadatas"])
            return [
                self._chunk(chunk_id, content, meta)
                for chunk_id, content, meta in zip(
                    rows["ids"], rows["documents"], rows["metadatas"], strict=True
                )
            ]
        except StageFailure:
            raise
        except Exception as exc:
            logger.exception("Chroma read failed")
            raise StageFailure(
                "INDEX_READ_ERROR", "Could not read indexed paper chunks.",
                "Check Chroma storage and rebuild the selected paper index.",
            ) from exc


class ArxivIndexServices:
    def __init__(
        self, settings: Settings, understanding: InputUnderstandingServices, *,
        selection_rank: int = 1, chunker: TokenAwareChunker | None = None,
        store: ChromaIndexStore | None = None, **kwargs,
    ):
        self.parsing = ArxivParseServices(
            settings, understanding, selection_rank=selection_rank, **kwargs
        )
        self.chunker = chunker or TokenAwareChunker(settings)
        self.store = store or ChromaIndexStore(settings)

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage != Stage.INDEX:
            return self.parsing.run_stage(stage, state)
        if not state.parsed_path or not state.selected_paper or not state.pdf_checksum:
            raise StageFailure(
                "MISSING_PARSED_PAPER", "Indexing requires a parsed paper and PDF checksum.",
                "Run parse-paper before indexing.",
            )
        expected = (
            self.chunker.settings.data_dir / "parsed" /
            f"{state.selected_paper.arxiv_id}v{state.selected_paper.version}"
            .replace("/", "_")
        )
        expected = expected.parent / f"{expected.name}-{state.pdf_checksum[:12]}.json"
        path = Path(state.parsed_path).resolve()
        if path != expected.resolve():
            raise StageFailure(
                "PARSED_PAPER_MISMATCH", "Parsed path does not match the selected paper.",
                "Rerun parse-paper for this exact version.",
            )
        try:
            parsed = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            raise StageFailure(
                "PARSED_PAPER_INVALID", "Parsed paper artifact is missing or invalid.",
                "Rerun parse-paper.",
            ) from exc
        if parsed.paper != state.selected_paper or parsed.pdf_checksum != state.pdf_checksum:
            raise StageFailure(
                "PARSED_PAPER_MISMATCH", "Parsed paper differs from the selected PDF.",
                "Rerun fetch-paper and parse-paper.",
            )
        return {"index": self.store.build(parsed, self.chunker.chunk(parsed))}
