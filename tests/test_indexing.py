import re

import numpy as np
import pytest

from arxiv_agent.contracts import ParsedBlock, ParsedPaper, ParsedSection, RetrievalIndex
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.indexing import ChromaIndexStore, TokenAwareChunker
from arxiv_agent.services.synthetic import synthetic_paper


class WordTokenizer:
    is_fast = True

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        return 2

    @staticmethod
    def encode(text, add_special_tokens=True):
        return [0] * (len(list(re.finditer(r"\S+", text))) + 2 * add_special_tokens)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class CountingEncoder:
    calls = 0

    def encode(self, texts):
        self.calls += 1
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, content in enumerate(texts):
            for word in re.findall(r"[a-z]+", content.lower()):
                vectors[row, sum(word.encode()) % 384] += 1
                if word in {"distinctive", "adapter"}:
                    vectors[row, sum(word.encode()) % 384] += 20
        return vectors


def parsed():
    blocks = [
        ParsedBlock(
            block_id="p1-b1", page=1, section="Method", kind="body",
            text=" ".join(f"method{i}" for i in range(35)), bbox=(0, 0, 1, 1),
        ),
        ParsedBlock(
            block_id="p1-b2", page=1, section="Method", kind="caption",
            text="distinctive adapter result", bbox=(0, 1, 1, 2),
        ),
        ParsedBlock(
            block_id="p2-b1", page=2, section="Results", kind="body",
            text=" ".join(f"result{i}" for i in range(30)), bbox=(0, 0, 1, 1),
        ),
    ]
    return ParsedPaper(
        paper=synthetic_paper(), pdf_checksum="a" * 64, page_count=2,
        pages_with_text=2, abstract="A synthetic but nonempty abstract.",
        abstract_source="pdf", sections=[
            ParsedSection(title="Method", page_start=1, page_end=1),
            ParsedSection(title="Results", page_start=2, page_end=2),
        ], blocks=blocks,
    )


def test_chunk_limits_overlap_and_provenance(settings):
    settings = settings.model_copy(update={
        "chunk_target_tokens": 14, "chunk_max_tokens": 16, "chunk_overlap_tokens": 3,
    })
    chunks = TokenAwareChunker(settings, WordTokenizer()).chunk(parsed())
    assert len(chunks) > 3
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)
    assert all(chunk.embedding_tokens <= 16 for chunk in chunks)
    assert all(chunk.source_block_ids for chunk in chunks)
    assert all(chunk.section == ("Method" if chunk.page_start == 1 else "Results")
               for chunk in chunks)
    assert any("distinctive adapter result" in chunk.text for chunk in chunks)
    assert all(set(chunk.source_block_ids) <= {"p1-b1", "p1-b2", "p2-b1"}
               for chunk in chunks)
    first, second = chunks[:2]
    assert set(first.text.split()) & set(second.text.split())


def test_chroma_reopen_query_and_idempotent_build(settings):
    source = parsed()
    settings = settings.model_copy(update={
        "chunk_target_tokens": 14, "chunk_max_tokens": 16, "chunk_overlap_tokens": 3,
    })
    chunks = TokenAwareChunker(settings, WordTokenizer()).chunk(source)
    encoder = CountingEncoder()
    store = ChromaIndexStore(settings, encoder)
    index = store.build(source, chunks)
    assert index.chunk_count == len(chunks)
    calls = encoder.calls
    assert store.build(source, chunks) == index
    assert encoder.calls == calls
    reopened = ChromaIndexStore(settings, CountingEncoder())
    matches = reopened.query(index, "distinctive adapter result", limit=3)
    assert len(matches) == 3
    assert all(chunk.chunk_id in {item.chunk_id for item in chunks} for chunk, _ in matches)
    assert all(chunk.source_block_ids and chunk.page_start == chunk.page_end
               for chunk, _ in matches)
    assert any("distinctive adapter result" in chunk.text for chunk, _ in matches)
    assert all(distance >= 0 for _, distance in matches)
    bad = RetrievalIndex(
        collection=index.collection, fingerprint="wrong", chunk_count=index.chunk_count
    )
    with pytest.raises(StageFailure, match="another paper"):
        reopened.query(bad, "adapter")


def test_version_checksum_settings_and_text_change_index_identity(settings):
    source = parsed()
    base = ChromaIndexStore(settings)
    collection, fingerprint = base._identity(source)
    assert (collection, fingerprint) != base._identity(
        source.model_copy(update={"pdf_checksum": "b" * 64})
    )
    changed = source.model_copy(deep=True)
    changed.blocks[0].text = "Changed source text"
    assert (collection, fingerprint) != base._identity(changed)
    changed_settings = settings.model_copy(update={"chunk_overlap_tokens": 20})
    assert (collection, fingerprint) != ChromaIndexStore(changed_settings)._identity(source)


def test_invalid_embedding_shape_fails(settings):
    class BrokenEncoder:
        def encode(self, texts):
            return np.ones((len(texts), 3))

    source = parsed()
    chunks = TokenAwareChunker(settings, WordTokenizer()).chunk(source)
    with pytest.raises(StageFailure, match="384-dimensional"):
        ChromaIndexStore(settings, BrokenEncoder()).build(source, chunks)
