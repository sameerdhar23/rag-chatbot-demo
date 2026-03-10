# tests/test_rag.py
"""Unit tests for app/rag.py RAG logic."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag import (
    _build_context_block,
    _doc_at,
    _extract_chunk_id,
    _extract_page,
    _extract_text,
    _make_prompt,
    answer_question,
    retrieve,
    select_contexts,
)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_uses_text_key(self):
        assert _extract_text({"text": "hello"}) == "hello"

    def test_uses_chunk_key(self):
        assert _extract_text({"chunk": "chunk content"}) == "chunk content"

    def test_uses_content_key(self):
        assert _extract_text({"content": "content value"}) == "content value"

    def test_uses_page_content_key(self):
        assert _extract_text({"page_content": "page content"}) == "page content"

    def test_prefers_text_over_others(self):
        d = {"text": "t", "chunk": "c", "content": "cn", "page_content": "pc"}
        assert _extract_text(d) == "t"

    def test_fallback_stringify(self):
        d = {"other_key": "value"}
        result = _extract_text(d)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_non_string_value_skipped(self):
        d = {"text": 123, "chunk": "good"}
        assert _extract_text(d) == "good"


# ---------------------------------------------------------------------------
# _extract_page
# ---------------------------------------------------------------------------

class TestExtractPage:
    def test_page_key(self):
        assert _extract_page({"page": 3}) == 3

    def test_page_num_key(self):
        assert _extract_page({"page_num": "7"}) == 7

    def test_pageno_key(self):
        assert _extract_page({"pageno": 12}) == 12

    def test_returns_none_when_no_page_key(self):
        assert _extract_page({"text": "hi"}) is None

    def test_returns_none_on_unconvertible_value(self):
        assert _extract_page({"page": "not-a-number"}) is None

    def test_page_string_converted_to_int(self):
        assert _extract_page({"page": "5"}) == 5


# ---------------------------------------------------------------------------
# _extract_chunk_id
# ---------------------------------------------------------------------------

class TestExtractChunkId:
    def test_chunk_id_key(self):
        assert _extract_chunk_id({"chunk_id": "abc"}) == "abc"

    def test_id_key(self):
        assert _extract_chunk_id({"id": "xyz"}) == "xyz"

    def test_chunkId_key(self):
        assert _extract_chunk_id({"chunkId": "cid"}) == "cid"

    def test_returns_empty_string_when_missing(self):
        assert _extract_chunk_id({"text": "no id here"}) == ""

    def test_non_string_value_skipped(self):
        assert _extract_chunk_id({"chunk_id": 99, "id": "fallback"}) == "fallback"


# ---------------------------------------------------------------------------
# _doc_at
# ---------------------------------------------------------------------------

class TestDocAt:
    def test_list_structure(self):
        docs = [{"text": "a"}, {"text": "b"}]
        assert _doc_at(docs, 1) == {"text": "b"}

    def test_dict_with_int_key(self):
        docs = {0: {"text": "zero"}, 1: {"text": "one"}}
        assert _doc_at(docs, 0) == {"text": "zero"}

    def test_dict_with_str_key(self):
        docs = {"0": {"text": "zero"}, "1": {"text": "one"}}
        assert _doc_at(docs, 0) == {"text": "zero"}

    def test_unsupported_structure_raises_type_error(self):
        with pytest.raises(TypeError):
            _doc_at("not a list or dict", 0)


# ---------------------------------------------------------------------------
# select_contexts
# ---------------------------------------------------------------------------

class TestSelectContexts:
    def _make_hits(self, scores: List[float]) -> List[Dict[str, Any]]:
        return [{"score": s, "text": "t", "faiss_id": i} for i, s in enumerate(scores)]

    def test_filters_below_min_score(self):
        hits = self._make_hits([0.9, 0.4, 0.7])
        result = select_contexts(hits, max_ctx=5, min_score=0.6)
        scores = [h["score"] for h in result]
        assert all(s >= 0.6 for s in scores)
        assert len(result) == 2

    def test_limits_by_max_ctx(self):
        hits = self._make_hits([0.9, 0.8, 0.7, 0.6])
        result = select_contexts(hits, max_ctx=2, min_score=0.0)
        assert len(result) == 2

    def test_returns_empty_when_all_below_threshold(self):
        hits = self._make_hits([0.3, 0.2])
        result = select_contexts(hits, max_ctx=3, min_score=0.5)
        assert result == []

    def test_preserves_order(self):
        hits = self._make_hits([0.9, 0.8, 0.7])
        result = select_contexts(hits, max_ctx=3, min_score=0.0)
        assert [h["score"] for h in result] == [0.9, 0.8, 0.7]


# ---------------------------------------------------------------------------
# _build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:
    def _ctx(self, text: str, page=None, chunk_id="", faiss_id=0) -> Dict[str, Any]:
        return {"text": text, "page": page, "chunk_id": chunk_id, "faiss_id": faiss_id}

    def test_includes_text(self):
        block = _build_context_block([self._ctx("hello world")])
        assert "hello world" in block

    def test_includes_page_in_header(self):
        block = _build_context_block([self._ctx("text", page=5)])
        assert "Page 5" in block

    def test_includes_chunk_id_in_header(self):
        block = _build_context_block([self._ctx("text", chunk_id="c42")])
        assert "c42" in block

    def test_falls_back_to_faiss_id_header(self):
        block = _build_context_block([self._ctx("text", faiss_id=7)])
        assert "Chunk 7" in block

    def test_respects_max_chars(self):
        large_text = "x" * 500
        contexts = [self._ctx(large_text, faiss_id=i) for i in range(10)]
        block = _build_context_block(contexts, max_chars=100)
        assert len(block) <= 100 + 50  # a bit of header overhead is fine

    def test_empty_contexts_returns_empty_string(self):
        assert _build_context_block([]) == ""


# ---------------------------------------------------------------------------
# _make_prompt
# ---------------------------------------------------------------------------

class TestMakePrompt:
    def _ctx(self, text: str) -> Dict[str, Any]:
        return {"text": text, "page": 1, "chunk_id": "c1", "faiss_id": 0}

    def test_contains_system_instructions(self):
        prompt = _make_prompt("Q?", [self._ctx("some context")])
        assert "ONLY the provided context" in prompt

    def test_contains_question(self):
        prompt = _make_prompt("What is X?", [self._ctx("some context")])
        assert "What is X?" in prompt

    def test_contains_context_text(self):
        prompt = _make_prompt("Q?", [self._ctx("important context")])
        assert "important context" in prompt

    def test_contains_answer_label(self):
        prompt = _make_prompt("Q?", [self._ctx("ctx")])
        assert "Answer:" in prompt

    def test_not_found_instruction_present(self):
        prompt = _make_prompt("Q?", [self._ctx("ctx")])
        assert "Not found in the document" in prompt


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_list(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )
        mock_docs = [{"text": "doc0"}, {"text": "doc1"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((1, 384), dtype=np.float32)

        with (
            patch("app.rag._load_artifacts", return_value=(mock_index, mock_docs)),
            patch("app.rag._get_embedder", return_value=mock_embedder),
        ):
            hits = retrieve("test query", top_k=2)

        assert isinstance(hits, list)
        assert len(hits) == 2

    def test_retrieve_hit_structure(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        mock_docs = [{"text": "doc text", "page": 2, "chunk_id": "cid1"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((1, 384), dtype=np.float32)

        with (
            patch("app.rag._load_artifacts", return_value=(mock_index, mock_docs)),
            patch("app.rag._get_embedder", return_value=mock_embedder),
        ):
            hits = retrieve("query", top_k=1)

        hit = hits[0]
        assert "faiss_id" in hit
        assert "score" in hit
        assert "page" in hit
        assert "chunk_id" in hit
        assert "text" in hit

    def test_retrieve_ignores_negative_indices(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, -1.0]], dtype=np.float32),
            np.array([[0, -1]], dtype=np.int64),
        )
        mock_docs = [{"text": "only"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((1, 384), dtype=np.float32)

        with (
            patch("app.rag._load_artifacts", return_value=(mock_index, mock_docs)),
            patch("app.rag._get_embedder", return_value=mock_embedder),
        ):
            hits = retrieve("query", top_k=2)

        assert len(hits) == 1


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    def _mock_hit(self, score: float) -> Dict[str, Any]:
        return {"faiss_id": 0, "score": score, "page": 1, "chunk_id": "c1", "text": "sample"}

    def test_not_found_when_no_contexts(self):
        with (
            patch("app.rag.retrieve", return_value=[self._mock_hit(0.1)]),
            patch("app.rag.select_contexts", return_value=[]),
        ):
            result = answer_question("some question")
        assert result["answer"] == "Not found in the document."
        assert result["contexts"] == []

    def test_not_found_includes_hits(self):
        hits = [self._mock_hit(0.1)]
        with (
            patch("app.rag.retrieve", return_value=hits),
            patch("app.rag.select_contexts", return_value=[]),
        ):
            result = answer_question("some question")
        assert len(result["hits"]) == 1

    def test_calls_generator_when_contexts_exist(self):
        ctx = [self._mock_hit(0.9)]
        mock_gen = MagicMock(return_value=[{"generated_text": "  the answer  "}])

        with (
            patch("app.rag.retrieve", return_value=ctx),
            patch("app.rag.select_contexts", return_value=ctx),
            patch("app.rag._get_generator", return_value=mock_gen),
        ):
            result = answer_question("some question")

        mock_gen.assert_called_once()
        assert result["answer"] == "the answer"

    def test_answer_is_stripped(self):
        ctx = [self._mock_hit(0.9)]
        mock_gen = MagicMock(return_value=[{"generated_text": "\n  whitespace answer \n"}])

        with (
            patch("app.rag.retrieve", return_value=ctx),
            patch("app.rag.select_contexts", return_value=ctx),
            patch("app.rag._get_generator", return_value=mock_gen),
        ):
            result = answer_question("q")

        assert result["answer"] == "whitespace answer"

    def test_result_structure_with_contexts(self):
        ctx = [self._mock_hit(0.9)]
        mock_gen = MagicMock(return_value=[{"generated_text": "answer text"}])

        with (
            patch("app.rag.retrieve", return_value=ctx),
            patch("app.rag.select_contexts", return_value=ctx),
            patch("app.rag._get_generator", return_value=mock_gen),
        ):
            result = answer_question("q")

        assert "answer" in result
        assert "contexts" in result
        assert "hits" in result

    def test_hits_capped_at_five(self):
        many_hits = [self._mock_hit(0.9) for _ in range(10)]
        mock_gen = MagicMock(return_value=[{"generated_text": "ans"}])

        with (
            patch("app.rag.retrieve", return_value=many_hits),
            patch("app.rag.select_contexts", return_value=[many_hits[0]]),
            patch("app.rag._get_generator", return_value=mock_gen),
        ):
            result = answer_question("q")

        assert len(result["hits"]) <= 5
