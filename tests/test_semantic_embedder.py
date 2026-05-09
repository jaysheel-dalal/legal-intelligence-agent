from __future__ import annotations

from typing import Any, Dict, List

import pytest

from legal_intelligence_agent.semantic_embedder import EmbeddingError, SemanticEmbedder


def _make_chunk(base: Dict[str, Any], position: int, text_suffix: str) -> Dict[str, Any]:
    text = f"{base['text']}{text_suffix}"
    return {
        "text": text,
        "clause_type": str(base.get("clause_type", "general")),
        "position": int(position),
        "char_count": len(text),
    }


class TestHappyPath:
    def test_embed_and_retrieve_returns_results(self, semantic_embedder, document_parser, clean_nda_path):
        """test_embed_and_retrieve_returns_results validates output schema and top-k bound."""
        doc = document_parser.parse_document(document_path=clean_nda_path)
        chunks = doc["chunks"][:5]
        if len(chunks) < 5:
            chunks = [_make_chunk(chunks[i % len(chunks)], i, f" Clause {i}") for i in range(5)]

        semantic_embedder.embed_chunks(chunks)
        results = semantic_embedder.retrieve("confidentiality obligations", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        required = {"text", "clause_type", "position", "char_count", "similarity_score"}
        for r in results:
            assert isinstance(r, dict)
            assert required.issubset(set(r.keys()))

    def test_similarity_score_is_float(self, semantic_embedder, document_parser, clean_vendor_path):
        """test_similarity_score_is_float ensures scores are floats within [0, 1]."""
        doc = document_parser.parse_document(document_path=clean_vendor_path)
        chunks = doc["chunks"]

        semantic_embedder.embed_chunks(chunks)
        results = semantic_embedder.retrieve("payment and liability", top_k=3)

        for r in results:
            assert isinstance(r["similarity_score"], float)
            assert 0.0 <= r["similarity_score"] <= 1.0

    def test_top_k_limits_results(self, semantic_embedder, document_parser, clean_vendor_path):
        """test_top_k_limits_results verifies top_k exact/upper bounds."""
        doc = document_parser.parse_document(document_path=clean_vendor_path)
        base_chunks = doc["chunks"]
        assert len(base_chunks) > 0

        # Ensure we embed at least 10 chunks for deterministic top-k assertions.
        chunks: List[Dict[str, Any]] = []
        while len(chunks) < 10:
            for i, b in enumerate(base_chunks):
                if len(chunks) >= 10:
                    break
                chunks.append(_make_chunk(b, position=len(chunks), text_suffix=f" #{len(chunks)}"))

        semantic_embedder.embed_chunks(chunks)
        r2 = semantic_embedder.retrieve("payment schedule fees", top_k=2)
        assert len(r2) == 2

        r5 = semantic_embedder.retrieve("payment schedule fees", top_k=5)
        assert len(r5) <= 5

    def test_retrieve_returns_grounded_chunks_only(self, semantic_embedder, document_parser, clean_vendor_path):
        """test_retrieve_returns_grounded_chunks_only ensures returned chunks come from the embedded set."""
        doc = document_parser.parse_document(document_path=clean_vendor_path)
        original_chunks = doc["chunks"]
        semantic_embedder.embed_chunks(original_chunks)

        results = semantic_embedder.retrieve("IP ownership rights", top_k=3)
        original_texts = {c["text"] for c in original_chunks}

        assert len(results) <= 3
        for r in results:
            assert r["text"] in original_texts


class TestEdgeCases:
    def test_empty_chunks_raises_error(self, semantic_embedder):
        """test_empty_chunks_raises_error raises EmbeddingError for empty input."""
        with pytest.raises(EmbeddingError) as excinfo:
            semantic_embedder.embed_chunks([])
        msg = str(excinfo.value).lower()
        assert "empty" in msg or "no chunks" in msg

    def test_retrieve_before_embed_returns_empty(self, semantic_embedder):
        """test_retrieve_before_embed_returns_empty returns [] without crashing."""
        fresh = SemanticEmbedder(llm_client=semantic_embedder.llm_client)
        results = fresh.retrieve("confidentiality")
        assert results == []

    def test_reset_clears_index(self, semantic_embedder, document_parser, clean_nda_path):
        """test_reset_clears_index ensures reset removes stored index and chunks."""
        doc = document_parser.parse_document(document_path=clean_nda_path)
        chunks = doc["chunks"]

        semantic_embedder.embed_chunks(chunks)
        assert len(semantic_embedder.retrieve("confidentiality", top_k=3)) > 0

        semantic_embedder.reset()
        assert semantic_embedder.retrieve("confidentiality", top_k=3) == []

    def test_embedding_failure_raises_error(self, semantic_embedder, mock_embedding_client):
        """test_embedding_failure_raises_error converts embed failures to EmbeddingError."""
        doc_chunks = [
            {"text": "Confidentiality clause", "clause_type": "confidentiality", "position": 0, "char_count": 25}
        ]
        mock_embedding_client.embed.side_effect = RuntimeError("embed failed")

        with pytest.raises(EmbeddingError) as excinfo:
            semantic_embedder.embed_chunks(doc_chunks)

        assert "embedding" in str(excinfo.value).lower() or "failed" in str(excinfo.value).lower()


class TestRetrievalQuality:
    def test_relevant_query_scores_higher_than_irrelevant(self, semantic_embedder, document_parser, clean_nda_path):
        """test_relevant_query_scores_higher_than_irrelevant compares top score for relevant vs irrelevant queries."""
        doc = document_parser.parse_document(document_path=clean_nda_path)
        semantic_embedder.embed_chunks(doc["chunks"])

        conf = semantic_embedder.retrieve("confidentiality obligations", top_k=3)
        pay = semantic_embedder.retrieve("payment schedule and fees", top_k=3)

        assert len(conf) > 0
        assert len(pay) > 0
        assert conf[0]["similarity_score"] >= pay[0]["similarity_score"]

    def test_query_with_no_match_returns_low_scores(self, semantic_embedder, document_parser, clean_nda_path):
        """test_query_with_no_match_returns_low_scores ensures all scores are below 0.5 when no semantic match exists."""
        doc = document_parser.parse_document(document_path=clean_nda_path)
        semantic_embedder.embed_chunks(doc["chunks"])

        results = semantic_embedder.retrieve("bankruptcy filing procedures", top_k=3)
        assert len(results) > 0
        assert all(r["similarity_score"] < 0.5 for r in results)

    def test_retrieve_after_reset_and_re_embed(self, semantic_embedder, document_parser, clean_vendor_path, clean_nda_path):
        """test_retrieve_after_reset_and_re_embed ensures reset prevents cross-document retrieval contamination."""
        vendor_doc = document_parser.parse_document(document_path=clean_vendor_path)
        nda_doc = document_parser.parse_document(document_path=clean_nda_path)

        semantic_embedder.embed_chunks(vendor_doc["chunks"])
        _ = semantic_embedder.retrieve("payment", top_k=3)
        semantic_embedder.reset()

        semantic_embedder.embed_chunks(nda_doc["chunks"])
        results = semantic_embedder.retrieve("confidentiality", top_k=3)

        nda_texts = {c["text"] for c in nda_doc["chunks"]}
        assert len(results) > 0
        assert all(r["text"] in nda_texts for r in results)

