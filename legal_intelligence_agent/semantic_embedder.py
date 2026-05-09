from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import faiss  # type: ignore


class EmbeddingError(Exception):
    pass


class SemanticEmbedder:
    def __init__(self, llm_client: Any = None):
        # llm_client is expected to provide: llm_client.embed(text) -> np.ndarray(shape=(384,))
        self.llm_client = llm_client
        self._dim = 384
        self._index: Optional[Any] = None
        self._chunk_store: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clears the index and chunk store for the next document."""
        self._index = None
        self._chunk_store = []

    def _embed_text(self, text: str) -> np.ndarray:
        if self.llm_client is None or not hasattr(self.llm_client, "embed"):
            raise EmbeddingError("llm_client.embed is not available")

        try:
            vec = self.llm_client.embed(text)
        except Exception as e:  # noqa: BLE001 - convert to domain error
            raise EmbeddingError(f"embedding call failed: {e}") from e

        vec = np.asarray(vec, dtype="float32")
        if vec.shape != (self._dim,):
            raise EmbeddingError(f"embedding vector must be shape ({self._dim},)")
        return vec

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero; zero vectors remain zero.
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def embed_chunks(self, chunks: list[dict]) -> None:
        """Embeds clause chunks and builds a local FAISS index."""
        if not chunks:
            raise EmbeddingError("empty or no chunks provided")

        # Embed a single document; replace any existing index.
        self.reset()

        vectors: List[np.ndarray] = []
        for chunk in chunks:
            if "text" not in chunk:
                raise EmbeddingError("each chunk must include a 'text' field")
            vec = self._embed_text(str(chunk["text"]))
            vectors.append(vec)

        mat = np.vstack(vectors).astype("float32")
        mat = self._normalize_rows(mat)

        index = faiss.IndexFlatIP(self._dim)
        index.add(mat)

        self._index = index
        self._chunk_store = list(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieves top-k semantically similar chunks grounded in the embedded index."""
        if self._index is None or not self._chunk_store:
            return []

        if top_k <= 0:
            return []

        qvec = self._embed_text(query).reshape(1, self._dim).astype("float32")
        qvec = self._normalize_rows(qvec)

        # FAISS returns (similarity, index)
        scores, indices = self._index.search(qvec, top_k)
        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._chunk_store):
                continue

            chunk = dict(self._chunk_store[idx])
            cosine = float(score)
            # Ensure score is in [0, 1] per tests (cosine in [-1, 1]).
            similarity_score = max(0.0, min(1.0, cosine))
            chunk["similarity_score"] = float(similarity_score)
            results.append(chunk)

        return results

