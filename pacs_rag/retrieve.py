from __future__ import annotations

from typing import Iterable

from .embedder import EmbeddingProvider
from .index import SqliteIndex, Suggestion


def retrieve(
    index_path: str,
    query: str,
    top_k: int = 10,
    min_score: float = 0.2,
    embedder: EmbeddingProvider | None = None,
) -> list[Suggestion]:
    if not query:
        return []
    if embedder is None:
        raise ValueError("embedder is required")
    index = SqliteIndex(index_path)
    vectors = embedder.embed([query])
    return index.retrieve(vectors[0], top_k=top_k, min_score=min_score)
