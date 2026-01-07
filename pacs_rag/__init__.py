"""PACS term indexing and retrieval utilities."""

from .embedder import EmbeddingProvider, HashEmbeddingProvider, OllamaEmbeddingProvider, build_embedder
from .index import SqliteIndex, Suggestion
from .lexicon import cluster_terms, suggest_ngrams
from .ingest import ingest_from_mcp, ingest_from_mcp_async, ingest_terms
from .retrieve import retrieve

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "SqliteIndex",
    "Suggestion",
    "build_embedder",
    "ingest_from_mcp",
    "ingest_from_mcp_async",
    "ingest_terms",
    "cluster_terms",
    "suggest_ngrams",
    "retrieve",
]
