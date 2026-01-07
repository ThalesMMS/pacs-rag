from __future__ import annotations

import asyncio

from pacs_rag.embedder import HashEmbeddingProvider
from pacs_rag.index import SqliteIndex
from pacs_rag.ingest import ingest_from_mcp_async


class FakeClient:
    async def query_studies(self, **kwargs):
        return [
            {"StudyDescription": "MR fetal study", "StudyDate": "20240101"},
            {"StudyDescription": "CT cranio", "StudyDate": "20240102"},
        ]

    async def query_series(self, **kwargs):
        return []


def test_ingest_from_mcp_async_respects_max_studies(tmp_path) -> None:
    index_path = tmp_path / "terms.sqlite"
    embedder = HashEmbeddingProvider(dim=8)

    asyncio.run(
        ingest_from_mcp_async(
            FakeClient(),
            index_path=str(index_path),
            embedder=embedder,
            max_studies=1,
            include_series=False,
        )
    )

    index = SqliteIndex(str(index_path))
    top = index.top_terms(min_count=1, limit=10)
    assert len(top) == 1
