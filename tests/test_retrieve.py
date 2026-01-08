from __future__ import annotations

import yaml

from pacs_rag.embedder import HashEmbeddingProvider
from pacs_rag.index import SqliteIndex
from pacs_rag.lexicon import cluster_terms, suggest_ngrams
from pacs_rag.ingest import ingest_terms
from pacs_rag.retrieve import retrieve


def test_ingest_and_retrieve(tmp_path) -> None:
    index_path = tmp_path / "terms.sqlite"
    embedder = HashEmbeddingProvider(dim=16)
    terms = [
        {
            "text": "MR fetal study",
            "level": "study",
            "modality": "MR",
            "count": 2,
            "last_seen_date": "20240101",
        }
    ]

    ingest_terms(str(index_path), terms, embedder)
    results = retrieve(
        index_path=str(index_path),
        query="fetal",
        top_k=5,
        min_score=0.0,
        embedder=embedder,
    )

    assert results
    assert results[0].text == "MR fetal study"


def test_export_lexicon(tmp_path) -> None:
    index_path = tmp_path / "terms.sqlite"
    embedder = HashEmbeddingProvider(dim=8)
    terms = [
        {"text": "MR fetal study", "level": "study", "modality": "MR", "count": 3},
        {"text": "CT cranial", "level": "study", "modality": "CT", "count": 1},
    ]

    ingest_terms(str(index_path), terms, embedder)
    index = SqliteIndex(str(index_path))
    top = index.top_terms(min_count=2, limit=10)
    term_texts = [
        term["text"]
        for term in top
        for _ in range(max(1, int(term.get("count") or 1)))
    ]
    output = {
        "synonyms": {term["text"]: [] for term in top},
        "ngrams": suggest_ngrams(term_texts, n=2, min_count=2),
        "clusters": [
            {"seed": cluster.seed, "terms": cluster.terms, "score": cluster.score}
            for cluster in cluster_terms([term["text"] for term in top])
            if len(cluster.terms) > 1
        ],
    }
    output_path = tmp_path / "lexicon.yaml"
    output_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")

    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert "MR fetal study" in loaded["synonyms"]
    assert "CT cranial" not in loaded["synonyms"]
    assert any(item["text"] == "mr fetal" for item in loaded["ngrams"])
