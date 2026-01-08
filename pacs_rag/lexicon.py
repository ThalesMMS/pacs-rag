from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Cluster:
    seed: str
    terms: list[str]
    score: float


STOPWORDS = {
    "a",
    "an",
    "and",
    "exam",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "study",
    "the",
    "to",
    "with",
    "without",
}


def suggest_ngrams(terms: list[str], n: int = 2, min_count: int = 2) -> list[dict]:
    counter: Counter[str] = Counter()
    for term in terms:
        tokens = _tokenize(term)
        if len(tokens) < n:
            continue
        for idx in range(len(tokens) - n + 1):
            gram = " ".join(tokens[idx : idx + n])
            counter[gram] += 1
    return [
        {"text": text, "count": count}
        for text, count in counter.most_common()
        if count >= min_count
    ]


def cluster_terms(terms: list[str], min_jaccard: float = 0.6) -> list[Cluster]:
    clusters: list[Cluster] = []
    for term in terms:
        tokens = set(_tokenize(term))
        if not tokens:
            continue
        placed = False
        for cluster in clusters:
            seed_tokens = set(_tokenize(cluster.seed))
            score = _jaccard(tokens, seed_tokens)
            if score >= min_jaccard:
                updated_terms = list(cluster.terms) + [term]
                clusters[clusters.index(cluster)] = Cluster(
                    seed=cluster.seed,
                    terms=updated_terms,
                    score=score,
                )
                placed = True
                break
        if not placed:
            clusters.append(Cluster(seed=term, terms=[term], score=1.0))
    return clusters


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [
        token
        for token in normalized.split()
        if token and token not in STOPWORDS and len(token) >= 2
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
