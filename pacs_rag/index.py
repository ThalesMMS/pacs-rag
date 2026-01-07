from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sqlite3
from typing import Iterable

from .embedder import EmbeddingProvider


@dataclass(frozen=True)
class Suggestion:
    text: str
    score: float
    level: str | None = None
    modality: str | None = None
    count: int | None = None
    last_seen_date: str | None = None


class SqliteIndex:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            existing = conn.execute("PRAGMA table_info(terms)").fetchall()
            if not existing:
                self._create_terms_table(conn)
                return
            pk_columns = [
                row[1]
                for row in sorted(existing, key=lambda item: item[5])
                if row[5] > 0
            ]
            if pk_columns != ["text", "level", "modality"]:
                self._migrate_terms_table(conn)

    def _create_terms_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                text TEXT NOT NULL,
                level TEXT NOT NULL,
                modality TEXT NOT NULL,
                count INTEGER,
                last_seen_date TEXT,
                vector TEXT,
                PRIMARY KEY (text, level, modality)
            )
            """
        )

    def _migrate_terms_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE terms RENAME TO terms_old")
        self._create_terms_table(conn)
        conn.execute(
            """
            INSERT INTO terms (text, level, modality, count, last_seen_date, vector)
            SELECT
                text,
                COALESCE(level, ''),
                COALESCE(modality, ''),
                count,
                last_seen_date,
                vector
            FROM terms_old
            """
        )
        conn.execute("DROP TABLE terms_old")

    def upsert_terms(self, terms: list[dict], vectors: list[list[float]]) -> None:
        if len(terms) != len(vectors):
            raise ValueError("terms and vectors must have same length")
        with sqlite3.connect(self.path) as conn:
            for term, vector in zip(terms, vectors, strict=True):
                text = term.get("text")
                if not text:
                    continue
                level = _normalize_key(term.get("level"))
                modality = _normalize_key(term.get("modality"))
                existing = conn.execute(
                    "SELECT count FROM terms WHERE text = ? AND level = ? AND modality = ?",
                    (text, level, modality),
                ).fetchone()
                count = int(term.get("count") or 1)
                if existing:
                    count = max(count, int(existing[0]))
                conn.execute(
                    """
                    INSERT INTO terms (text, level, modality, count, last_seen_date, vector)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(text, level, modality) DO UPDATE SET
                        level=excluded.level,
                        modality=excluded.modality,
                        count=excluded.count,
                        last_seen_date=excluded.last_seen_date,
                        vector=excluded.vector
                    """,
                    (
                        text,
                        level,
                        modality,
                        count,
                        term.get("last_seen_date"),
                        json.dumps(vector),
                    ),
                )

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.2,
    ) -> list[Suggestion]:
        if not query_vector:
            return []
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT text, level, modality, count, last_seen_date, vector FROM terms"
            ).fetchall()
        results: list[Suggestion] = []
        for text, level, modality, count, last_seen_date, vector_json in rows:
            if not vector_json:
                continue
            vector = json.loads(vector_json)
            score = _cosine_similarity(query_vector, vector)
            if score < min_score:
                continue
            results.append(
                Suggestion(
                    text=text,
                    score=score,
                    level=_denormalize_key(level),
                    modality=_denormalize_key(modality),
                    count=count,
                    last_seen_date=last_seen_date,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def top_terms(self, min_count: int = 1, limit: int = 200) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT text, level, modality, count, last_seen_date
                FROM terms
                WHERE count >= ?
                ORDER BY count DESC, text ASC
                LIMIT ?
                """,
                (min_count, limit),
            ).fetchall()
        return [
            {
                "text": text,
                "level": _denormalize_key(level),
                "modality": _denormalize_key(modality),
                "count": count,
                "last_seen_date": last_seen_date,
            }
            for text, level, modality, count, last_seen_date in rows
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for i in range(length):
        numerator += left[i] * right[i]
        left_norm += left[i] * left[i]
        right_norm += right[i] * right[i]
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def _normalize_key(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _denormalize_key(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return value
