from __future__ import annotations

from datetime import date, datetime
import re
from itertools import islice
from typing import Iterable

from .embedder import EmbeddingProvider
from .index import SqliteIndex


def ingest_terms(
    index_path: str,
    terms: list[dict],
    embedder: EmbeddingProvider,
) -> None:
    entries = [term for term in terms if term.get("text")]
    vectors = embedder.embed([term.get("text", "") for term in entries])
    index = SqliteIndex(index_path)
    index.upsert_terms(entries, vectors)


def ingest_from_mcp(
    client: object,
    index_path: str,
    embedder: EmbeddingProvider,
    study_date: str | None = None,
    max_studies: int | None = None,
    include_series: bool = True,
) -> None:
    query_studies = getattr(client, "query_studies", None) or getattr(client, "query_study", None)
    if query_studies is None:
        raise RuntimeError("client lacks query_studies/query_study")
    query_series = getattr(client, "query_series", None)

    studies: Iterable[object] = query_studies(study_date=study_date)
    if max_studies is not None:
        studies = islice(studies, max_studies)

    terms: list[dict] = []
    for study in studies:
        description = _safe_text(_get_attr(study, "StudyDescription"))
        modality = _normalize_modality(_get_attr(study, "ModalitiesInStudy"))
        study_date_value = _get_attr(study, "StudyDate")
        if description:
            terms.append(
                {
                    "text": description,
                    "level": "study",
                    "modality": modality,
                    "count": 1,
                    "last_seen_date": study_date_value,
                }
            )

        if include_series and query_series is not None:
            study_uid = _get_attr(study, "StudyInstanceUID")
            if not study_uid:
                continue
            for series in query_series(study_instance_uid=study_uid):
                for field in ["SeriesDescription", "BodyPartExamined", "ProtocolName"]:
                    text = _safe_text(_get_attr(series, field))
                    if not text:
                        continue
                    terms.append(
                        {
                            "text": text,
                            "level": "series",
                            "modality": _normalize_modality(_get_attr(series, "Modality")),
                            "count": 1,
                            "last_seen_date": study_date_value,
                        }
                    )

    ingest_terms(index_path, _aggregate_terms(terms), embedder)


async def ingest_from_mcp_async(
    client: object,
    index_path: str,
    embedder: EmbeddingProvider,
    study_date: str | None = None,
    max_studies: int | None = None,
    include_series: bool = True,
) -> None:
    studies = await client.query_studies(study_date=study_date)
    if max_studies is not None:
        studies = list(studies)[:max_studies]
    terms: list[dict] = []
    for study in studies:
        description = _safe_text(_get_attr(study, "StudyDescription"))
        modality = _normalize_modality(_get_attr(study, "ModalitiesInStudy"))
        study_date_value = _get_attr(study, "StudyDate")
        if description:
            terms.append(
                {
                    "text": description,
                    "level": "study",
                    "modality": modality,
                    "count": 1,
                    "last_seen_date": study_date_value,
                }
            )
        if include_series and hasattr(client, "query_series"):
            study_uid = _get_attr(study, "StudyInstanceUID")
            if not study_uid:
                continue
            series_list = await client.query_series(study_instance_uid=study_uid)
            for series in series_list:
                for field in ["SeriesDescription", "BodyPartExamined", "ProtocolName"]:
                    text = _safe_text(_get_attr(series, field))
                    if not text:
                        continue
                    terms.append(
                        {
                            "text": text,
                            "level": "series",
                            "modality": _normalize_modality(_get_attr(series, "Modality")),
                            "count": 1,
                            "last_seen_date": study_date_value,
                        }
                    )
    ingest_terms(index_path, _aggregate_terms(terms), embedder)


def _aggregate_terms(terms: list[dict]) -> list[dict]:
    aggregated: dict[tuple[str, str | None, str | None], dict] = {}
    for term in terms:
        text = term.get("text")
        if not text:
            continue
        key = (text, term.get("level"), _normalize_modality(term.get("modality")))
        if key not in aggregated:
            normalized = dict(term)
            normalized["last_seen_date"] = _normalize_date(term.get("last_seen_date"))
            normalized["modality"] = key[2]
            aggregated[key] = normalized
            continue
        aggregated[key]["count"] = int(aggregated[key].get("count") or 0) + int(
            term.get("count") or 0
        )
        existing_date = _normalize_date(aggregated[key].get("last_seen_date"))
        incoming_date = _normalize_date(term.get("last_seen_date"))
        aggregated[key]["last_seen_date"] = max(existing_date, incoming_date)
    return list(aggregated.values())


def _safe_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "^" in text:
        return None
    if re.search(r"\b\d{6,}\b", text):
        return None
    return text


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_date(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return text
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return digits[:8]
    return text


def _normalize_modality(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return None
        return "\\".join(parts)
    text = str(value).strip()
    return text or None
