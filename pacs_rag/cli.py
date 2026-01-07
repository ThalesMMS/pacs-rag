from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .embedder import build_embedder
from .index import SqliteIndex
from .lexicon import cluster_terms, suggest_ngrams
from .ingest import ingest_terms
from .retrieve import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="pacs-rag CLI")
    sub = parser.add_subparsers(dest="command")

    ingest_cmd = sub.add_parser("ingest", help="Ingest terms from JSON")
    ingest_cmd.add_argument("--index", required=True)
    ingest_cmd.add_argument("--input", required=True)
    ingest_cmd.add_argument("--provider", default="hash")
    ingest_cmd.add_argument("--model", default=None)
    ingest_cmd.add_argument("--base-url", dest="base_url", default=None)
    ingest_cmd.add_argument("--dim", type=int, default=64)

    ingest_mcp_cmd = sub.add_parser("ingest-mcp", help="Ingest terms via MCP")
    ingest_mcp_cmd.add_argument("--mcp-command", dest="mcp_command", default="dicom-mcp")
    ingest_mcp_cmd.add_argument("--config-path", dest="config_path", default=None)
    ingest_mcp_cmd.add_argument("--arg", dest="args", action="append", default=[])
    ingest_mcp_cmd.add_argument("--index", required=True)
    ingest_mcp_cmd.add_argument("--study-date", dest="study_date", default=None)
    ingest_mcp_cmd.add_argument("--max-studies", dest="max_studies", type=int, default=None)
    ingest_mcp_cmd.add_argument("--include-series", dest="include_series", action="store_true")
    ingest_mcp_cmd.add_argument("--no-include-series", dest="include_series", action="store_false")
    ingest_mcp_cmd.set_defaults(include_series=True)
    ingest_mcp_cmd.add_argument("--provider", default="hash")
    ingest_mcp_cmd.add_argument("--model", default=None)
    ingest_mcp_cmd.add_argument("--base-url", dest="base_url", default=None)
    ingest_mcp_cmd.add_argument("--dim", type=int, default=64)

    retrieve_cmd = sub.add_parser("retrieve", help="Retrieve suggestions")
    retrieve_cmd.add_argument("--index", required=True)
    retrieve_cmd.add_argument("--query", required=True)
    retrieve_cmd.add_argument("--top-k", type=int, default=10)
    retrieve_cmd.add_argument("--min-score", type=float, default=0.2)
    retrieve_cmd.add_argument("--provider", default="hash")
    retrieve_cmd.add_argument("--model", default=None)
    retrieve_cmd.add_argument("--base-url", dest="base_url", default=None)
    retrieve_cmd.add_argument("--dim", type=int, default=64)

    export_cmd = sub.add_parser("export-lexicon", help="Export frequent terms to lexicon YAML")
    export_cmd.add_argument("--index", required=True)
    export_cmd.add_argument("--output", required=True)
    export_cmd.add_argument("--min-count", type=int, default=2)
    export_cmd.add_argument("--limit", type=int, default=200)

    args = parser.parse_args()
    if args.command == "ingest":
        embedder = build_embedder(args.provider, args.model, args.base_url, args.dim)
        terms = json.loads(open(args.input, "r", encoding="utf-8").read())
        ingest_terms(args.index, terms, embedder)
        return

    if args.command == "retrieve":
        embedder = build_embedder(args.provider, args.model, args.base_url, args.dim)
        results = retrieve(
            index_path=args.index,
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            embedder=embedder,
        )
        print(json.dumps([item.__dict__ for item in results], indent=2))
        return

    if args.command == "ingest-mcp":
        try:
            from .mcp_client import McpSession, build_stdio_server_params
            from .ingest import ingest_from_mcp_async
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"MCP dependency missing: {exc}")

        embedder = build_embedder(args.provider, args.model, args.base_url, args.dim)
        stdio_args = list(args.args)
        if args.config_path:
            stdio_args.insert(0, args.config_path)

        server_params = build_stdio_server_params(
            command=args.mcp_command,
            args=stdio_args,
        )

        async def _run() -> None:
            async with McpSession(server_params) as session:
                await ingest_from_mcp_async(
                    session,
                    index_path=args.index,
                    embedder=embedder,
                    study_date=args.study_date,
                    max_studies=args.max_studies,
                    include_series=args.include_series,
                )

        import asyncio

        asyncio.run(_run())
        return

    if args.command == "export-lexicon":
        index = SqliteIndex(args.index)
        terms = index.top_terms(min_count=args.min_count, limit=args.limit)
        term_texts = [
            term["text"]
            for term in terms
            for _ in range(max(1, int(term.get("count") or 1)))
        ]
        clusters = cluster_terms(term_texts)
        output = {
            "synonyms": {term["text"]: [] for term in terms},
            "ngrams": suggest_ngrams(term_texts, n=2, min_count=args.min_count),
            "clusters": [
                {"seed": cluster.seed, "terms": cluster.terms, "score": cluster.score}
                for cluster in clusters
                if len(cluster.terms) > 1
            ],
        }
        Path(args.output).write_text(
            yaml.safe_dump(output, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
