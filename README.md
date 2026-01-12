# pacs-rag

Lightweight PACS terminology index + retrieval to support DICOM query rewriting.

`pacs-rag` ingests non‑PHI study/series description terms, builds a local SQLite
index with embeddings, and returns top‑K similar terms for query rewriting. It
also exports a starter lexicon for manual curation.

## Why this exists

PACS sites often use local terminology. A lightweight RAG index lets
`dicom-nlquery` rewrite queries using **real site terms** without inventing
clinical filters.

## Install

```bash
cd pacs-rag
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# optional MCP client for ingest-mcp
uv pip install -e ".[dev,mcp]"
```

## CLI usage

### Ingest from JSON

```bash
pacs-rag ingest \
  --index data/pacs_terms.sqlite \
  --input ./terms.json \
  --provider hash \
  --dim 64
```

### Ingest from MCP (dicom-mcp)

```bash
pacs-rag ingest-mcp \
  --mcp-command dicom-mcp \
  --config-path ../configs/dicom.yaml \
  --index data/pacs_terms.sqlite \
  --study-date 20240101-20241231 \
  --max-studies 5000 \
  --include-series
```

### Retrieve suggestions

```bash
pacs-rag retrieve \
  --index data/pacs_terms.sqlite \
  --query "mr fetus" \
  --top-k 10 \
  --min-score 0.2
```

### Export lexicon for manual curation

```bash
pacs-rag export-lexicon \
  --index data/pacs_terms.sqlite \
  --output ../dicom-nlquery/configs/lexicon.generated.yaml \
  --min-count 2
```

The export includes:
- `synonyms`: empty buckets to fill manually
- `ngrams`: frequent bi‑grams
- `clusters`: simple token‑overlap clusters (review and edit)

## Data model & normalization

Each term row stores:
- `text` (StudyDescription/SeriesDescription/BodyPartExamined/ProtocolName)
- `level` (study/series)
- `modality` (normalized)
- `count`
- `last_seen_date`

Normalization notes:
- **Modality**: if multi‑valued (list/tuple), values are joined with `\` to
  match DICOM conventions. Empty values become `None`.
- **Dates**: stored as `YYYYMMDD` when possible; other values are sanitized to
  digit‑only formats.

## PHI safety

Ingestion skips terms that look like PHI:
- Strings containing `^` (likely DICOM person name)
- Long numeric tokens (6+ digits)

Only non‑PHI fields are stored.

## Embeddings & retrieval

Providers:
- `hash` (default): deterministic, no external dependency
- `ollama`: local HTTP embeddings

Retrieval uses simple cosine similarity over all stored vectors. This is
intentional to keep the system simple and easy to operate for small/medium term
sets. For larger corpora, consider swapping in ANN indexing.

## Design decisions (documented)

- **SQLite only**: keeps the index local and portable.
- **Deterministic outputs**: stable suggestions aid reproducible search.
- **No PHI persistence**: aggressive filtering reduces leakage risk.

## Testing

```bash
cd pacs-rag
uv run --with-editable .[dev] -m pytest -v
```
