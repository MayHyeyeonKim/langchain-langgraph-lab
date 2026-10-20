# AI Technical Document Reviewer

An educational project that reviews technical documents with local LLMs.

## What it does

1. Extracts text from a PDF (or accepts plain text).
2. `Analyzer` identifies actionable issues.
3. `Writer` revises the document.
4. `Reviewer` approves the revision or sends it back to the writer.
5. LangGraph limits revision attempts and preserves workflow state.

## Quick start

```bash
cp .env.example .env
uv sync --group dev
ollama serve # separate terminal, if it is not already running
ollama pull qwen3:8b
# Verify just the Analyzer before running the complete Writer/Reviewer workflow.
uv run python -m document_reviewer.cli --analyze-only examples/sample_readme.md

# Run the full workflow after the Analyzer result looks sound.
uv run python -m document_reviewer.cli examples/sample_readme.md
```

Run tests with `uv run pytest`.

## API

```bash
uv run python apps/api/manage.py runserver
curl -F "file=@examples/sample.pdf" http://127.0.0.1:8000/api/reviews/
```

`POST /api/reviews/` accepts a text-based PDF or `text` form field. The result includes the
analysis, revised document, final review, and revision count.

Optionally attach one or more `reference_files` (text-based PDFs) to enable local RAG. Install
the extra dependencies and embedding model first:

```bash
uv sync --group rag
ollama pull nomic-embed-text
```

The API creates a request-scoped local Chroma index, retrieves relevant chunks, gives them to the
Analyzer and Reviewer, and returns the source filenames/pages used in the response.

## Learning path

Start in `src/document_reviewer/agents.py`, then follow the state through
`src/document_reviewer/graph.py`. RAG and LoRA instructions live in `docs/` and `training/`.
