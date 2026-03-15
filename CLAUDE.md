# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development

```bash
# Local run
pip install -r requirements.txt
python -m app.main

# Docker
docker compose up -d --build

# Rebuild after code changes
docker compose up -d --build
```

Server: http://localhost:8004 (Web UI), http://localhost:8004/mcp (MCP endpoint)

## Architecture

Three-layer design, single FastAPI process:

```
MCP clients ──POST /mcp──► FastMCP (6 tools) ──► storage.py (SQLite, SoT)
Browser    ──GET /     ──► Jinja2 web UI     ──► search.py  (ChromaDB, derived index)
```

- **storage.py** — SQLite CRUD, source of truth. Table `templates` (id, name, description, code, tags, created_at, updated_at). Tags stored as JSON array. Has migration logic from old `snippets` schema.
- **search.py** — ChromaDB vector index + hybrid search. Embedding strategy: tries OpenAI-compatible API first (LM Studio/Ollama), falls back to local SentenceTransformer (`intfloat/multilingual-e5-small`). Hybrid search: 1 word → full-text first; 2-3 words → vector first; 4+ words → vector only.
- **main.py** — FastMCP tools + FastAPI web routes + Jinja2 templates. MCP mounted at `/mcp`, web UI at `/`.
- **config.py** — All configuration via environment variables.

## Adding/Modifying MCP Tools

Tools are defined in `app/main.py` as `@mcp.tool()` decorated functions. Each tool calls `storage.*` for data and `search.*` for indexing. When adding a tool:
1. Add the function with `@mcp.tool()` in `main.py`
2. Add storage function in `storage.py` if needed
3. Update search index via `search.index_template()` / `search.update_index()` / `search.delete_index()` for write operations

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_BASE` | `http://localhost:1234` | Embedding API URL |
| `OPENAI_MODEL` | — | API model name (overrides local) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local fallback model |
| `RESET_CHROMA` | `false` | Force reindex on startup |
| `HTTP_PORT` | `8004` | Server port |
| `USESSE` | `false` | SSE transport instead of Streamable HTTP |

## Startup Flow

`_startup()` → `storage.init_db()` (migrate if needed) → `search.init_search_engine()` → `search.reindex_all()` (if forced or empty) → Ready

## Web UI Validation

Form submissions require: name ≥ 3 chars, description ≥ 10 chars, code ≥ 10 chars.
