"""FastAPI + FastMCP application: 6 MCP tools + CRUD web UI."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastmcp import FastMCP
from pathlib import Path

from app import config, storage, search

# --- Jinja2 templates ---
TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- FastMCP ---
mcp = FastMCP(name=config.SERVER_NAME)


# ===== MCP Tools (6) =====

@mcp.tool()
def templatesearch(query: str) -> str:
    """Searches the 1C code template database using hybrid semantic + full-text search.

    Args:
        query: Search term or question in Russian describing desired functionality.

    Returns:
        Formatted string with matching templates (description + code).
    """
    logging.info(f"templatesearch: '{query}'")
    results = search.semantic_search(query)
    if not results:
        return "No relevant templates found for your query."

    parts = []
    for meta in results:
        desc = meta.get('description', '')
        code = meta.get('code', '')
        title = desc.split('.')[0][:80] if desc else 'Unknown'
        parts.append(f"**Template:** {title}\n**Description:** {desc}\n**Code:**\n```bsl\n{code}\n```")
    return "\n---\n".join(parts)


@mcp.tool()
def list_templates() -> str:
    """List all templates (id, name, description, tags) without code.

    Returns:
        JSON array of template summaries.
    """
    import json
    templates = storage.list_templates()
    summaries = [
        {"id": t["id"], "name": t["name"], "description": t["description"][:200], "tags": t["tags"]}
        for t in templates
    ]
    return json.dumps(summaries, ensure_ascii=False, indent=2)


@mcp.tool()
def get_template(template_id: int) -> str:
    """Get full template with code by ID.

    Args:
        template_id: Template ID.

    Returns:
        JSON object with full template data, or error message.
    """
    import json
    tpl = storage.get_template(template_id)
    if tpl is None:
        return f"Template with id={template_id} not found."
    return json.dumps(tpl, ensure_ascii=False, indent=2)


@mcp.tool()
def add_template(name: str, description: str, code: str, tags: str = "") -> str:
    """Add a new template to the database.

    Args:
        name: Short template name.
        description: Detailed description of what the template does.
        code: BSL code of the template.
        tags: Comma-separated tags (optional).

    Returns:
        Confirmation message with new template ID.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tpl = storage.create_template(name, description, code, tag_list)
    search.index_template(tpl)
    return f"Template created with id={tpl['id']}"


@mcp.tool()
def update_template(template_id: int, name: str = None, description: str = None,
                    code: str = None, tags: str = None) -> str:
    """Update an existing template.

    Args:
        template_id: Template ID to update.
        name: New name (optional).
        description: New description (optional).
        code: New code (optional).
        tags: New comma-separated tags (optional).

    Returns:
        Confirmation or error message.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags is not None else None
    tpl = storage.update_template(template_id, name=name, description=description,
                                  code=code, tags=tag_list)
    if tpl is None:
        return f"Template with id={template_id} not found."
    search.update_index(tpl)
    return f"Template id={template_id} updated."


@mcp.tool()
def delete_template(template_id: int) -> str:
    """Delete a template by ID.

    Args:
        template_id: Template ID to delete.

    Returns:
        Confirmation or error message.
    """
    if storage.delete_template(template_id):
        search.delete_index(template_id)
        return f"Template id={template_id} deleted."
    return f"Template with id={template_id} not found."


# ===== Startup / Lifespan =====

def _startup():
    """Initialize storage and search engine."""
    logging.info("=== Application Startup ===")

    # Step 1: SQLite
    migrated = storage.init_db()

    # Step 2: ChromaDB + embeddings
    force_reindex = config.RESET_CHROMA or migrated
    search.init_search_engine(force_reindex=force_reindex)

    # Step 3: Reindex if needed
    if search.collection is not None and (force_reindex or search.collection.count() == 0):
        logging.info("Indexing all templates...")
        templates = storage.list_all_for_indexing()
        search.reindex_all(templates)

    logging.info("=== Application Ready ===")


# --- MCP ASGI app ---
logging.info(f"Creating FastMCP app, transport={config.TRANSPORT}")
mcp_app = mcp.http_app(transport=config.TRANSPORT, path="/")


@asynccontextmanager
async def combined_lifespan(app_instance):
    _startup()
    if hasattr(mcp_app, 'router') and hasattr(mcp_app.router, 'lifespan_context'):
        async with mcp_app.router.lifespan_context(mcp_app) as state:
            yield state
    else:
        yield
    logging.info("Server shutting down...")


# --- FastAPI app ---
app = FastAPI(
    title="1C Templates MCP Server",
    version="2.0.0",
    lifespan=combined_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Web Routes =====

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, q: str = ""):
    templates = storage.list_templates(query=q if q else None)
    return jinja.TemplateResponse("index.html", {
        "request": request, "templates": templates, "query": q
    })


@app.get("/new", response_class=HTMLResponse)
async def new_template_form(request: Request):
    return jinja.TemplateResponse("edit.html", {"request": request, "tpl": None})


@app.post("/new", response_class=HTMLResponse)
async def new_template_submit(request: Request,
                              name: str = Form(""),
                              description: str = Form(""),
                              code: str = Form(""),
                              tags: str = Form("")):
    errors = _validate(name, description, code)
    if errors:
        return jinja.TemplateResponse("edit.html", {
            "request": request, "tpl": None,
            "errors": errors, "form": {"name": name, "description": description, "code": code, "tags": tags}
        })
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tpl = storage.create_template(name.strip(), description.strip(), code.strip(), tag_list)
    search.index_template(tpl)
    return RedirectResponse(url=f"/{tpl['id']}", status_code=303)


@app.get("/extend", response_class=HTMLResponse)
@app.get("/extend/", response_class=HTMLResponse)
async def extend_redirect():
    return RedirectResponse(url="/new", status_code=302)


@app.get("/{template_id:int}", response_class=HTMLResponse)
async def view_template(request: Request, template_id: int):
    tpl = storage.get_template(template_id)
    if tpl is None:
        return HTMLResponse("<h1>404 - Template not found</h1>", status_code=404)
    return jinja.TemplateResponse("view.html", {"request": request, "tpl": tpl})


@app.get("/{template_id:int}/edit", response_class=HTMLResponse)
async def edit_template_form(request: Request, template_id: int):
    tpl = storage.get_template(template_id)
    if tpl is None:
        return HTMLResponse("<h1>404 - Template not found</h1>", status_code=404)
    return jinja.TemplateResponse("edit.html", {"request": request, "tpl": tpl})


@app.post("/{template_id:int}/edit", response_class=HTMLResponse)
async def edit_template_submit(request: Request, template_id: int,
                               name: str = Form(""),
                               description: str = Form(""),
                               code: str = Form(""),
                               tags: str = Form("")):
    errors = _validate(name, description, code)
    if errors:
        tpl = storage.get_template(template_id)
        return jinja.TemplateResponse("edit.html", {
            "request": request, "tpl": tpl,
            "errors": errors, "form": {"name": name, "description": description, "code": code, "tags": tags}
        })
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tpl = storage.update_template(template_id, name=name.strip(), description=description.strip(),
                                  code=code.strip(), tags=tag_list)
    if tpl is None:
        return HTMLResponse("<h1>404 - Template not found</h1>", status_code=404)
    search.update_index(tpl)
    return RedirectResponse(url=f"/{template_id}", status_code=303)


@app.post("/{template_id:int}/delete")
async def delete_template_web(template_id: int):
    storage.delete_template(template_id)
    search.delete_index(template_id)
    return RedirectResponse(url="/", status_code=303)


def _validate(name: str, description: str, code: str) -> list[str]:
    errors = []
    if not name or len(name.strip()) < 3:
        errors.append("Название должно содержать не менее 3 символов")
    if not description or len(description.strip()) < 10:
        errors.append("Описание должно содержать не менее 10 символов")
    if not code or len(code.strip()) < 10:
        errors.append("Код должен содержать не менее 10 символов")
    return errors


# ===== Mount MCP + static =====

# Static files for Monaco BSL console
bsl_console_path = Path("/app/bsl_console")
if bsl_console_path.exists():
    app.mount("/bsl_console", StaticFiles(directory=str(bsl_console_path), html=True), name="bsl_console")
else:
    # Local dev fallback
    local_bsl = Path(__file__).parent.parent / "bsl_console"
    if local_bsl.exists():
        app.mount("/bsl_console", StaticFiles(directory=str(local_bsl), html=True), name="bsl_console")

# Mount MCP (always at /mcp to avoid overriding web routes)
app.mount("/mcp", mcp_app)

logging.info(f"MCP endpoint: {'GET /mcp (SSE)' if config.TRANSPORT == 'sse' else 'POST /mcp'}")
logging.info(f"Web UI: http://0.0.0.0:{config.HTTP_PORT}/")


# --- Entry point ---
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info(f"Starting {config.SERVER_NAME} on port {config.HTTP_PORT}")
    logging.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT, log_level="info")
