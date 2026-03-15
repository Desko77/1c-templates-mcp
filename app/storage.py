"""SQLite storage for templates. Source of truth for all template data."""

import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from app.config import TEMPLATES_DB_PATH, BUNDLED_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TEMPLATES_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> bool:
    """Initialize DB: copy seed if needed, migrate snippets -> templates.
    Returns True if migration was performed (ChromaDB reindex needed)."""
    migrated = False

    # Copy seed DB if runtime copy does not exist
    db_path = Path(TEMPLATES_DB_PATH)
    if not db_path.exists():
        bundled = Path(BUNDLED_DB_PATH)
        if not bundled.exists():
            logging.error(f"Seed database not found at {BUNDLED_DB_PATH}")
            raise FileNotFoundError(f"Seed database not found: {BUNDLED_DB_PATH}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, db_path)
        logging.info(f"Copied seed DB from {BUNDLED_DB_PATH} to {TEMPLATES_DB_PATH}")

    conn = _connect()
    try:
        cur = conn.cursor()

        # Check which tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row['name'] for row in cur.fetchall()}

        if 'snippets' in tables and 'templates' not in tables:
            # Migrate: rename + add columns
            logging.info("Migrating snippets -> templates...")
            cur.execute("ALTER TABLE snippets RENAME TO templates")
            cur.execute("ALTER TABLE templates ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE templates ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'")
            cur.execute("ALTER TABLE templates ADD COLUMN created_at TEXT")
            cur.execute("ALTER TABLE templates ADD COLUMN updated_at TEXT")
            cur.execute("UPDATE templates SET created_at = datetime('now'), updated_at = datetime('now')")
            cur.execute("""
                UPDATE templates
                SET name = CASE
                    WHEN LENGTH(description) <= 80 THEN description
                    ELSE SUBSTR(description, 1, 77) || '...'
                END
            """)
            conn.commit()
            migrated = True
            logging.info("Migration completed successfully")

        elif 'templates' in tables:
            # Ensure all columns exist (fix partial migration)
            cur.execute("PRAGMA table_info(templates)")
            columns = {row['name'] for row in cur.fetchall()}
            added = False
            for col, default in [('name', "''"), ('tags', "'[]'"),
                                 ('created_at', None), ('updated_at', None)]:
                if col not in columns:
                    default_clause = f" DEFAULT {default}" if default else ""
                    not_null = " NOT NULL" if default else ""
                    cur.execute(f"ALTER TABLE templates ADD COLUMN {col} TEXT{not_null}{default_clause}")
                    added = True
                    logging.info(f"Added missing column: {col}")
            if added:
                cur.execute("UPDATE templates SET created_at = datetime('now') WHERE created_at IS NULL")
                cur.execute("UPDATE templates SET updated_at = datetime('now') WHERE updated_at IS NULL")
                cur.execute("""
                    UPDATE templates
                    SET name = CASE
                        WHEN LENGTH(description) <= 80 THEN description
                        ELSE SUBSTR(description, 1, 77) || '...'
                    END
                    WHERE name = '' OR name IS NULL
                """)
                conn.commit()
                migrated = True
                logging.info("Partial migration fix completed")

        if 'templates' not in tables and 'snippets' not in tables:
            # Fresh DB
            cur.execute("""
                CREATE TABLE templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    code TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
            logging.info("Created fresh templates table")
    finally:
        conn.close()
    count = get_count()
    logging.info(f"Storage ready: {count} templates")
    return migrated


def get_count() -> int:
    conn = _connect()
    row = conn.execute("SELECT COUNT(*) AS cnt FROM templates").fetchone()
    conn.close()
    return row['cnt']


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Parse tags from JSON string
    try:
        d['tags'] = json.loads(d.get('tags', '[]'))
    except (json.JSONDecodeError, TypeError):
        d['tags'] = []
    return d


def list_templates(query: Optional[str] = None) -> list[dict]:
    """List templates (without code). Optional substring search by name/description/tags."""
    conn = _connect()
    if query:
        q = f"%{query}%"
        rows = conn.execute(
            "SELECT id, name, description, tags, created_at, updated_at FROM templates "
            "WHERE name LIKE ? OR description LIKE ? OR tags LIKE ? "
            "ORDER BY updated_at DESC",
            (q, q, q)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, description, tags, created_at, updated_at FROM templates "
            "ORDER BY updated_at DESC"
        ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_template(template_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def create_template(name: str, description: str, code: str, tags: list[str] | None = None) -> dict:
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO templates (name, description, code, tags) VALUES (?, ?, ?, ?)",
        (name, description, code, tags_json)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    logging.info(f"Created template id={new_id}")
    return get_template(new_id)


def update_template(template_id: int, name: str = None, description: str = None,
                    code: str = None, tags: list[str] = None) -> Optional[dict]:
    existing = get_template(template_id)
    if existing is None:
        return None

    new_name = name if name is not None else existing['name']
    new_desc = description if description is not None else existing['description']
    new_code = code if code is not None else existing['code']
    new_tags = json.dumps(tags if tags is not None else existing['tags'], ensure_ascii=False)

    conn = _connect()
    conn.execute(
        "UPDATE templates SET name=?, description=?, code=?, tags=?, updated_at=datetime('now') WHERE id=?",
        (new_name, new_desc, new_code, new_tags, template_id)
    )
    conn.commit()
    conn.close()
    logging.info(f"Updated template id={template_id}")
    return get_template(template_id)


def delete_template(template_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    if deleted:
        logging.info(f"Deleted template id={template_id}")
    return deleted


def list_all_for_indexing() -> list[dict]:
    """Return all templates with full data for ChromaDB indexing."""
    conn = _connect()
    rows = conn.execute("SELECT id, name, description, tags, code FROM templates").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]
