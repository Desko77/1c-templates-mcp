"""Build templates.db from seed_templates.jsonl.

Standalone script. Does NOT import app.config or app.storage so it can run
during Docker build, outside the runtime environment.

# DDL must stay in sync with app/storage.py (CREATE TABLE templates, lines ~94-103).
# If schema changes there, update here too.

Two-phase import:
    Phase 1: validate the entire JSONL file — collect every error with line
             number. If any error, abort without touching the output DB.
    Phase 2: open a single transaction, batch-INSERT all rows, COMMIT.
             On any SQL error, ROLLBACK and delete the partial output file.

Usage:
    python scripts/build_db_from_jsonl.py --jsonl <in.jsonl> --output <out.db>
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

DDL_TEMPLATES = """
CREATE TABLE templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    code TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jsonl", required=True, help="Input JSONL file.")
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "templates.db"),
        help="Output SQLite DB path (default: <project_root>/templates.db).",
    )
    return p.parse_args()


def validate(jsonl_path: Path) -> tuple[list[dict], list[str]]:
    """Return (valid_rows, errors). If errors non-empty, do not write DB.

    Rows with any validation failure are NOT added to valid_rows.
    """
    rows: list[dict] = []
    errors: list[str] = []
    if not jsonl_path.exists():
        errors.append(f"file not found: {jsonl_path}")
        return rows, errors
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e.msg} at col {e.colno})")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {lineno}: top-level value must be an object")
                continue
            name = obj.get("name", "")
            description = obj.get("description", "")
            code = obj.get("code", "")
            tags = obj.get("tags", [])
            line_errors: list[str] = []
            if not isinstance(name, str) or not name.strip():
                line_errors.append(f"line {lineno}: 'name' must be a non-empty string")
            if not isinstance(description, str) or not description.strip():
                line_errors.append(f"line {lineno}: 'description' must be a non-empty string")
            if not isinstance(code, str) or not code.strip():
                line_errors.append(f"line {lineno}: 'code' must be a non-empty string")
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                line_errors.append(f"line {lineno}: 'tags' must be an array of strings (or omitted)")
            if line_errors:
                errors.extend(line_errors)
                continue
            rows.append({
                "name": name,
                "description": description,
                "code": code,
                "tags": tags,
            })
    return rows, errors


def build(rows: list[dict], out_path: Path) -> None:
    """Write rows to a fresh SQLite DB in a single transaction.

    On any exception during write, the output file is deleted so the caller
    never observes a partial DB.
    """
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    success = False
    try:
        conn.execute(DDL_TEMPLATES)
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO templates (name, description, tags, code) VALUES (?, ?, ?, ?)",
            [
                (
                    r["name"],
                    r["description"],
                    json.dumps(r["tags"], ensure_ascii=False),
                    r["code"],
                )
                for r in rows
            ],
        )
        conn.commit()
        success = True
    finally:
        conn.close()
        if not success and out_path.exists():
            out_path.unlink()


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    out_path = Path(args.output)

    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}", file=sys.stderr)
        return 1

    rows, errors = validate(jsonl_path)
    if errors:
        print(f"Validation failed: {len(errors)} error(s) in {jsonl_path}", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 2
    if not rows:
        print(f"ERROR: JSONL contains no valid templates: {jsonl_path}", file=sys.stderr)
        return 2

    try:
        build(rows, out_path)
    except sqlite3.Error as e:
        print(f"ERROR: SQL failure during import: {e}", file=sys.stderr)
        return 3

    print(f"Built {out_path} with {len(rows)} templates from {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
