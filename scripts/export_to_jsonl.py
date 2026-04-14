"""Export templates from SQLite DB to seed_templates.jsonl.

Standalone script. Does NOT import app.config to avoid its side effects
(FileHandler('app.log'), TRANSFORMERS_CACHE env) during one-off exports.

Usage:
    python scripts/export_to_jsonl.py --db <source.db> --output <seed.jsonl>

The output JSONL contains one JSON object per line with fields:
name, description, tags, code. Omits id/created_at/updated_at so the file
is resilient to merge conflicts (ids are regenerated on import).
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--db",
        default=os.getenv("TEMPLATES_DB_PATH"),
        help="Path to source SQLite DB (or set TEMPLATES_DB_PATH env).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file.",
    )
    args = p.parse_args()
    if not args.db:
        p.error("--db is required (or set TEMPLATES_DB_PATH env)")
    return args


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    out_path = Path(args.output)

    if not db_path.exists():
        print(f"ERROR: DB file not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        cur = conn.execute(
            "SELECT name, description, tags, code FROM templates ORDER BY id ASC"
        )
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in cur:
                try:
                    tags = json.loads(row["tags"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    tags = []
                obj = {
                    "name": row["name"] or "",
                    "description": row["description"] or "",
                    "tags": tags,
                    "code": row["code"] or "",
                }
                f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")
                written += 1
    finally:
        conn.close()

    print(f"Exported {written} templates from {db_path} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
