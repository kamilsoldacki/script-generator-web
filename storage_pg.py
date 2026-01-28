import json
import os
from typing import Any, Literal

import psycopg2
import psycopg2.extras


def _db_url() -> str:
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("Missing DATABASE_URL env var.")
    return url


def _connect():
    # Render internal Postgres typically works via internal URL without SSL config.
    return psycopg2.connect(_db_url())


def ensure_schema() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS scripts (
      job_id TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      script_type TEXT,
      language TEXT,
      accent TEXT,
      target_minutes INT,
      script_txt TEXT NOT NULL,
      meta_json JSONB NOT NULL,
      outline_json JSONB NOT NULL
    );
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def save_script(
    *,
    job_id: str,
    script_type: str,
    language: str,
    accent: str,
    target_minutes: int,
    script_txt: str,
    meta: dict[str, Any],
    outline: dict[str, Any],
) -> None:
    ensure_schema()
    q = """
    INSERT INTO scripts (job_id, script_type, language, accent, target_minutes, script_txt, meta_json, outline_json)
    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
    ON CONFLICT (job_id) DO UPDATE SET
      script_type=EXCLUDED.script_type,
      language=EXCLUDED.language,
      accent=EXCLUDED.accent,
      target_minutes=EXCLUDED.target_minutes,
      script_txt=EXCLUDED.script_txt,
      meta_json=EXCLUDED.meta_json,
      outline_json=EXCLUDED.outline_json;
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                q,
                (
                    job_id,
                    script_type,
                    language,
                    accent,
                    int(target_minutes),
                    script_txt,
                    json.dumps(meta, ensure_ascii=False),
                    json.dumps(outline, ensure_ascii=False),
                ),
            )
        conn.commit()


def fetch_blob(job_id: str, kind: Literal["txt", "meta", "outline"]) -> tuple[str, str]:
    """
    Returns (content, filename_suggestion).
    """
    ensure_schema()
    col = {"txt": "script_txt", "meta": "meta_json", "outline": "outline_json"}[kind]
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT script_type, {col} AS payload FROM scripts WHERE job_id = %s",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                raise FileNotFoundError("Job not found")
            script_type = (row.get("script_type") or "script").strip() or "script"
            safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in script_type)[:40]
            if kind == "txt":
                return str(row["payload"]), f"{safe}_{job_id}.txt"
            if kind == "meta":
                return json.dumps(row["payload"], ensure_ascii=False, indent=2), f"{safe}_{job_id}.meta.json"
            return json.dumps(row["payload"], ensure_ascii=False, indent=2), f"{safe}_{job_id}.outline.json"

