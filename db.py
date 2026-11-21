#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db.py — Neon Postgres access for the YouTube playlist dashboard.

The schema is owned by Alembic (migrations/), applied as a deploy step with
`alembic upgrade head` over the direct (non-pooled) DATABASE_URL_UNPOOLED. The app
never creates or alters tables: at startup it only checks that the database is at
EXPECTED_REVISION, and turns the YouTube features off (503) until it is.

The app itself connects through Neon's pooled DATABASE_URL. Neon closes idle
connections and scales to zero, so pooled connections are checked before use and
the readiness check is retried while it fails.

Without DATABASE_URL the rest of the app works exactly as before.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

EXPECTED_REVISION = "0001"  # keep in step with migrations/versions/
RECHECK_AFTER_SEC = 30

LANGUAGES = ("te", "en")
FORMATS = ("srt", "sbv")
SOURCE_TYPES = ("generated", "youtube_original")
STAGES = ("draft", "approved", "original")
ORIGINAL_STATUSES = ("unknown", "confirmed", "not_original")
WORKFLOW_STATUSES = ("not_started", "queued", "downloading", "transcribing", "telugu_review",
                     "telugu_approved", "translating", "english_review", "completed", "failed")

# Columns the app may set through update_video().
_VIDEO_COLUMNS = {
    "original_status", "original_by", "original_at", "workflow_status",
    "telugu_approved_by", "telugu_approved_at", "english_approved_by", "english_approved_at",
    "current_job_id", "last_error", "audio_track",
}

_pool = None
_pool_lock = threading.Lock()
_state: Dict[str, Any] = {"ok": False, "error": "The database has not been checked yet.", "checked": 0.0}


def configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                os.environ["DATABASE_URL"],
                min_size=0,
                max_size=5,
                timeout=20,
                open=True,
                check=ConnectionPool.check_connection,
                # prepare_threshold=None: no server-side prepared statements, which a
                # transaction-mode pooler (Neon's -pooler host) cannot keep per client.
                kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
            )
        return _pool


def close() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[Any]:
    with _get_pool().connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[Any]:
    with _get_pool().connection() as conn:
        with conn.transaction():
            yield conn


def fetchall(sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetchone(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params: Optional[Sequence[Any]] = None) -> int:
    with connection() as conn:
        return conn.execute(sql, params).rowcount


def _json(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return None if value is None else Jsonb(value)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Readiness (schema version check — never migrates)
# --------------------------------------------------------------------------- #
def check_schema() -> Dict[str, Any]:
    """Is the database reachable and at EXPECTED_REVISION? Never changes it."""
    global _state
    if not configured():
        _state = {"ok": False, "error": "The database is not configured: set DATABASE_URL in .env.",
                  "checked": time.time()}
        return _state
    try:
        import psycopg

        try:
            row = fetchone("SELECT version_num FROM alembic_version")
        except psycopg.errors.UndefinedTable:
            row, missing = None, True
        else:
            missing = False
        revision = row["version_num"] if row else None
        if missing or revision is None:
            error = "The database has no schema yet. Run `alembic upgrade head` (see README)."
        elif revision != EXPECTED_REVISION:
            error = (f"The database schema is at revision {revision}, but this version of the app needs "
                     f"{EXPECTED_REVISION}. Run `alembic upgrade head` (see README).")
        else:
            error = None
        _state = {"ok": error is None, "error": error, "checked": time.time()}
    except Exception as err:
        _state = {"ok": False, "error": f"Could not connect to the database ({type(err).__name__}: "
                                         f"{str(err).strip()[:200]}).", "checked": time.time()}
    return _state


def ready() -> bool:
    """True when the schema check passed; a failed check is retried every 30 s."""
    if not _state["ok"] and configured() and time.time() - _state["checked"] > RECHECK_AFTER_SEC:
        check_schema()
    return bool(_state["ok"])


def problem() -> Optional[str]:
    return None if ready() else _state["error"]


# --------------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------------- #
def upsert_videos(playlist_id: str, videos: List[Dict[str, Any]]) -> List[str]:
    """Insert or refresh playlist metadata; returns the IDs that are new.

    Workflow, approvals and the human caption decision are never touched here.
    """
    if not videos:
        return []
    ids = [v["video_id"] for v in videos]
    with transaction() as conn:
        known = {r["video_id"] for r in conn.execute(
            "SELECT video_id FROM youtube_videos WHERE video_id = ANY(%s)", (ids,)).fetchall()}
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO youtube_videos
                    (video_id, playlist_id, position, title, published_at, duration_sec,
                     thumbnail_url, privacy_status, yt_caption_flag, in_playlist, last_synced_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, now(), now())
                ON CONFLICT (video_id) DO UPDATE SET
                    playlist_id = EXCLUDED.playlist_id,
                    position = EXCLUDED.position,
                    title = EXCLUDED.title,
                    published_at = EXCLUDED.published_at,
                    duration_sec = EXCLUDED.duration_sec,
                    thumbnail_url = EXCLUDED.thumbnail_url,
                    privacy_status = EXCLUDED.privacy_status,
                    yt_caption_flag = EXCLUDED.yt_caption_flag,
                    in_playlist = TRUE,
                    last_synced_at = now(),
                    updated_at = now()
                """,
                [(v["video_id"], playlist_id, v.get("position"), v["title"], v.get("published_at"),
                  v.get("duration_sec"), v.get("thumbnail_url"), v.get("privacy_status"),
                  bool(v.get("yt_caption_flag"))) for v in videos],
            )
    return [i for i in ids if i not in known]


def mark_removed(playlist_id: str, present_ids: Iterable[str]) -> int:
    """Hide videos that are no longer in the playlist (their history is kept)."""
    return execute(
        "UPDATE youtube_videos SET in_playlist = FALSE, updated_at = now() "
        "WHERE playlist_id = %s AND in_playlist AND NOT (video_id = ANY(%s))",
        (playlist_id, list(present_ids)),
    )


def list_videos(playlist_id: str) -> List[Dict[str, Any]]:
    return fetchall(
        "SELECT * FROM youtube_videos WHERE playlist_id = %s AND in_playlist "
        "ORDER BY position NULLS LAST, published_at",
        (playlist_id,),
    )


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    return fetchone("SELECT * FROM youtube_videos WHERE video_id = %s", (video_id,))


def videos_in_status(statuses: Sequence[str]) -> List[Dict[str, Any]]:
    return fetchall("SELECT * FROM youtube_videos WHERE workflow_status = ANY(%s)", (list(statuses),))


def update_video(video_id: str, conn: Any = None, **fields: Any) -> None:
    bad = set(fields) - _VIDEO_COLUMNS
    if bad:
        raise ValueError(f"Not updatable: {', '.join(sorted(bad))}")
    if "workflow_status" in fields and fields["workflow_status"] not in WORKFLOW_STATUSES:
        raise ValueError(f"Unknown workflow status {fields['workflow_status']!r}")
    if not fields:
        return
    sets = ", ".join(f"{k} = %s" for k in fields)  # keys are whitelisted above
    sql = f"UPDATE youtube_videos SET {sets}, updated_at = now() WHERE video_id = %s"
    params = [*fields.values(), video_id]
    if conn is not None:
        conn.execute(sql, params)
    else:
        execute(sql, params)


def log_event(video_id: Optional[str], event: str, actor: Optional[str] = None,
              detail: Optional[Dict[str, Any]] = None, conn: Any = None) -> None:
    sql = "INSERT INTO video_events (video_id, event, actor, detail) VALUES (%s, %s, %s, %s)"
    params = (video_id, event, actor or None, _json(detail))
    if conn is not None:
        conn.execute(sql, params)
    else:
        execute(sql, params)


def log_events(video_ids: Iterable[str], event: str, actor: Optional[str] = None) -> None:
    rows = [(v, event, actor or None) for v in video_ids]
    if not rows:
        return
    with connection() as conn, conn.cursor() as cur:
        cur.executemany("INSERT INTO video_events (video_id, event, actor) VALUES (%s, %s, %s)", rows)


# --------------------------------------------------------------------------- #
# Sync runs
# --------------------------------------------------------------------------- #
def start_sync(playlist_id: str, actor: Optional[str]) -> int:
    row = fetchone("INSERT INTO sync_runs (playlist_id, actor) VALUES (%s, %s) RETURNING id",
                   (playlist_id, actor or None))
    return row["id"]


def finish_sync(run_id: int, *, playlist_title: Optional[str] = None, total: Optional[int] = None,
                new_count: Optional[int] = None, removed_count: Optional[int] = None,
                error: Optional[str] = None) -> None:
    execute(
        "UPDATE sync_runs SET finished_at = now(), playlist_title = %s, total = %s, new_count = %s, "
        "removed_count = %s, error = %s WHERE id = %s",
        (playlist_title, total, new_count, removed_count, error, run_id),
    )


def last_syncs(playlist_id: str) -> Dict[str, Optional[Dict[str, Any]]]:
    """{"latest": the latest finished sync (may have failed), "ok": the latest successful one}."""
    latest = fetchone(
        "SELECT * FROM sync_runs WHERE playlist_id = %s AND finished_at IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1", (playlist_id,))
    ok = latest if latest and not latest["error"] else fetchone(
        "SELECT * FROM sync_runs WHERE playlist_id = %s AND finished_at IS NOT NULL AND error IS NULL "
        "ORDER BY started_at DESC LIMIT 1", (playlist_id,))
    return {"latest": latest, "ok": ok}


# --------------------------------------------------------------------------- #
# Job text (the permanent copy; job folders on disk are a 7-day cache)
# --------------------------------------------------------------------------- #
def save_artifacts(job_id: str, video_id: str, *, meta: Dict[str, Any], source: Any, review: Any,
                   brief: Optional[str], translation_raw: Any) -> None:
    execute(
        """
        INSERT INTO job_artifacts (job_id, video_id, meta, source, review, brief, translation_raw, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (job_id) DO UPDATE SET
            meta = EXCLUDED.meta, source = EXCLUDED.source, review = EXCLUDED.review,
            brief = EXCLUDED.brief, translation_raw = EXCLUDED.translation_raw, updated_at = now()
        """,
        (job_id, video_id, _json(meta), _json(source), _json(review), brief, _json(translation_raw)),
    )


def get_artifacts(job_id: str) -> Optional[Dict[str, Any]]:
    return fetchone("SELECT * FROM job_artifacts WHERE job_id = %s", (job_id,))


# --------------------------------------------------------------------------- #
# Subtitle files: drafts (one current copy), approved and original (versioned)
# --------------------------------------------------------------------------- #
def _check_kind(source_type: str, stage: str, language: str, fmt: str) -> None:
    if source_type not in SOURCE_TYPES or stage not in STAGES or language not in LANGUAGES or fmt not in FORMATS:
        raise ValueError(f"Bad subtitle kind: {source_type}/{stage}/{language}/{fmt}")


def save_draft(video_id: str, job_id: str, language: str, fmt: str, content: str) -> None:
    """Keep one current generated draft per language and format, updated in place."""
    _check_kind("generated", "draft", language, fmt)
    execute(
        """
        INSERT INTO subtitle_files (video_id, source_type, language, format, stage, version, content, sha256, job_id)
        VALUES (%s, 'generated', %s, %s, 'draft', 1, %s, %s, %s)
        ON CONFLICT (video_id, source_type, language, format, stage) WHERE is_current
        DO UPDATE SET content = EXCLUDED.content, sha256 = EXCLUDED.sha256,
                      job_id = EXCLUDED.job_id, created_at = now()
        WHERE subtitle_files.sha256 IS DISTINCT FROM EXCLUDED.sha256
           OR subtitle_files.job_id IS DISTINCT FROM EXCLUDED.job_id
        """,
        (video_id, language, fmt, content, sha256(content), job_id),
    )


def _add_version(conn: Any, video_id: str, source_type: str, stage: str, language: str, fmt: str,
                 content: str, job_id: Optional[str], created_by: Optional[str]) -> int:
    _check_kind(source_type, stage, language, fmt)
    key = (video_id, source_type, language, fmt, stage)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM subtitle_files "
        "WHERE video_id = %s AND source_type = %s AND language = %s AND format = %s AND stage = %s", key).fetchone()
    conn.execute(
        "UPDATE subtitle_files SET is_current = FALSE WHERE video_id = %s AND source_type = %s "
        "AND language = %s AND format = %s AND stage = %s AND is_current", key)
    conn.execute(
        "INSERT INTO subtitle_files (video_id, source_type, language, format, stage, version, content, "
        "sha256, job_id, created_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (video_id, source_type, language, fmt, stage, row["v"], content, sha256(content), job_id, created_by))
    return row["v"]


def approve(video_id: str, language: str, texts: Dict[str, str], job_id: str, reviewer: str) -> None:
    """Freeze the current generated text as a new approved version and record who approved."""
    col = {"te": "telugu", "en": "english"}[language]
    with transaction() as conn:
        conn.execute("SELECT 1 FROM youtube_videos WHERE video_id = %s FOR UPDATE", (video_id,))
        versions = {fmt: _add_version(conn, video_id, "generated", "approved", language, fmt, text, job_id, reviewer)
                    for fmt, text in texts.items()}
        conn.execute(
            f"UPDATE youtube_videos SET {col}_approved_by = %s, {col}_approved_at = now(), updated_at = now() "
            "WHERE video_id = %s", (reviewer, video_id))
        log_event(video_id, f"approve_{col}", reviewer, {"job_id": job_id, "versions": versions}, conn=conn)


def clear_approvals(video_id: str, *, telugu: bool, english: bool, actor: Optional[str], reason: str) -> bool:
    """Drop approvals after an edit. Approved versions stay in subtitle_files as history."""
    sets, conds = [], []
    if telugu:
        sets += ["telugu_approved_by = NULL", "telugu_approved_at = NULL"]
        conds.append("telugu_approved_at IS NOT NULL")
    if english:
        sets += ["english_approved_by = NULL", "english_approved_at = NULL"]
        conds.append("english_approved_at IS NOT NULL")
    if not sets:
        return False
    with transaction() as conn:
        cur = conn.execute(
            f"UPDATE youtube_videos SET {', '.join(sets)}, updated_at = now() "
            f"WHERE video_id = %s AND ({' OR '.join(conds)})", (video_id,))
        if cur.rowcount:
            log_event(video_id, "approval_cleared", actor,
                      {"telugu": telugu, "english": english, "reason": reason}, conn=conn)
        return bool(cur.rowcount)


def set_original(video_id: str, decision: str, reviewer: str, file: Optional[Dict[str, str]] = None) -> None:
    """Record the human decision on YouTube's captions; optionally store the original file."""
    if decision not in ORIGINAL_STATUSES:
        raise ValueError(f"Unknown decision {decision!r}")
    with transaction() as conn:
        conn.execute("SELECT 1 FROM youtube_videos WHERE video_id = %s FOR UPDATE", (video_id,))
        version = None
        if file:
            version = _add_version(conn, video_id, "youtube_original", "original", file["language"],
                                   file["format"], file["content"], None, reviewer)
        conn.execute(
            "UPDATE youtube_videos SET original_status = %s, original_by = %s, original_at = now(), "
            "updated_at = now() WHERE video_id = %s",
            (decision, reviewer if decision != "unknown" else None, video_id))
        log_event(video_id, f"original_{decision}", reviewer,
                  {"file": {k: file[k] for k in ("language", "format")} | {"version": version}} if file else None,
                  conn=conn)


def get_subtitle(video_id: str, source_type: str, stage: str, language: str, fmt: str) -> Optional[Dict[str, Any]]:
    _check_kind(source_type, stage, language, fmt)
    return fetchone(
        "SELECT * FROM subtitle_files WHERE video_id = %s AND source_type = %s AND stage = %s "
        "AND language = %s AND format = %s AND is_current",
        (video_id, source_type, stage, language, fmt))


def current_subtitles(playlist_id: str) -> List[Dict[str, Any]]:
    """Metadata (no content) of every current subtitle file in the playlist."""
    return fetchall(
        "SELECT s.video_id, s.source_type, s.stage, s.language, s.format, s.version, s.created_by, s.created_at "
        "FROM subtitle_files s JOIN youtube_videos v ON v.video_id = s.video_id "
        "WHERE v.playlist_id = %s AND s.is_current ORDER BY s.video_id, s.language, s.format",
        (playlist_id,))
