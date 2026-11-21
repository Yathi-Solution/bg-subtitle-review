#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — FastAPI backend + static web UI for the subtitle review pipeline.

Auth: a single shared password (env APP_PASSWORD). Clients send it as
`X-App-Password`. This is lightweight gatekeeping for a private/team tool,
not enterprise auth — always serve over HTTPS (Render does this for you).

Workflow (one job per upload, human in the loop at every stage):
    audio  -> transcribe -> [human edits Telugu cues] -> translate
    sbv/srt ------------->  [human edits Telugu cues] -> translate
    translate -> [human reviews English] -> optional AI review -> [human decides] -> export

Jobs live on disk under jobs/<id>/ and are deleted RETENTION_HOURS (default 168,
i.e. 7 days) after creation.

YouTube playlist dashboard (optional; needs DATABASE_URL, YOUTUBE_API_KEY and
YOUTUBE_PLAYLIST_ID): videos of one playlist are tracked in Neon Postgres. A video
gets a job like an upload does, but its audio is the original track downloaded
from YouTube. Its jobs need two human approvals (Telugu, then English) before the
video counts as completed, and their text is copied to the database after every
step and edit, so the review survives the 7-day job cleanup (audio is downloaded
again when an old video is reopened).

Endpoints:
    GET    /                                -> web UI
    GET    /healthz                         -> liveness (no auth)
    POST   /api/login                       -> validate password
    GET    /api/jobs                        -> recent (unexpired) jobs
    POST   /api/jobs                        -> upload audio or subtitle, create job
    GET    /api/jobs/{id}                   -> job status + progress log
    DELETE /api/jobs/{id}                   -> delete job and its files now
    GET    /api/jobs/{id}/source            -> Telugu cues (editable)
    PUT    /api/jobs/{id}/source            -> save edited Telugu cues
    GET    /api/jobs/{id}/audio?token=...   -> playback audio for the editor
    POST   /api/jobs/{id}/translate         -> start translation of the current cues
    GET    /api/jobs/{id}/review            -> English review rows
    PUT    /api/jobs/{id}/review            -> save human corrections
    POST   /api/jobs/{id}/ai-review         -> start optional AI review pass
    GET    /api/jobs/{id}/download/{kind}   -> telugu|english (?fmt=srt|sbv), workbook, brief, raw
    POST   /api/jobs/{id}/approve           -> approve Telugu or English (playlist videos only)

    GET    /api/youtube/videos              -> playlist videos with status, counts, last sync
    POST   /api/youtube/sync                -> re-read the playlist from the YouTube Data API
    POST   /api/youtube/videos/{v}/generate -> download original audio, transcribe (also Retry)
    POST   /api/youtube/videos/{v}/original -> human decision on YouTube's captions (+ file)
    POST   /api/youtube/videos/{v}/open     -> the video's job, rebuilt from Neon if expired
    GET    /api/youtube/videos/{v}/subtitles -> approved/original/draft SRT or SBV from Neon
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import urllib.parse
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import db
import pipeline
import transcribe
import youtube

load_dotenv()
load_dotenv(".env.local")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = Path(os.getenv("JOBS_DIR") or BASE_DIR / "jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))  # 2 MB subtitles
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(300 * 1024 * 1024)))  # 300 MB audio
RETENTION_HOURS = float(os.getenv("RETENTION_HOURS", "168"))  # 7 days
CLEANUP_INTERVAL_SEC = 15 * 60

# YouTube playlist dashboard (optional)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
YOUTUBE_PLAYLIST_ID = youtube.playlist_id_from(os.getenv("YOUTUBE_PLAYLIST_ID", ""))
YOUTUBE_AUDIO_LANG = os.getenv("YOUTUBE_AUDIO_LANG", "te").strip().lower()
YOUTUBE_CONTEXT_HINT = os.getenv("YOUTUBE_CONTEXT_HINT", "").strip()[:1000]
YOUTUBE_MAX_CONCURRENT = max(1, int(os.getenv("YOUTUBE_MAX_CONCURRENT", "1")))
YOUTUBE_AUTO_SYNC_HOURS = float(os.getenv("YOUTUBE_AUTO_SYNC_HOURS", "0"))  # 0 = only the Sync button
YT_SLOTS = threading.BoundedSemaphore(YOUTUBE_MAX_CONCURRENT)  # concurrent YouTube downloads
_SYNC_LOCK = threading.Lock()

REVIEW_FLAGS = ("missing", "recovered", "low")  # see transcribe.build_cues

SUBTITLE_EXTS = {".sbv", ".srt"}
AUDIO_EXTS = {".mp3", ".mpeg", ".mpga", ".m4a", ".mp4", ".wav", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac"}
FORMATS = ("sbv", "srt")

# Job states. *ing states are busy; a failed step returns to the last good state
# with `error` set, so the user can fix things and retry.
TRANSCRIBING, SOURCE, TRANSLATING, REVIEW, REVIEWING, FAILED = (
    "transcribing", "source", "translating", "review", "reviewing", "failed",
)
BUSY_STATES = {TRANSCRIBING, TRANSLATING, REVIEWING}

# --------------------------------------------------------------------------- #
# Job store: job.json + data files per job directory; in-memory index
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic, so a crash never leaves half-written JSON


class Job:
    """Metadata in job.json; cues/rows in their own files inside the job directory."""

    FIELDS = ("id", "filename", "kind", "formats", "state", "progress", "error",
              "created_at", "expires_at", "options", "cue_count", "has_audio", "reviewed")

    def __init__(self, **data: Any) -> None:
        self.id: str = data["id"]
        self.filename: str = data["filename"]
        self.kind: str = data["kind"]                       # audio | subtitle
        self.formats: List[str] = data.get("formats") or list(FORMATS)
        self.state: str = data.get("state", SOURCE)
        self.progress: List[str] = data.get("progress", [])
        self.error: Optional[str] = data.get("error")
        self.created_at: str = data.get("created_at") or _now().isoformat()
        self.options: Dict[str, Any] = data.get("options", {})
        self.cue_count: int = data.get("cue_count", 0)
        self.has_audio: bool = data.get("has_audio", False)
        self.reviewed: bool = data.get("reviewed", False)

    @property
    def expires_at(self) -> str:
        # Derived rather than read from job.json, so a change to RETENTION_HOURS also
        # applies to jobs that already exist (a stored value would pin the old window).
        return (datetime.fromisoformat(self.created_at) + timedelta(hours=RETENTION_HOURS)).isoformat()

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def stem(self) -> str:
        return pipeline.safe_stem(self.filename)

    def path(self, name: str) -> Path:
        return self.dir / name

    def save(self) -> None:
        _write_json(self.path("job.json"), {k: getattr(self, k) for k in self.FIELDS})

    def expired(self) -> bool:
        return datetime.fromisoformat(self.expires_at) <= _now()

    def public(self) -> dict:
        data = {k: getattr(self, k) for k in self.FIELDS if k != "options"}
        data["busy"] = self.state in BUSY_STATES
        data["downloads"] = self.downloads()
        data["audio_token"] = audio_token(self.id) if self.has_audio else None
        data["youtube"] = _youtube_public(self)  # None unless the job belongs to a playlist video
        return data

    def summary(self) -> dict:
        keys = ("id", "filename", "kind", "state", "error", "created_at", "expires_at", "cue_count")
        return {k: getattr(self, k) for k in keys}

    def downloads(self) -> List[str]:
        out = []
        if self.path("source.json").exists():
            out += [f"telugu.{f}" for f in self.formats]
        if self.path("review.json").exists():
            out += ["english.srt", "english.sbv", "workbook"]
        if self.path("brief.md").exists():
            out.append("brief")
        if self.path("translation_raw.json").exists():
            out.append("raw")
        return out


JOBS: Dict[str, Job] = {}
_LOCK = threading.RLock()


def load_jobs() -> None:
    """Rebuild the index from disk so jobs survive restarts within their retention window."""
    for job_file in JOBS_DIR.glob("*/job.json"):
        data = _read_json(job_file)
        if not data:
            continue
        job = Job(**data)
        if job.state in BUSY_STATES:  # the worker thread died with the old process
            job.error = "The server restarted while this step was running. Please run it again."
            job.state = {TRANSCRIBING: FAILED, TRANSLATING: SOURCE, REVIEWING: REVIEW}[job.state]
            job.progress.append(f"ERROR: {job.error}")
            job.save()
        JOBS[job.id] = job


def cleanup_expired() -> int:
    """Delete expired jobs, plus orphan directories older than the retention window."""
    removed = 0
    with _LOCK:
        for job in [j for j in JOBS.values() if j.expired() and j.state not in BUSY_STATES]:
            shutil.rmtree(job.dir, ignore_errors=True)
            JOBS.pop(job.id, None)
            removed += 1
        cutoff = time.time() - RETENTION_HOURS * 3600
        for d in JOBS_DIR.iterdir():
            if d.is_dir() and d.name not in JOBS and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed


def _cleanup_loop() -> None:
    while True:
        try:
            cleanup_expired()
        except Exception:  # never let the janitor thread die
            pass
        time.sleep(CLEANUP_INTERVAL_SEC)


def _quiet_client_disconnects(loop: asyncio.AbstractEventLoop) -> None:
    """On Windows, asyncio logs a ConnectionResetError traceback whenever a browser drops
    a connection, which the audio player does each time it seeks. Nothing is wrong, so
    those are not logged; everything else still is."""
    previous = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        if previous:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if sys.platform == "win32":
        _quiet_client_disconnects(asyncio.get_running_loop())
    load_jobs()
    cleanup_expired()
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    if db.configured():
        state = await asyncio.to_thread(db.check_schema)  # checks the version, never migrates
        if state["ok"]:
            await asyncio.to_thread(_reconcile_youtube)
        else:
            print(f"[youtube] dashboard disabled: {state['error']}", file=sys.stderr, flush=True)
    if YOUTUBE_AUTO_SYNC_HOURS > 0:
        threading.Thread(target=_auto_sync_loop, daemon=True).start()
    yield
    db.close()


app = FastAPI(title="Telugu Subtitle Review Pipeline", lifespan=lifespan)


def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if not job or job.expired():
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job


def _start_step(job: Job, busy_state: str, on_error_state: str, work: Callable[[Callable[[str], None]], str]) -> None:
    """Run a long step in a background thread; `work` returns the state to end in."""

    def progress(msg: str) -> None:
        with _LOCK:
            job.progress.append(msg)
            job.save()

    def run() -> None:
        try:
            final_state = work(progress)
            with _LOCK:
                job.state = final_state
                job.save()
        except Exception as err:  # surface failure to the UI
            with _LOCK:
                job.error = str(err)
                job.state = on_error_state
                job.progress.append(f"ERROR: {err}")
                job.save()
        _persist(job)  # playlist videos: copy the result to Neon (no-op for other jobs)

    with _LOCK:
        if job.state in BUSY_STATES:
            raise HTTPException(status_code=409, detail="This job is already running a step.")
        job.state = busy_state
        job.error = None
        job.save()
    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_password(x_app_password: str = Header(default="")) -> None:
    if not APP_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="Server not configured: APP_PASSWORD is not set.",
        )
    if not secrets.compare_digest(x_app_password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password.")


def audio_token(job_id: str) -> str:
    # <audio src> cannot send headers, so the player gets a per-job signed token.
    return hmac.new(APP_PASSWORD.encode(), f"audio:{job_id}".encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# Cue validation (editor saves)
# --------------------------------------------------------------------------- #
def _clean_cues(items: Any, text_key: str = "text") -> List[Dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="Expected a non-empty list of cues.")
    cues = []
    for i, item in enumerate(items, 1):
        try:
            start = pipeline.ts_to_seconds(str(item["start"]))
            end = pipeline.ts_to_seconds(str(item["end"]))
        except (KeyError, ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Cue {i}: invalid start/end time.")
        text = str(item.get(text_key) or "").strip()
        review = item.get("review") if item.get("review") in REVIEW_FLAGS else None
        if text and review == "missing":
            review = None  # the reviewer typed what is said there
        if end <= start:
            raise HTTPException(status_code=400, detail=f"Cue {i}: end time must be after start time.")
        if not text and review != "missing":
            raise HTTPException(status_code=400, detail=f"Cue {i}: text is empty (delete the cue instead).")
        cues.append({
            "start": pipeline.seconds_to_srt_ts(start),
            "end": pipeline.seconds_to_srt_ts(end),
            "text": text,
            "confidence": item["confidence"] if isinstance(item.get("confidence"), (int, float)) else None,
            "review": review,
        })
    cues.sort(key=lambda c: pipeline.ts_to_seconds(c["start"]))
    return cues


def _source_cues(job: Job) -> List[pipeline.Cue]:
    items = [c for c in _read_json(job.path("source.json"), []) if c["text"]]  # skip unfilled blanks
    return [pipeline.Cue(i, c["start"], c["end"], c["text"]) for i, c in enumerate(items, 1)]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "configured": bool(APP_PASSWORD), "youtube": _youtube_problem() is None}


@app.post("/api/login")
def login(_: None = Depends(require_password)) -> dict:
    return {"ok": True, "retention_hours": RETENTION_HOURS}


@app.get("/api/jobs")
def list_jobs(_: None = Depends(require_password)) -> List[dict]:
    jobs = [j for j in JOBS.values() if not j.expired()]
    return [j.summary() for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)]


async def _save_upload(file: UploadFile, dest: Path, limit: int) -> int:
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"File too large (max {limit // (1024 * 1024)} MB).")
            out.write(chunk)
    return size


def _transcribe_work(job: Job, upload_path: Path) -> Callable[[Callable[[str], None]], str]:
    """The audio -> Telugu cues step, shared by uploads and YouTube downloads."""

    def work(progress: Callable[[str], None]) -> str:
        progress("Preparing audio...")
        transcribe.make_compact_audio(upload_path, job.path("audio.mp3"))  # for the editor's player
        speech = transcribe.make_speech_audio(upload_path, job.dir)  # lossless, for Azure
        upload_path.unlink(missing_ok=True)
        with _LOCK:
            job.has_audio = True
        try:
            cues = transcribe.transcribe_audio(speech, context_hint=job.options["context_hint"], progress=progress)
        finally:
            speech.unlink(missing_ok=True)
        source = [{"start": pipeline.seconds_to_srt_ts(c["start"]), "end": pipeline.seconds_to_srt_ts(c["end"]),
                   "text": c["text"], "confidence": c["confidence"], "review": c["review"]} for c in cues]
        _write_json(job.path("source.json"), source)
        with _LOCK:
            job.cue_count = len(source)
        progress("Review the Telugu cues (flagged ones are highlighted), then translate.")
        return SOURCE

    return work


@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
    formats: str = Form("sbv,srt"),
    context_hint: str = Form(""),
    youtube_video_id: str = Form(""),
    x_reviewer: str = Header(default=""),
    _: None = Depends(require_password),
) -> JSONResponse:
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in SUBTITLE_EXTS | AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail="Upload an audio file (.mp3, .m4a, .wav, ...) or a Telugu subtitle file (.sbv/.srt).",
        )
    chosen = [f for f in FORMATS if f in {x.strip().lower() for x in formats.split(",")}]
    if not chosen:
        raise HTTPException(status_code=400, detail="Choose at least one output format (SBV or SRT).")

    # "Upload audio instead" on the playlist dashboard links the upload to a video.
    video_id = youtube_video_id.strip()
    if video_id:
        video = await asyncio.to_thread(_video_for_new_job, video_id)
        chosen = list(FORMATS)  # playlist videos always keep both formats

    kind = "audio" if ext in AUDIO_EXTS else "subtitle"
    options: Dict[str, Any] = {"context_hint": (context_hint.strip() or (YOUTUBE_CONTEXT_HINT if video_id else ""))[:1000]}
    if video_id:
        options["youtube_video_id"] = video_id
    job = Job(id=uuid.uuid4().hex, filename=filename, kind=kind, formats=chosen, options=options)
    job.dir.mkdir(parents=True)
    upload_path = job.path(f"upload{ext}")
    try:
        await _save_upload(file, upload_path, MAX_AUDIO_BYTES if kind == "audio" else MAX_UPLOAD_BYTES)
        if kind == "subtitle":
            try:
                source_text = upload_path.read_bytes().decode("utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text (.sbv/.srt).")
            try:
                cues = pipeline.parse_source_text(source_text)
                if not cues:
                    raise ValueError("No subtitle cues found.")
            except Exception as err:
                raise HTTPException(status_code=400, detail=f"Could not parse subtitle file: {err}")
            _write_json(job.path("source.json"),
                        [{"start": c.start, "end": c.end, "text": c.telugu, "confidence": None, "review": None}
                         for c in cues])
            job.cue_count = len(cues)
            job.progress.append(f"Loaded {len(cues)} cues from {filename}. Review them, then translate.")
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise

    with _LOCK:
        JOBS[job.id] = job
        job.save()

    if video_id:
        try:
            await asyncio.to_thread(
                _link_job, job, video, "transcribing" if kind == "audio" else "telugu_review",
                f"manual upload: {filename}", _reviewer(x_reviewer))
        except Exception:
            with _LOCK:
                JOBS.pop(job.id, None)
            shutil.rmtree(job.dir, ignore_errors=True)
            raise

    if kind == "audio":
        _start_step(job, TRANSCRIBING, FAILED, _transcribe_work(job, upload_path))
    else:
        upload_path.unlink(missing_ok=True)
        if video_id:
            await asyncio.to_thread(_persist, job)

    return JSONResponse(status_code=202, content=await asyncio.to_thread(job.public))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, _: None = Depends(require_password)) -> dict:
    return get_job(job_id).public()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, _: None = Depends(require_password)) -> dict:
    job = get_job(job_id)
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish before deleting.")
    with _LOCK:
        JOBS.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/source")
def get_source(job_id: str, _: None = Depends(require_password)) -> List[dict]:
    job = get_job(job_id)
    if not job.path("source.json").exists():
        raise HTTPException(status_code=409, detail="Transcript is not ready yet.")
    return _read_json(job.path("source.json"), [])


@app.put("/api/jobs/{job_id}/source")
def save_source(job_id: str, cues: Any = Body(...), x_reviewer: str = Header(default=""),
                _: None = Depends(require_password)) -> dict:
    job = get_job(job_id)
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish before editing.")
    cleaned = _clean_cues(cues)
    with _LOCK:
        changed = _read_json(job.path("source.json"), []) != cleaned
        _write_json(job.path("source.json"), cleaned)
        job.cue_count = len(cleaned)
        job.save()
    if changed:  # an approved Telugu (and the English made from it) no longer matches
        _after_edit(job, telugu=True, actor=_reviewer(x_reviewer))
    _persist(job)
    stale = job.path("review.json").exists()
    return {"ok": True, "cue_count": len(cleaned), "translation_stale": stale}


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str, token: str = Query("")) -> FileResponse:
    job = get_job(job_id)
    if not APP_PASSWORD or not secrets.compare_digest(token, audio_token(job.id)):
        raise HTTPException(status_code=401, detail="Invalid token.")
    path = job.path("audio.mp3")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No audio for this job.")
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/jobs/{job_id}/translate")
def start_translation(
    job_id: str,
    chunk_size: int = Form(pipeline.DEFAULT_CHUNK_SIZE),
    overlap: int = Form(pipeline.DEFAULT_OVERLAP),
    model: str = Form(""),
    x_reviewer: str = Header(default=""),
    _: None = Depends(require_password),
) -> dict:
    job = get_job(job_id)
    if job.state not in (SOURCE, REVIEW):
        raise HTTPException(status_code=409, detail="The Telugu cues are not ready to translate.")
    cues = _source_cues(job)
    if not cues:
        raise HTTPException(status_code=409, detail="No cues to translate.")
    telugu_sha = None
    if _video_id(job):  # playlist videos: the Telugu must be approved first
        _require_youtube()
        video = db.get_video(_video_id(job))
        if video and video["current_job_id"] == job.id:
            if not video["telugu_approved_at"]:
                raise HTTPException(status_code=409, detail="Approve the Telugu subtitles before translating.")
            db.clear_approvals(video["video_id"], telugu=False, english=True,
                               actor=_reviewer(x_reviewer), reason="re-translated")
        telugu_sha = _telugu_sha(job)
    chunk_size = max(5, min(200, chunk_size))
    overlap = max(0, min(20, overlap))
    job.options.update(chunk_size=chunk_size, overlap=overlap, model=model.strip() or None)
    back_to = job.state

    def work(progress: Callable[[str], None]) -> str:
        result = pipeline.translate_cues(
            cues,
            skill_text=pipeline.load_skill_text(None),
            model=job.options["model"],
            chunk_size=chunk_size,
            overlap=overlap,
            progress=progress,
        )
        job.path("brief.md").write_text(result["brief"], encoding="utf-8")
        _write_json(job.path("translation_raw.json"), result["all_items"])
        _write_json(job.path("review.json"), result["rows"])
        with _LOCK:
            job.reviewed = False
            if telugu_sha:  # which Telugu this English was made from (approval checks it)
                job.options["translated_from"] = telugu_sha
        progress("Translation done. Review the English, then download or run the optional AI review.")
        return REVIEW

    _start_step(job, TRANSLATING, back_to, work)
    return job.public()


@app.get("/api/jobs/{job_id}/review")
def get_review(job_id: str, _: None = Depends(require_password)) -> List[dict]:
    job = get_job(job_id)
    if not job.path("review.json").exists():
        raise HTTPException(status_code=409, detail="Translation is not ready yet.")
    return _read_json(job.path("review.json"), [])


@app.put("/api/jobs/{job_id}/review")
def save_review(job_id: str, edits: Any = Body(...), x_reviewer: str = Header(default=""),
                _: None = Depends(require_password)) -> dict:
    """Save human corrections: [{cue, correction}]. The AI draft itself is never overwritten."""
    job = get_job(job_id)
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish before editing.")
    if not isinstance(edits, list):
        raise HTTPException(status_code=400, detail="Expected a list of {cue, correction}.")
    changed = False
    with _LOCK:
        rows = _read_json(job.path("review.json"), [])
        by_cue = {r["cue"]: r for r in rows}
        for e in edits:
            row = by_cue.get(e.get("cue")) if isinstance(e, dict) else None
            if row is None:
                raise HTTPException(status_code=400, detail=f"Unknown cue in edit: {e!r}")
            correction = str(e.get("correction") or "").strip()
            # A "correction" identical to the AI draft is no correction at all.
            correction = "" if correction == row["english"] else correction
            changed = changed or correction != (row.get("correction") or "")
            row["correction"] = correction
        _write_json(job.path("review.json"), rows)
    if changed:  # an approved English no longer matches
        _after_edit(job, telugu=False, actor=_reviewer(x_reviewer))
    _persist(job)
    return {"ok": True, "corrected": sum(1 for r in rows if r["correction"])}


@app.post("/api/jobs/{job_id}/ai-review")
def start_ai_review(job_id: str, _: None = Depends(require_password)) -> dict:
    job = get_job(job_id)
    if job.state != REVIEW:
        raise HTTPException(status_code=409, detail="Translate the cues before running the AI review.")

    def work(progress: Callable[[str], None]) -> str:
        rows = _read_json(job.path("review.json"), [])
        brief = job.path("brief.md").read_text(encoding="utf-8") if job.path("brief.md").exists() else ""
        pipeline.review_rows(
            rows,
            skill_text=pipeline.load_skill_text(None),
            brief=brief,
            model=job.options.get("model"),
            progress=progress,
        )
        with _LOCK:
            # Keep corrections the reviewer saved while the AI was running.
            latest = {r["cue"]: r for r in _read_json(job.path("review.json"), [])}
            for r in rows:
                r["correction"] = latest.get(r["cue"], r)["correction"]
            _write_json(job.path("review.json"), rows)
            job.reviewed = True
        return REVIEW

    _start_step(job, REVIEWING, REVIEW, work)
    return job.public()


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(
    job_id: str,
    kind: str,
    fmt: str = Query("srt"),
    _: None = Depends(require_password),
) -> Response:
    job = get_job(job_id)
    if fmt not in FORMATS:
        raise HTTPException(status_code=400, detail="fmt must be srt or sbv.")

    def text_file(body: str, name: str) -> Response:
        return Response(
            content=body.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    if kind == "telugu":
        items = _read_json(job.path("source.json"))
        if not items:
            raise HTTPException(status_code=404, detail="Transcript is not ready yet.")
        return text_file(_telugu_text(items, fmt), f"{job.stem}_telugu.{fmt}")

    rows = _read_json(job.path("review.json"))
    if kind in ("english", "workbook") and not rows:
        raise HTTPException(status_code=404, detail="Translation is not ready yet.")
    if kind == "english":
        return text_file(pipeline.english_subtitles(rows, fmt), f"{job.stem}_english.{fmt}")
    if kind == "workbook":
        path = job.path(f"{job.stem}_master_review.xlsx")
        pipeline.create_workbook(rows, path)  # always reflects the latest corrections
        return FileResponse(path, filename=path.name)
    if kind == "brief" and job.path("brief.md").exists():
        return FileResponse(job.path("brief.md"), filename=f"{job.stem}_discourse_brief.md")
    if kind == "raw" and job.path("translation_raw.json").exists():
        return FileResponse(job.path("translation_raw.json"), filename=f"{job.stem}_translation_raw.json")
    raise HTTPException(status_code=404, detail="Download not available.")


# --------------------------------------------------------------------------- #
# YouTube playlist dashboard
#
# Neon (db.py) is the permanent record of each playlist video: its metadata, its
# workflow status, the two human approvals and every job's text. Job folders on
# disk stay a 7-day working cache; an expired one is rebuilt from Neon on demand.
# --------------------------------------------------------------------------- #
YT_BUSY = ("queued", "downloading", "transcribing", "translating")
YT_IN_REVIEW = ("telugu_review", "telugu_approved", "english_review")


def _log(msg: str) -> None:
    print(f"[youtube] {msg}", file=sys.stderr, flush=True)


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else value


def _reviewer(raw: str) -> str:
    """The X-Reviewer header: the name typed in the browser, URL-encoded (headers are Latin-1)."""
    return urllib.parse.unquote(raw or "").strip()[:100]


def _youtube_problem() -> Optional[str]:
    if not YOUTUBE_API_KEY or not YOUTUBE_PLAYLIST_ID:
        return "The playlist dashboard is not configured: set YOUTUBE_API_KEY and YOUTUBE_PLAYLIST_ID in .env."
    return db.problem()


def _require_youtube() -> None:
    problem = _youtube_problem()
    if problem:
        raise HTTPException(status_code=503, detail=problem)


def _video_id(job: Job) -> Optional[str]:
    return job.options.get("youtube_video_id")


def _telugu_text(items: List[Dict[str, Any]], fmt: str) -> str:
    return pipeline.format_subtitles(((c["start"], c["end"], c["text"]) for c in items if c["text"]), fmt)


def _telugu_sha(job: Job) -> Optional[str]:
    items = _read_json(job.path("source.json"))
    return db.sha256(_telugu_text(items, "srt")) if items else None


def _english_fresh(job: Job) -> bool:
    """The English draft exists and was translated from the Telugu as it is now."""
    return job.path("review.json").exists() and job.options.get("translated_from") == _telugu_sha(job)


def _ffmpeg_or_none() -> Optional[str]:
    try:
        return transcribe.ffmpeg_exe()
    except RuntimeError:
        return None


def _workflow_status(job: Job, video: Dict[str, Any]) -> Optional[str]:
    """Where a playlist video stands, from its current job and its approvals."""
    if job.state == FAILED:
        return "failed"
    if job.state == TRANSCRIBING:
        return None  # queued / downloading / transcribing are set by the step itself
    if job.state == TRANSLATING:
        return "translating"
    if not video["telugu_approved_at"]:
        return "telugu_review"
    if not _english_fresh(job):
        return "telugu_approved"
    if not video["english_approved_at"]:
        return "english_review"
    return "completed"


def _persist(job: Job) -> None:
    """Copy a playlist video's job text to Neon and move its workflow on.

    Runs after every step and edit. Never raises: a database problem must not break
    the job itself; it is logged, and the next save tries again.
    """
    vid = _video_id(job)
    if not vid:
        return
    try:
        if not db.ready():
            raise RuntimeError(db.problem())
        video = db.get_video(vid)
        if not video:
            raise RuntimeError(f"video {vid} is not in the database")
        with _LOCK:
            meta = {k: getattr(job, k) for k in Job.FIELDS}
            meta["progress"] = list(job.progress)
            meta["options"] = dict(job.options)
        source = _read_json(job.path("source.json"))
        review = _read_json(job.path("review.json"))
        brief = job.path("brief.md").read_text(encoding="utf-8") if job.path("brief.md").exists() else None
        db.save_artifacts(job.id, vid, meta=meta, source=source, review=review, brief=brief,
                          translation_raw=_read_json(job.path("translation_raw.json")))
        if video["current_job_id"] != job.id:
            return  # an older job of this video: keep its text, leave the video alone
        for fmt in FORMATS:
            if source:
                db.save_draft(vid, job.id, "te", fmt, _telugu_text(source, fmt))
            if review:
                db.save_draft(vid, job.id, "en", fmt, pipeline.english_subtitles(review, fmt))
        status = _workflow_status(job, video)
        if status:
            db.update_video(vid, workflow_status=status, last_error=job.error)
    except Exception as err:
        _log(f"could not save job {job.id} to the database: {err}")


def _set_workflow(video_id: str, status: str, **fields: Any) -> None:
    try:
        db.update_video(video_id, workflow_status=status, **fields)
    except Exception as err:
        _log(f"could not update {video_id}: {err}")


def _after_edit(job: Job, *, telugu: bool, actor: str) -> None:
    """An edit after approval withdraws it: Telugu edits both approvals, English edits English."""
    vid = _video_id(job)
    if not vid:
        return
    try:
        video = db.get_video(vid) if db.ready() else None
        if video and video["current_job_id"] == job.id:
            db.clear_approvals(vid, telugu=telugu, english=True, actor=actor,
                               reason="Telugu edited" if telugu else "English edited")
    except Exception as err:
        _log(f"could not clear approvals of {vid}: {err}")


def _youtube_public(job: Job) -> Optional[dict]:
    vid = _video_id(job)
    if not vid:
        return None
    info: Dict[str, Any] = {
        "video_id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "audio_track": job.options.get("audio_track"),
        "english_fresh": _english_fresh(job),
    }
    try:
        if not db.ready():
            info["error"] = db.problem()
            return info
        video = db.get_video(vid)
        if video:
            current = video["current_job_id"] == job.id
            info.update({
                "title": video["title"],
                "current": current,
                "workflow_status": (_workflow_status(job, video) if current else None) or video["workflow_status"],
                **{k: _iso(video[k]) for k in ("telugu_approved_by", "telugu_approved_at",
                                               "english_approved_by", "english_approved_at")},
            })
    except Exception as err:
        info["error"] = f"The database is unavailable ({type(err).__name__})."
    return info


def _video_or_404(video_id: str) -> Dict[str, Any]:
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="This video is not in the playlist database. Sync first.")
    return video


def _video_for_new_job(video_id: str) -> Dict[str, Any]:
    """The video, if a new job may start for it (Generate, Retry, Upload audio instead)."""
    _require_youtube()
    video = _video_or_404(video_id)
    job = JOBS.get(video["current_job_id"] or "")
    alive = bool(job and not job.expired())
    if alive and job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="This video is already being processed.")
    if video["original_status"] == "confirmed":
        raise HTTPException(status_code=409, detail="This video's YouTube captions are confirmed as original.")
    stuck = video["workflow_status"] in YT_BUSY and not (alive and job.state in BUSY_STATES)
    if video["workflow_status"] not in ("not_started", "failed") and not stuck:
        raise HTTPException(status_code=409, detail="This video already has subtitles in review or completed. "
                                                    "Open it from the dashboard instead.")
    return video


def _link_job(job: Job, video: Dict[str, Any], status: str, audio_track: Optional[str], actor: str) -> None:
    """Make `job` the video's current job. Approvals belong to a job's text, so they reset;
    earlier approved versions stay in subtitle_files."""
    with db.transaction() as conn:
        db.update_video(video["video_id"], conn=conn, current_job_id=job.id, workflow_status=status,
                        last_error=None, audio_track=audio_track,
                        telugu_approved_by=None, telugu_approved_at=None,
                        english_approved_by=None, english_approved_at=None)
        db.log_event(video["video_id"], "generate" if audio_track is None else "upload", actor,
                     {"job_id": job.id, "audio_track": audio_track}, conn=conn)


def _job_filename(title: str, video_id: str) -> str:
    clean = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", title).strip() or video_id
    return f"{clean[:120]} [{video_id}].m4a"


def _youtube_work(job: Job, video_id: str) -> Callable[[Callable[[str], None]], str]:
    """Download the original audio (one download at a time), then transcribe as for an upload."""

    def work(progress: Callable[[str], None]) -> str:
        if not YT_SLOTS.acquire(blocking=False):
            progress("Waiting for another YouTube download to finish...")
            YT_SLOTS.acquire()
        try:
            _set_workflow(video_id, "downloading")
            progress("Fetching the original audio track from YouTube...")
            path, label, _ = youtube.download_original_audio(
                video_id, job.dir, progress=progress, lang=YOUTUBE_AUDIO_LANG,
                max_bytes=MAX_AUDIO_BYTES, ffmpeg=_ffmpeg_or_none())
        finally:
            YT_SLOTS.release()
        with _LOCK:
            job.options["audio_track"] = label
        _set_workflow(video_id, "transcribing", audio_track=label)
        return _transcribe_work(job, path)(progress)

    return work


def _refetch_audio(job: Job, video_id: str) -> None:
    """A rebuilt job has its text but no audio: download it again for the player."""
    try:
        with YT_SLOTS:
            path, _label, _ = youtube.download_original_audio(
                video_id, job.dir, lang=YOUTUBE_AUDIO_LANG, max_bytes=MAX_AUDIO_BYTES, ffmpeg=_ffmpeg_or_none())
        transcribe.make_compact_audio(path, job.path("audio.mp3"))
        path.unlink(missing_ok=True)
        with _LOCK:
            job.has_audio = True
            job.progress.append("Audio downloaded again for the player.")
            job.save()
    except Exception as err:
        with _LOCK:
            job.progress.append(f"Could not download the audio again ({err}). Editing works; cue playback does not.")
            job.save()


def _restore_job(art: Dict[str, Any], video_id: str) -> Job:
    """Rebuild an expired job folder from its copy in Neon, with a fresh retention window."""
    meta = dict(art["meta"])
    with _LOCK:
        old = JOBS.pop(meta["id"], None)
        if old:
            shutil.rmtree(old.dir, ignore_errors=True)
        state = meta.get("state") or SOURCE
        if state in BUSY_STATES:
            state = {TRANSCRIBING: FAILED, TRANSLATING: SOURCE, REVIEWING: REVIEW}[state]
        job = Job(**{**meta, "state": state, "created_at": _now().isoformat(), "has_audio": False})
        job.dir.mkdir(parents=True, exist_ok=True)
        if art.get("source") is not None:
            _write_json(job.path("source.json"), art["source"])
        if art.get("review") is not None:
            _write_json(job.path("review.json"), art["review"])
        if art.get("brief"):
            job.path("brief.md").write_text(art["brief"], encoding="utf-8")
        if art.get("translation_raw") is not None:
            _write_json(job.path("translation_raw.json"), art["translation_raw"])
        job.progress.append("Rebuilt from the database after the job folder expired.")
        job.save()
        JOBS[job.id] = job
    threading.Thread(target=_refetch_audio, args=(job, video_id), daemon=True).start()
    return job


def _category(video: Dict[str, Any], status: str, busy: bool) -> str:
    """The dashboard's filter bucket. Only two approvals or a human-confirmed original count as done."""
    if video["original_status"] == "confirmed":
        return "original"
    if status == "completed":
        return "generated"
    if busy or status in YT_BUSY:
        return "progress"
    if status in YT_IN_REVIEW:
        return "review"
    if status == "failed":
        return "failed"
    if video["yt_caption_flag"] and video["original_status"] == "unknown":
        return "confirm"
    return "not_started"


def _video_public(video: Dict[str, Any], files: List[dict], last_ok: Optional[Dict[str, Any]]) -> dict:
    job = JOBS.get(video["current_job_id"] or "")
    if job and job.expired():
        job = None
    busy = bool(job and job.state in BUSY_STATES)
    # A live job is the truth; the stored status may lag it by one save (see _start_step).
    status = (_workflow_status(job, video) if job else None) or video["workflow_status"]
    # NEW: found by the latest successful sync (not on the first sync, when everything is new).
    is_new = bool(last_ok and last_ok["new_count"] and last_ok["new_count"] < (last_ok["total"] or 0)
                  and video["first_seen_at"] >= last_ok["started_at"])
    keys = ("video_id", "title", "thumbnail_url", "duration_sec", "position", "yt_caption_flag",
            "original_status", "original_by", "telugu_approved_by", "english_approved_by",
            "current_job_id", "last_error", "audio_track")
    return {
        **{k: video[k] for k in keys},
        **{k: _iso(video[k]) for k in ("published_at", "original_at", "telugu_approved_at", "english_approved_at")},
        "url": f"https://www.youtube.com/watch?v={video['video_id']}",
        "workflow_status": status,
        "category": _category(video, status, busy),
        "busy": busy,
        "job_alive": bool(job),
        "job_state": job.state if job else None,
        "last_progress": job.progress[-1] if busy and job.progress else None,
        "files": files,
        "is_new": is_new,
    }


def _run_sync(actor: str) -> dict:
    run_id = db.start_sync(YOUTUBE_PLAYLIST_ID, actor)
    try:
        data = youtube.fetch_playlist(YOUTUBE_PLAYLIST_ID, YOUTUBE_API_KEY)
        new_ids = db.upsert_videos(YOUTUBE_PLAYLIST_ID, data["videos"])
        removed = db.mark_removed(YOUTUBE_PLAYLIST_ID, [v["video_id"] for v in data["videos"]])
        db.log_events(new_ids, "discovered", actor)
    except Exception as err:
        message = str(err) if isinstance(err, youtube.YouTubeError) else f"Sync failed ({type(err).__name__}: {err})"
        db.finish_sync(run_id, error=message[:1000])
        raise HTTPException(status_code=502, detail=message) from None
    db.finish_sync(run_id, playlist_title=data["playlist_title"], total=len(data["videos"]),
                   new_count=len(new_ids), removed_count=removed)
    titles = {v["video_id"]: v["title"] for v in data["videos"]}
    return {"total": len(data["videos"]), "new": [titles[i] for i in new_ids], "removed": removed,
            "skipped": data["skipped"], "playlist_title": data["playlist_title"]}


def _reconcile_youtube() -> None:
    """After a restart: videos whose step died with the old process become failed (Retry)."""
    try:
        for video in db.videos_in_status(YT_BUSY):
            job = JOBS.get(video["current_job_id"] or "")
            if job is None:
                db.update_video(video["video_id"], workflow_status="failed",
                                last_error="The server restarted while this video was being processed. Click Retry.")
            else:
                _persist(job)  # load_jobs() already moved the job out of its busy state
    except Exception as err:
        _log(f"could not reconcile videos after the restart: {err}")


def _auto_sync_loop() -> None:
    while True:
        time.sleep(YOUTUBE_AUTO_SYNC_HOURS * 3600)
        if _youtube_problem() or not _SYNC_LOCK.acquire(blocking=False):
            continue
        try:
            _run_sync("auto-sync")
        except Exception as err:
            _log(f"auto-sync failed: {getattr(err, 'detail', err)}")
        finally:
            _SYNC_LOCK.release()


@app.get("/api/youtube/videos")
def youtube_videos(_: None = Depends(require_password)) -> dict:
    _require_youtube()
    files: Dict[str, List[dict]] = {}
    for f in db.current_subtitles(YOUTUBE_PLAYLIST_ID):
        if f["stage"] != "draft":  # drafts are downloaded from the job page
            files.setdefault(f["video_id"], []).append({
                "source_type": f["source_type"], "stage": f["stage"], "language": f["language"],
                "format": f["format"], "version": f["version"], "created_by": f["created_by"],
                "created_at": _iso(f["created_at"])})
    syncs = db.last_syncs(YOUTUBE_PLAYLIST_ID)
    videos = [_video_public(v, files.get(v["video_id"], []), syncs["ok"]) for v in db.list_videos(YOUTUBE_PLAYLIST_ID)]
    latest, ok = syncs["latest"], syncs["ok"]
    return {
        "playlist_id": YOUTUBE_PLAYLIST_ID,
        "playlist_url": f"https://www.youtube.com/playlist?list={YOUTUBE_PLAYLIST_ID}",
        "playlist_title": (ok or {}).get("playlist_title"),
        "last_sync": None if not latest else {
            "started_at": _iso(latest["started_at"]), "error": latest["error"], "actor": latest["actor"],
            "total": latest["total"], "new_count": latest["new_count"], "removed_count": latest["removed_count"]},
        "counts": dict(Counter(v["category"] for v in videos)),
        "videos": videos,
        "ytdlp_version": youtube.ytdlp_version(),
        "audio_lang": YOUTUBE_AUDIO_LANG,
    }


@app.post("/api/youtube/sync")
def youtube_sync(x_reviewer: str = Header(default=""), _: None = Depends(require_password)) -> dict:
    _require_youtube()
    if not _SYNC_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A sync is already running.")
    try:
        return _run_sync(_reviewer(x_reviewer) or "dashboard")
    finally:
        _SYNC_LOCK.release()


@app.post("/api/youtube/videos/{video_id}/generate")
def youtube_generate(video_id: str, x_reviewer: str = Header(default=""),
                     _: None = Depends(require_password)) -> dict:
    video = _video_for_new_job(video_id)
    job = Job(id=uuid.uuid4().hex, filename=_job_filename(video["title"], video_id), kind="audio",
              formats=list(FORMATS),
              options={"context_hint": YOUTUBE_CONTEXT_HINT, "youtube_video_id": video_id})
    job.dir.mkdir(parents=True)
    with _LOCK:
        JOBS[job.id] = job
        job.progress.append("Queued: the original audio will be downloaded from YouTube.")
        job.save()
    try:
        _link_job(job, video, "queued", None, _reviewer(x_reviewer))
    except Exception:
        with _LOCK:
            JOBS.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    _start_step(job, TRANSCRIBING, FAILED, _youtube_work(job, video_id))
    return job.public()


@app.post("/api/youtube/videos/{video_id}/original")
def youtube_original(
    video_id: str,
    decision: str = Form(...),
    reviewer: str = Form(""),
    language: str = Form("en"),
    file: Optional[UploadFile] = None,
    _: None = Depends(require_password),
) -> dict:
    """A person's decision on the captions YouTube reports: original, not original, or undo."""
    _require_youtube()
    _video_or_404(video_id)
    decision, reviewer = decision.strip().lower(), reviewer.strip()[:100]
    if decision not in db.ORIGINAL_STATUSES:
        raise HTTPException(status_code=400, detail="decision must be confirmed, not_original or unknown.")
    if decision != "unknown" and not reviewer:
        raise HTTPException(status_code=400, detail="Type your name first.")
    payload = None
    if file is not None and file.filename:
        if decision != "confirmed":
            raise HTTPException(status_code=400, detail="Attach a file only when confirming original captions.")
        ext = Path(file.filename).suffix.lower()
        if ext not in SUBTITLE_EXTS:
            raise HTTPException(status_code=400, detail="Attach the original captions as .srt or .sbv.")
        if language not in db.LANGUAGES:
            raise HTTPException(status_code=400, detail="language must be te or en.")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
        try:
            text = data.decode("utf-8-sig")
            if not pipeline.parse_source_text(text):
                raise ValueError("no subtitle cues found")
        except (UnicodeDecodeError, ValueError) as err:
            raise HTTPException(status_code=400, detail=f"Could not read the subtitle file: {err}")
        payload = {"language": language, "format": ext[1:], "content": text}
    db.set_original(video_id, decision, reviewer, payload)
    return {"ok": True}


@app.post("/api/youtube/videos/{video_id}/open")
def youtube_open(video_id: str, _: None = Depends(require_password)) -> dict:
    """The video's current job, rebuilt from Neon when its folder has expired."""
    _require_youtube()
    video = _video_or_404(video_id)
    job_id = video["current_job_id"]
    if not job_id:
        raise HTTPException(status_code=404, detail="Nothing has been generated for this video yet.")
    with _LOCK:
        job = JOBS.get(job_id)
        if job and not job.expired():
            return {"job_id": job_id, "restored": False}
    art = db.get_artifacts(job_id)
    if not art:
        raise HTTPException(status_code=404, detail="This video's job has expired and no saved copy was found. "
                                                    "Generate it again.")
    return {"job_id": _restore_job(art, video_id).id, "restored": True}


@app.get("/api/youtube/videos/{video_id}/subtitles")
def youtube_subtitle(
    video_id: str,
    source: str = Query("generated"),
    stage: str = Query("approved"),
    lang: str = Query("te"),
    fmt: str = Query("srt"),
    _: None = Depends(require_password),
) -> Response:
    _require_youtube()
    if source == "youtube_original":
        stage = "original"
    try:
        found = db.get_subtitle(video_id, source, stage, lang, fmt)
    except ValueError:
        raise HTTPException(status_code=400, detail="Unknown source, stage, language or format.")
    if not found:
        raise HTTPException(status_code=404, detail="No such subtitle file for this video.")
    video = db.get_video(video_id) or {}
    stem = pipeline.safe_stem(f"{video.get('title') or ''}.x")
    stem = video_id if stem == "subtitle" else stem
    name = (f"{stem}_{'telugu' if lang == 'te' else 'english'}"
            f"{'_original' if source == 'youtube_original' else '_draft' if stage == 'draft' else ''}"
            f"_v{found['version']}.{fmt}")
    return Response(content=found["content"].encode("utf-8"), media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/jobs/{job_id}/approve")
def approve_job(job_id: str, body: Dict[str, Any] = Body(...), _: None = Depends(require_password)) -> dict:
    """Approve Telugu (unlocks translation) or English (completes the video). Playlist videos only."""
    job = get_job(job_id)
    vid = _video_id(job)
    if not vid:
        raise HTTPException(status_code=400, detail="Only jobs of playlist videos have approvals.")
    _require_youtube()
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish.")
    stage, reviewer = body.get("stage"), str(body.get("reviewer") or "").strip()[:100]
    if not reviewer:
        raise HTTPException(status_code=400, detail="Type your name before approving.")
    video = db.get_video(vid)
    if not video or video["current_job_id"] != job.id:
        raise HTTPException(status_code=409, detail="This job is no longer the video's current job.")
    if stage == "telugu":
        items = _read_json(job.path("source.json"))
        if not items:
            raise HTTPException(status_code=409, detail="There are no Telugu cues to approve yet.")
        blank = sum(1 for c in items if not c["text"])
        if blank:
            raise HTTPException(status_code=409, detail=f"{blank} cue(s) have no text yet. Type their words "
                                                        "or delete them before approving.")
        db.approve(vid, "te", {fmt: _telugu_text(items, fmt) for fmt in FORMATS}, job.id, reviewer)
    elif stage == "english":
        if not video["telugu_approved_at"]:
            raise HTTPException(status_code=409, detail="Approve the Telugu subtitles first.")
        rows = _read_json(job.path("review.json"))
        if not rows:
            raise HTTPException(status_code=409, detail="Translate to English first.")
        if not _english_fresh(job):
            raise HTTPException(status_code=409, detail="The English was translated from an earlier version of "
                                                        "the Telugu. Re-translate before approving.")
        db.approve(vid, "en", {fmt: pipeline.english_subtitles(rows, fmt) for fmt in FORMATS}, job.id, reviewer)
    else:
        raise HTTPException(status_code=400, detail="stage must be telugu or english.")
    _persist(job)
    return job.public()


# Serve the web UI (index.html) and static assets at the root.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
