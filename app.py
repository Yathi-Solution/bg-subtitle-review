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

Jobs live on disk under jobs/<id>/ and are deleted RETENTION_HOURS (default 24)
after creation.

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
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import pipeline
import transcribe

load_dotenv()
load_dotenv(".env.local")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = Path(os.getenv("JOBS_DIR") or BASE_DIR / "jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))  # 2 MB subtitles
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(300 * 1024 * 1024)))  # 300 MB audio
RETENTION_HOURS = float(os.getenv("RETENTION_HOURS", "24"))
CLEANUP_INTERVAL_SEC = 15 * 60

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
        self.expires_at: str = data.get("expires_at") or (
            datetime.fromisoformat(self.created_at) + timedelta(hours=RETENTION_HOURS)
        ).isoformat()
        self.options: Dict[str, Any] = data.get("options", {})
        self.cue_count: int = data.get("cue_count", 0)
        self.has_audio: bool = data.get("has_audio", False)
        self.reviewed: bool = data.get("reviewed", False)

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
    """Rebuild the index from disk so jobs survive restarts within their 24 h."""
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
    yield


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
    return {"ok": True, "configured": bool(APP_PASSWORD)}


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


@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
    formats: str = Form("sbv,srt"),
    context_hint: str = Form(""),
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

    kind = "audio" if ext in AUDIO_EXTS else "subtitle"
    job = Job(id=uuid.uuid4().hex, filename=filename, kind=kind, formats=chosen,
              options={"context_hint": context_hint.strip()[:1000]})
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

    if kind == "audio":
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

        _start_step(job, TRANSCRIBING, FAILED, work)
    else:
        upload_path.unlink(missing_ok=True)

    return JSONResponse(status_code=202, content=job.public())


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
def save_source(job_id: str, cues: Any = Body(...), _: None = Depends(require_password)) -> dict:
    job = get_job(job_id)
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish before editing.")
    cleaned = _clean_cues(cues)
    with _LOCK:
        _write_json(job.path("source.json"), cleaned)
        job.cue_count = len(cleaned)
        job.save()
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
    _: None = Depends(require_password),
) -> dict:
    job = get_job(job_id)
    if job.state not in (SOURCE, REVIEW):
        raise HTTPException(status_code=409, detail="The Telugu cues are not ready to translate.")
    cues = _source_cues(job)
    if not cues:
        raise HTTPException(status_code=409, detail="No cues to translate.")
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
def save_review(job_id: str, edits: Any = Body(...), _: None = Depends(require_password)) -> dict:
    """Save human corrections: [{cue, correction}]. The AI draft itself is never overwritten."""
    job = get_job(job_id)
    if job.state in BUSY_STATES:
        raise HTTPException(status_code=409, detail="Wait for the running step to finish before editing.")
    if not isinstance(edits, list):
        raise HTTPException(status_code=400, detail="Expected a list of {cue, correction}.")
    with _LOCK:
        rows = _read_json(job.path("review.json"), [])
        by_cue = {r["cue"]: r for r in rows}
        for e in edits:
            row = by_cue.get(e.get("cue")) if isinstance(e, dict) else None
            if row is None:
                raise HTTPException(status_code=400, detail=f"Unknown cue in edit: {e!r}")
            correction = str(e.get("correction") or "").strip()
            # A "correction" identical to the AI draft is no correction at all.
            row["correction"] = "" if correction == row["english"] else correction
        _write_json(job.path("review.json"), rows)
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
        body = pipeline.format_subtitles(((c["start"], c["end"], c["text"]) for c in items if c["text"]), fmt)
        return text_file(body, f"{job.stem}_telugu.{fmt}")

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


# Serve the web UI (index.html) and static assets at the root.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
