"""End-to-end test of the playlist dashboard against a real (throwaway) Postgres.

YouTube, Azure Speech and Azure OpenAI are replaced by fakes; everything else is
real: Alembic migrations, db.py, the FastAPI routes, background steps, approvals,
persistence to the database and rebuilding an expired job from it.

    TEST_DATABASE_URL=postgresql://postgres@localhost:55432/subtitle_test pytest tests

The database is wiped first, so its name must contain "test".
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to a throwaway Postgres database")

PW = {"X-App-Password": "test-pw", "X-Reviewer": "Tester"}

PLAYLIST = [
    {"video_id": "vid1", "position": 0, "title": "Gita Ch 1 Ep 1", "published_at": "2026-01-01T00:00:00Z",
     "duration_sec": 1800, "thumbnail_url": "https://i.ytimg.com/vi/vid1/mqdefault.jpg",
     "privacy_status": "public", "yt_caption_flag": False},
    {"video_id": "vid2", "position": 1, "title": "Gita Ch 1 Ep 2", "published_at": "2026-01-08T00:00:00Z",
     "duration_sec": 1700, "thumbnail_url": None, "privacy_status": "public", "yt_caption_flag": True},
    {"video_id": "vid3", "position": 2, "title": "Gita Ch 1 Ep 3", "published_at": "2026-01-15T00:00:00Z",
     "duration_sec": 1600, "thumbnail_url": None, "privacy_status": "public", "yt_caption_flag": False},
]
EP4 = {"video_id": "vid4", "position": 3, "title": "Gita Ch 1 Ep 4", "published_at": "2026-01-22T00:00:00Z",
       "duration_sec": 1500, "thumbnail_url": None, "privacy_status": "unlisted", "yt_caption_flag": False}

CUES = [{"start": 0.5, "end": 2.5, "text": "నమస్కారం", "confidence": 0.9, "review": None},
        {"start": 3.0, "end": 5.0, "text": "భగవద్గీత", "confidence": 0.8, "review": None}]

SRT = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import psycopg

    assert "test" in TEST_DB.rsplit("/", 1)[-1], "TEST_DATABASE_URL's database name must contain 'test'"
    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True,
                   env={**os.environ, "DATABASE_URL_UNPOOLED": TEST_DB}, capture_output=True)

    mp = pytest.MonkeyPatch()
    for key, value in {"DATABASE_URL": TEST_DB, "APP_PASSWORD": "test-pw", "YOUTUBE_API_KEY": "key",
                       "YOUTUBE_PLAYLIST_ID": "https://www.youtube.com/playlist?list=PLtest",
                       "JOBS_DIR": str(tmp_path_factory.mktemp("jobs")), "YOUTUBE_AUTO_SYNC_HOURS": "0"}.items():
        mp.setenv(key, value)
    for name in ("app", "db"):
        sys.modules.pop(name, None)
    import app as app_module

    state = {"playlist": list(PLAYLIST), "download_error": None, "downloads": []}

    def fake_fetch(playlist_id, api_key):
        assert (playlist_id, api_key) == ("PLtest", "key")
        return {"playlist_title": "Bhagavad Gita", "videos": list(state["playlist"]), "skipped": 0}

    def fake_download(video_id, dest_dir, **kw):
        state["downloads"].append(video_id)
        if state["download_error"]:
            raise app_module.youtube.YouTubeError(state["download_error"])
        path = Path(dest_dir) / "upload.m4a"
        path.write_bytes(b"audio")
        return path, "140 · te · Telugu original", {}

    def fake_speech(src, dest_dir):
        out = Path(dest_dir) / "speech.flac"
        out.write_bytes(b"flac")
        return out

    def fake_translate(cues, **kw):
        rows = [{"cue": c.number, "start": c.start, "end": c.end, "telugu": c.telugu,
                 "english": f"English {c.number}", "correction": "", "review": None} for c in cues]
        return {"brief": "Brief.", "rows": rows, "all_items": rows}

    mp.setattr(app_module.youtube, "fetch_playlist", fake_fetch)
    mp.setattr(app_module.youtube, "download_original_audio", fake_download)
    mp.setattr(app_module.transcribe, "make_compact_audio", lambda src, dest: Path(dest).write_bytes(b"mp3"))
    mp.setattr(app_module.transcribe, "make_speech_audio", fake_speech)
    mp.setattr(app_module.transcribe, "transcribe_audio", lambda *a, **k: [dict(c) for c in CUES])
    mp.setattr(app_module.pipeline, "translate_cues", fake_translate)
    mp.setattr(app_module.pipeline, "load_skill_text", lambda _p: "skill")

    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as client:
        yield client, app_module, state
    mp.undo()


def wait_idle(client, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=PW).json()
        if not job["busy"]:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def eventually(fn, timeout=10):
    """The copy to Neon runs right after a step ends, so give it a moment."""
    deadline = time.time() + timeout
    while True:
        result = fn()
        if result or time.time() > deadline:
            return result
        time.sleep(0.1)


def videos(client):
    res = client.get("/api/youtube/videos", headers=PW)
    assert res.status_code == 200, res.text
    return res.json()


def video(client, vid):
    return next(v for v in videos(client)["videos"] if v["video_id"] == vid)


def test_requires_password(env):
    client, _, _ = env
    assert client.get("/api/youtube/videos").status_code == 401


def test_sync_and_categories(env):
    client, _, state = env
    assert videos(client)["videos"] == []
    res = client.post("/api/youtube/sync", headers=PW).json()
    assert res["total"] == 3 and len(res["new"]) == 3
    data = videos(client)
    assert data["playlist_title"] == "Bhagavad Gita"
    cats = {v["video_id"]: v["category"] for v in data["videos"]}
    assert cats == {"vid1": "not_started", "vid2": "confirm", "vid3": "not_started"}
    assert not any(v["is_new"] for v in data["videos"])  # first sync: nothing is "new"

    state["playlist"] = [PLAYLIST[0], PLAYLIST[1], EP4]  # vid3 left the playlist, vid4 arrived
    res = client.post("/api/youtube/sync", headers=PW).json()
    assert res["new"] == ["Gita Ch 1 Ep 4"] and res["removed"] == 1
    data = videos(client)
    assert [v["video_id"] for v in data["videos"]] == ["vid1", "vid2", "vid4"]
    assert [v["is_new"] for v in data["videos"]] == [False, False, True]
    assert data["last_sync"]["new_count"] == 1 and data["last_sync"]["actor"] == "Tester"


def test_generate_two_approvals_and_completion(env):
    client, app_module, state = env
    job = client.post("/api/youtube/videos/vid1/generate", headers=PW)
    assert job.status_code == 200, job.text
    job_id = job.json()["id"]
    job = wait_idle(client, job_id)
    assert job["state"] == "source" and job["youtube"]["workflow_status"] == "telugu_review"
    v = video(client, "vid1")
    assert v["category"] == "review" and v["audio_track"] == "140 · te · Telugu original"
    assert client.post("/api/youtube/videos/vid1/generate", headers=PW).status_code == 409  # already in review

    # Translation needs the Telugu approval first; approval needs a name.
    assert client.post(f"/api/jobs/{job_id}/translate", headers=PW).status_code == 409
    assert client.post(f"/api/jobs/{job_id}/approve", json={"stage": "telugu"}, headers=PW).status_code == 400
    assert client.post(f"/api/jobs/{job_id}/approve", json={"stage": "english", "reviewer": "A"},
                       headers=PW).status_code == 409
    res = client.post(f"/api/jobs/{job_id}/approve", json={"stage": "telugu", "reviewer": "Asha"}, headers=PW)
    assert res.status_code == 200 and res.json()["youtube"]["telugu_approved_by"] == "Asha"
    assert video(client, "vid1")["workflow_status"] == "telugu_approved"

    assert client.post(f"/api/jobs/{job_id}/translate", headers=PW).status_code == 200
    job = wait_idle(client, job_id)
    assert job["state"] == "review" and job["youtube"]["english_fresh"]
    assert video(client, "vid1")["workflow_status"] == "english_review"

    res = client.post(f"/api/jobs/{job_id}/approve", json={"stage": "english", "reviewer": "Ravi"}, headers=PW)
    assert res.status_code == 200
    v = video(client, "vid1")
    assert v["workflow_status"] == "completed" and v["category"] == "generated"
    kinds = {(f["language"], f["format"], f["stage"], f["version"]) for f in v["files"]}
    assert kinds == {("te", "srt", "approved", 1), ("te", "sbv", "approved", 1),
                     ("en", "srt", "approved", 1), ("en", "sbv", "approved", 1)}
    dl = client.get("/api/youtube/videos/vid1/subtitles?lang=te&fmt=srt", headers=PW)
    assert dl.status_code == 200 and "నమస్కారం" in dl.text
    assert 'filename="Gita_Ch_1_Ep_1_telugu_v1.srt"' in dl.headers["content-disposition"]
    dl = client.get("/api/youtube/videos/vid1/subtitles?lang=en&fmt=sbv", headers=PW)
    assert "English 1" in dl.text


def test_edits_withdraw_approvals_and_keep_history(env):
    client, app_module, _ = env
    job_id = video(client, "vid1")["current_job_id"]
    # Saving without a change keeps the approvals.
    rows = client.get(f"/api/jobs/{job_id}/review", headers=PW).json()
    client.put(f"/api/jobs/{job_id}/review", json=[{"cue": r["cue"], "correction": ""} for r in rows], headers=PW)
    assert video(client, "vid1")["workflow_status"] == "completed"

    client.put(f"/api/jobs/{job_id}/review", json=[{"cue": 1, "correction": "Greetings"}], headers=PW)
    v = video(client, "vid1")
    assert v["workflow_status"] == "english_review" and v["english_approved_by"] is None
    assert v["telugu_approved_by"] == "Asha"
    client.post(f"/api/jobs/{job_id}/approve", json={"stage": "english", "reviewer": "Ravi"}, headers=PW)
    v = video(client, "vid1")
    assert v["workflow_status"] == "completed"
    assert {(f["language"], f["format"]): f["version"] for f in v["files"]}[("en", "srt")] == 2
    assert "Greetings" in client.get("/api/youtube/videos/vid1/subtitles?lang=en&fmt=srt", headers=PW).text

    # A Telugu edit withdraws both approvals and makes the English stale.
    source = client.get(f"/api/jobs/{job_id}/source", headers=PW).json()
    source[0]["text"] = "నమస్కారము"
    assert client.put(f"/api/jobs/{job_id}/source", json=source, headers=PW).status_code == 200
    v = video(client, "vid1")
    assert v["workflow_status"] == "telugu_review" and v["telugu_approved_by"] is None
    client.post(f"/api/jobs/{job_id}/approve", json={"stage": "telugu", "reviewer": "Asha"}, headers=PW)
    assert video(client, "vid1")["workflow_status"] == "telugu_approved"  # English must be re-translated
    res = client.post(f"/api/jobs/{job_id}/approve", json={"stage": "english", "reviewer": "Ravi"}, headers=PW)
    assert res.status_code == 409 and "Re-translate" in res.json()["detail"]

    rows = app_module.db.fetchall(
        "SELECT language, format, stage, version, is_current FROM subtitle_files WHERE video_id = 'vid1' "
        "AND stage = 'approved' ORDER BY language, format, version")
    assert [(r["language"], r["format"], r["version"], r["is_current"]) for r in rows if r["format"] == "srt"] == [
        ("en", "srt", 1, False), ("en", "srt", 2, True), ("te", "srt", 1, False), ("te", "srt", 2, True)]
    events = [r["event"] for r in app_module.db.fetchall(
        "SELECT event FROM video_events WHERE video_id = 'vid1' ORDER BY id")]
    assert events.count("approval_cleared") == 2 and "approve_telugu" in events and "generate" in events


def test_expired_job_is_rebuilt_from_database(env):
    client, app_module, state = env
    job_id = video(client, "vid1")["current_job_id"]
    before = client.get(f"/api/jobs/{job_id}/source", headers=PW).json()
    with app_module._LOCK:  # what the 7-day cleanup does
        job = app_module.JOBS.pop(job_id)
        shutil.rmtree(job.dir)
    assert client.get(f"/api/jobs/{job_id}", headers=PW).status_code == 404
    res = client.post("/api/youtube/videos/vid1/open", headers=PW).json()
    assert res == {"job_id": job_id, "restored": True}
    assert client.get(f"/api/jobs/{job_id}/source", headers=PW).json() == before
    assert client.get(f"/api/jobs/{job_id}/review", headers=PW).json()[0]["correction"] == "Greetings"
    deadline = time.time() + 10
    while not client.get(f"/api/jobs/{job_id}", headers=PW).json()["has_audio"]:
        assert time.time() < deadline, "audio was not downloaded again"
        time.sleep(0.1)
    assert state["downloads"].count("vid1") == 2
    assert client.post("/api/youtube/videos/vid1/open", headers=PW).json()["restored"] is False


def test_confirm_original_with_file(env):
    client, _, _ = env
    res = client.post("/api/youtube/videos/vid2/original", data={"decision": "confirmed"}, headers=PW)
    assert res.status_code == 400  # needs a name
    res = client.post("/api/youtube/videos/vid2/original",
                      data={"decision": "confirmed", "reviewer": "Asha", "language": "en"},
                      files={"file": ("orig.srt", SRT.encode(), "text/plain")}, headers=PW)
    assert res.status_code == 200, res.text
    v = video(client, "vid2")
    assert v["category"] == "original" and v["original_by"] == "Asha"
    assert v["files"][0]["source_type"] == "youtube_original"
    dl = client.get("/api/youtube/videos/vid2/subtitles?source=youtube_original&lang=en&fmt=srt", headers=PW)
    assert dl.text == SRT and "_english_original_v1.srt" in dl.headers["content-disposition"]
    assert client.post("/api/youtube/videos/vid2/generate", headers=PW).status_code == 409


def test_failed_download_retry_and_manual_upload(env):
    client, _, state = env
    state["download_error"] = "YouTube refused the download from this server."
    job_id = client.post("/api/youtube/videos/vid4/generate", headers=PW).json()["id"]
    job = wait_idle(client, job_id)
    assert job["state"] == "failed"
    v = video(client, "vid4")
    assert v["category"] == "failed" and "refused" in v["last_error"]

    # "Upload audio instead" links a manual upload to the video.
    res = client.post("/api/jobs", data={"youtube_video_id": "vid4", "formats": "srt"},
                      files={"file": ("ep4.mp3", b"mp3", "audio/mpeg")}, headers=PW)
    assert res.status_code == 202, res.text
    job = wait_idle(client, res.json()["id"])
    assert job["formats"] == ["sbv", "srt"] and job["youtube"]["workflow_status"] == "telugu_review"
    assert video(client, "vid4")["audio_track"] == "manual upload: ep4.mp3"
    drafts = eventually(lambda: (r := client.get("/api/youtube/videos/vid4/subtitles?stage=draft&lang=te&fmt=sbv",
                                                 headers=PW)).status_code == 200 and r)
    assert drafts and "భగవద్గీత" in drafts.text


def test_manual_jobs_are_unchanged(env):
    client, _, _ = env
    res = client.post("/api/jobs", data={"formats": "srt"}, files={"file": ("talk.mp3", b"mp3", "audio/mpeg")},
                      headers=PW)
    job = wait_idle(client, res.json()["id"])
    assert job["youtube"] is None
    assert client.post(f"/api/jobs/{job['id']}/translate", headers=PW).status_code == 200  # no approval gate
    wait_idle(client, job["id"])
    assert client.post(f"/api/jobs/{job['id']}/approve", json={"stage": "telugu", "reviewer": "A"},
                       headers=PW).status_code == 400


def test_schema_version_mismatch_disables_youtube(env):
    client, app_module, _ = env
    app_module.db.execute("UPDATE alembic_version SET version_num = '0000'")
    try:
        app_module.db.check_schema()
        res = client.get("/api/youtube/videos", headers=PW)
        assert res.status_code == 503 and "alembic upgrade head" in res.json()["detail"]
        assert client.get("/healthz").json()["youtube"] is False
    finally:
        app_module.db.execute("UPDATE alembic_version SET version_num = %s", (app_module.db.EXPECTED_REVISION,))
        app_module.db.check_schema()
    assert client.get("/api/youtube/videos", headers=PW).status_code == 200


def test_restart_marks_interrupted_videos_failed(env):
    client, app_module, _ = env
    app_module.db.update_video("vid4", workflow_status="downloading", current_job_id="gone")
    app_module._reconcile_youtube()
    v = video(client, "vid4")
    assert v["workflow_status"] == "failed" and "restarted" in v["last_error"]
