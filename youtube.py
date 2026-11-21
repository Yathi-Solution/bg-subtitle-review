#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
youtube.py — the YouTube side of the playlist dashboard (no web framework code).

* Playlist metadata comes from the official YouTube Data API v3 with an API key
  (public data only). `contentDetails.caption` is stored as a *hint*: the API only
  says "captions are available", not whether they were uploaded by a person, in
  which language, or whether they are YouTube's automatic captions. A human
  confirms "original" in the dashboard.
* Audio comes from yt-dlp. A video can carry several audio tracks (YouTube's
  auto-dubbing adds AI-dubbed ones), so the track is picked here, not left to
  yt-dlp's default: the track YouTube labels as original, never a dubbed one.
  No bot-check workarounds (proxies, PO-token plugins, cookies) are used; when
  YouTube refuses, the error says so and the operator uploads the audio instead.
"""

from __future__ import annotations

import functools
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
API_TIMEOUT_SEC = 30
PAGE_SIZE = 50                      # the API's maximum for both list calls
AZURE_MAX_SECONDS = 5 * 3600 - 60   # Azure fast transcription: under 5 hours per file

# yt-dlp's language_preference values (yt_dlp/extractor/youtube/_video.py):
# 10 = the track YouTube names "original", 5 = the default track, -1 = other,
# -10 = descriptive audio.
ORIGINAL_PREFERENCE = 10
DESCRIPTIVE_PREFERENCE = -10

ProgressFn = Callable[[str], None]


class YouTubeError(RuntimeError):
    """A failure worth showing to the operator as-is."""


def _noop(_msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Small parsers
# --------------------------------------------------------------------------- #
def playlist_id_from(value: str) -> str:
    """Accept a bare playlist ID or any YouTube URL with ?list=..."""
    value = (value or "").strip()
    if "://" in value or value.startswith("www."):
        query = urllib.parse.urlparse(value if "://" in value else "https://" + value).query
        value = urllib.parse.parse_qs(query).get("list", [""])[0]
    return value


_DURATION_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def parse_duration(iso: Optional[str]) -> Optional[int]:
    """ISO-8601 duration as used by the API ("PT1H2M3S") -> seconds."""
    m = _DURATION_RE.match(iso or "")
    if not m:
        return None
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


# --------------------------------------------------------------------------- #
# YouTube Data API v3 (API key, public data)
# --------------------------------------------------------------------------- #
def _api_get(endpoint: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    # The key goes in a header, not the URL, so it can never leak into an error
    # message, a log line or the dashboard.
    try:
        res = requests.get(f"{API_BASE}/{endpoint}", params=params,
                           headers={"X-Goog-Api-Key": api_key}, timeout=API_TIMEOUT_SEC)
    except requests.RequestException as err:
        raise YouTubeError(f"Could not reach the YouTube Data API ({type(err).__name__}).") from None
    if res.status_code != 200:
        try:
            error = res.json()["error"]
            reason = ", ".join(e.get("reason", "") for e in error.get("errors", []) if e.get("reason"))
            msg = error.get("message", "") + (f" ({reason})" if reason else "")
        except (ValueError, KeyError, TypeError):
            msg = res.text[:300]
        raise YouTubeError(f"YouTube Data API {endpoint} failed ({res.status_code}): {msg}")
    return res.json()


def _best_thumbnail(thumbnails: Dict[str, Any]) -> Optional[str]:
    for size in ("medium", "high", "default"):
        url = (thumbnails.get(size) or {}).get("url")
        if url:
            return url
    return None


def fetch_playlist(playlist_id: str, api_key: str) -> Dict[str, Any]:
    """All public, finished videos of a playlist, in playlist order.

    Quota: 1 unit per page of 50 for each of playlistItems.list and videos.list,
    plus 1 for the playlist title — about 13 units for 300 videos (10,000 a day).
    """
    if not playlist_id or not api_key:
        raise YouTubeError("Set YOUTUBE_API_KEY and YOUTUBE_PLAYLIST_ID.")

    meta = _api_get("playlists", {"part": "snippet", "id": playlist_id, "maxResults": 1}, api_key)
    if not meta.get("items"):
        raise YouTubeError(f"Playlist {playlist_id} was not found, or it is private.")
    playlist_title = meta["items"][0]["snippet"].get("title", "")

    positions: Dict[str, int] = {}
    token: Optional[str] = None
    while True:
        params = {"part": "snippet,contentDetails", "playlistId": playlist_id, "maxResults": PAGE_SIZE}
        if token:
            params["pageToken"] = token
        page = _api_get("playlistItems", params, api_key)
        for item in page.get("items", []):
            vid = (item.get("contentDetails") or {}).get("videoId")
            if vid and vid not in positions:  # a playlist can list a video twice
                positions[vid] = (item.get("snippet") or {}).get("position", len(positions))
        token = page.get("nextPageToken")
        if not token:
            break

    ids = list(positions)
    videos: List[Dict[str, Any]] = []
    for i in range(0, len(ids), PAGE_SIZE):
        page = _api_get("videos", {"part": "snippet,contentDetails,status",
                                   "id": ",".join(ids[i:i + PAGE_SIZE]), "maxResults": PAGE_SIZE}, api_key)
        for v in page.get("items", []):
            snippet, details, status = v.get("snippet", {}), v.get("contentDetails", {}), v.get("status", {})
            if snippet.get("liveBroadcastContent", "none") != "none":
                continue  # live now or an upcoming premiere: no finished audio yet
            if status.get("privacyStatus") == "private":
                continue
            videos.append({
                "video_id": v["id"],
                "position": positions[v["id"]],
                "title": snippet.get("title") or v["id"],
                "published_at": snippet.get("publishedAt"),
                "duration_sec": parse_duration(details.get("duration")),
                "thumbnail_url": _best_thumbnail(snippet.get("thumbnails") or {}),
                "privacy_status": status.get("privacyStatus"),
                # A hint only: "captions are available" (any kind, any language).
                "yt_caption_flag": str(details.get("caption", "")).lower() == "true",
            })
    videos.sort(key=lambda v: v["position"])
    # Deleted and private entries are simply not returned by videos.list.
    return {"playlist_title": playlist_title, "videos": videos, "skipped": len(ids) - len(videos)}


# --------------------------------------------------------------------------- #
# Picking the original audio track
# --------------------------------------------------------------------------- #
def _note(f: Dict[str, Any]) -> str:
    return str(f.get("format_note") or "").lower()


def _is_audio_only(f: Dict[str, Any]) -> bool:
    return f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")


def _is_drc(f: Dict[str, Any]) -> bool:
    # Dynamic-range-compressed copies of a track: same speech, altered loudness.
    return str(f.get("format_id", "")).endswith("-drc") or "drc" in re.split(r"[\s,]+", _note(f))


def _is_descriptive(f: Dict[str, Any]) -> bool:
    return (f.get("language_preference") == DESCRIPTIVE_PREFERENCE
            or str(f.get("language") or "").endswith("-desc") or "descriptive" in _note(f))


def _is_original(f: Dict[str, Any]) -> bool:
    return f.get("language_preference") == ORIGINAL_PREFERENCE or "original" in _note(f)


def _lang_matches(f: Dict[str, Any], lang: str) -> bool:
    code = str(f.get("language") or "").lower()
    return bool(lang) and (code == lang or code.startswith(lang + "-"))


def pick_original_audio(formats: List[Dict[str, Any]], lang: str = "te") -> Dict[str, Any]:
    """Choose the original-language audio-only format from yt-dlp's format list.

    1. Audio-only formats, without DRM, DRC copies or descriptive-audio tracks.
    2. If YouTube marks a track as original, only that track.
    3. Otherwise no track whose note says "dubbed"; and when several languages are
       left, only `lang` — if that is missing too, refuse rather than guess.
    4. Among the copies of that track: undamaged, then m4a, then the highest bitrate.
    """
    lang = (lang or "").strip().lower()
    audio = [f for f in formats
             if _is_audio_only(f) and not f.get("has_drm") and not _is_drc(f) and not _is_descriptive(f)]
    if not audio:
        raise YouTubeError("This video has no downloadable audio-only stream.")

    original = [f for f in audio if _is_original(f)]
    if original:
        pool = original
    else:
        pool = [f for f in audio if "dubbed" not in _note(f)]
        if not pool:
            raise YouTubeError("Only AI-dubbed audio tracks were offered for this video, so it was not "
                               "downloaded. Upload the original audio instead.")
        languages = {str(f.get("language") or "").lower() for f in pool if f.get("language")}
        if len(languages) > 1:
            matching = [f for f in pool if _lang_matches(f, lang)]
            if not matching:
                raise YouTubeError(
                    f"This video has several audio languages ({', '.join(sorted(languages))}) and none is "
                    f"marked as the original or matches YOUTUBE_AUDIO_LANG={lang or '(empty)'}. "
                    "Upload the original audio instead.")
            pool = matching

    if lang and any(_lang_matches(f, lang) for f in pool):
        pool = [f for f in pool if _lang_matches(f, lang)]

    def rank(f: Dict[str, Any]) -> Tuple[bool, bool, bool, float]:
        note = _note(f)
        return ("damaged" not in note, "missing pot" not in note, f.get("ext") == "m4a",
                float(f.get("abr") or f.get("tbr") or 0))

    return max(pool, key=rank)


def track_label(f: Dict[str, Any]) -> str:
    """e.g. '140 · te · Telugu original, medium (original)' — kept for auditing."""
    parts = [str(f.get("format_id") or "?"), str(f.get("language") or "unknown language")]
    if f.get("format_note"):
        parts.append(str(f["format_note"]))
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Download (yt-dlp)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def ytdlp_version() -> Optional[str]:
    try:
        from yt_dlp.version import __version__
        return __version__
    except Exception:  # not installed
        return None


class _Logger:
    """yt-dlp logger that keeps the last error text and drops the chatter."""

    def __init__(self) -> None:
        self.last_error = ""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        self.last_error = msg


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def explain_download_error(message: str) -> str:
    """Turn yt-dlp's error text into something an operator can act on."""
    text = _ANSI_RE.sub("", message or "").replace("ERROR: ", "").strip()
    low = text.lower()
    if ("confirm you" in low and "bot" in low) or "sign in to confirm" in low:
        return ("YouTube refused the download from this server (\"confirm you're not a bot\"). "
                "Use “Upload audio instead” for this video, or retry later.")
    if "private video" in low:
        return "This video is private, so its audio cannot be downloaded."
    if "members-only" in low or "join this channel" in low:
        return "This is a members-only video, so its audio cannot be downloaded."
    if "live event" in low or "premieres in" in low:
        return "This live stream or premiere has not finished yet. Try again once it is a normal video."
    if "video unavailable" in low or ("not available" in low and "format" not in low):
        return "YouTube says this video is unavailable."
    if "http error 429" in low or "too many requests" in low:
        return "YouTube is rate-limiting this server (HTTP 429). Retry later, or upload the audio instead."
    return f"yt-dlp could not download the audio: {text[:400] or 'unknown error'}"


def _ytdlp_options(dest_dir: Path, progress: ProgressFn, logger: _Logger,
                   selector: Callable[[Dict[str, Any]], Any], ffmpeg: Optional[str]) -> Dict[str, Any]:
    step = {"next": 10}

    def hook(d: Dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = int(100 * (d.get("downloaded_bytes") or 0) / total)
                if pct >= step["next"]:
                    step["next"] = pct - pct % 10 + 10
                    progress(f"Downloading audio from YouTube… {pct - pct % 10}%")
        elif d.get("status") == "finished":
            progress("Audio downloaded.")

    opts: Dict[str, Any] = {
        "format": selector,
        "outtmpl": str(Path(dest_dir) / "upload.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": logger,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "overwrites": True,
        "cachedir": False,
    }
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    return opts


def download_original_audio(
    video_id: str,
    dest_dir: Path,
    *,
    progress: ProgressFn = _noop,
    lang: str = "te",
    max_bytes: Optional[int] = None,
    ffmpeg: Optional[str] = None,
    info_only: bool = False,
) -> Tuple[Optional[Path], str, Dict[str, Any]]:
    """Download the original audio track to dest_dir/upload.<ext>.

    Returns (path, track_label, details). With info_only=True nothing is
    downloaded and path is None; details then lists every audio track.
    """
    try:
        import yt_dlp
    except ImportError:
        raise YouTubeError("yt-dlp is not installed on the server (pip install -r requirements.txt).") from None

    chosen: Dict[str, Any] = {}

    def selector(ctx: Dict[str, Any]):
        fmt = pick_original_audio(ctx.get("formats") or [], lang)
        chosen.clear()
        chosen.update(fmt)
        yield fmt

    logger = _Logger()
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = _ytdlp_options(Path(dest_dir), progress, logger, selector, ffmpeg)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            audio_tracks = [
                {"format_id": f.get("format_id"), "language": f.get("language"), "note": f.get("format_note"),
                 "ext": f.get("ext"), "abr": f.get("abr") or f.get("tbr"),
                 "language_preference": f.get("language_preference")}
                for f in info.get("formats") or [] if _is_audio_only(f)
            ]
            details = {"title": info.get("title"), "duration": info.get("duration"),
                       "live_status": info.get("live_status"), "audio_tracks": audio_tracks,
                       "chosen": track_label(chosen) if chosen else None}
            if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                raise YouTubeError("This live stream or premiere has not finished processing yet. Try again later.")
            duration = info.get("duration") or 0
            if duration > AZURE_MAX_SECONDS:
                raise YouTubeError(f"This video is {duration / 3600:.1f} hours long; speech-to-text accepts under 5 hours.")
            if not chosen:
                raise YouTubeError("No original audio track was selected.")
            size = chosen.get("filesize") or chosen.get("filesize_approx")
            if max_bytes and size and size > max_bytes:
                raise YouTubeError(f"The audio track is {size / 1048576:.0f} MB, over the {max_bytes // 1048576} MB limit.")
            label = track_label(chosen)
            if info_only:
                return None, label, details
            progress(f"Audio track: {label}")
            ydl.process_ie_result(info, download=True)
    except YouTubeError:
        raise
    except Exception as err:  # yt_dlp.utils.DownloadError and friends
        raise YouTubeError(explain_download_error(logger.last_error or str(err))) from None

    files = [p for p in Path(dest_dir).glob("upload.*") if p.suffix not in (".part", ".ytdl") and p.stat().st_size > 0]
    if not files:
        raise YouTubeError(explain_download_error(logger.last_error or "the download produced no file"))
    return files[0], label, details
