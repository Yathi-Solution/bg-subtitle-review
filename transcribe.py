#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe.py — Telugu audio (.mp3/.m4a/.wav/...) -> timed subtitle cues.

Uses Azure AI Speech fast transcription (te-IN). Measured on a 26-minute discourse,
the raw result has two problems this module corrects:

1. Dropped speech. The job sometimes returns only "." (or nothing) for 20-30 s of
   clear speech; with lossy audio it did so for 2 minutes, so audio is sent as
   lossless 16 kHz (make_speech_audio). Stretches that still contain speech but no
   text are re-sent as short clips; whatever stays empty becomes a blank cue
   flagged for the reviewer, never a silent gap.
2. Word timings. te-IN word timestamps squeeze runs of words into 0.08 s and stretch
   single words over pauses (up to 7 s), so text appeared seconds early. Words are
   re-timed onto the speech actually heard in the audio (see align_words), cues are
   cut at the speaker's real pauses (<= 7 s each, per the Netflix Timed Text Style
   Guide), and then each cue's own audio is transcribed on its own to settle which
   words belong to it (see assign_by_clips).

Each cue carries a review flag for the human editor.

Azure Speech is not offered in South India; Central India (Pune) supports fast
transcription and keeps audio in-region. Configure via environment variables:

    AZURE_SPEECH_KEY=...              key of the Speech resource (Standard S0 tier)
    AZURE_SPEECH_REGION=centralindia  or AZURE_SPEECH_ENDPOINT=https://<name>.cognitiveservices.azure.com
    AZURE_SPEECH_MODEL=               empty = standard model (GA);
                                      MAI-Transcribe-2 = Microsoft's newer model (preview)
"""

from __future__ import annotations

import bisect
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

import pipeline

MAX_CUE_SEC = 7.0               # Netflix: max 7 s per subtitle event (after padding)
MIN_CUE_SEC = 5 / 6             # Netflix: min 5/6 s per subtitle event
TARGET_MIN_SEC = 2.0            # keep merging short phrases until a cue is this long
SENTENCE_PAUSE_SEC = 0.7        # a pause this long ends a cue (once >= TARGET_MIN_SEC)
MAX_MERGE_GAP_SEC = 1.5         # never merge words across a longer silence
LEAD_IN_SEC = 0.1
LEAD_OUT_SEC = 0.25
MAX_SPEECH_SEC = MAX_CUE_SEC - LEAD_IN_SEC - LEAD_OUT_SEC
MAX_CUE_CHARS = 90              # about two subtitle lines of Telugu
# Azure scores groups of phrases (~20 s of speech), in a narrow band that shifts
# between recordings (median 0.81 on one discourse, 0.78 on another). Cues below
# LOW_CONFIDENCE or in the file's weakest tenth, whichever is lower, are flagged.
LOW_CONFIDENCE = 0.77
LOW_CONFIDENCE_SHARE = 0.10
SENTENCE_END = (".", "?", "!", "।", "॥")

# Audio sent to Azure: 16 kHz mono, the rate the recogniser uses. Measured on a
# 26-minute discourse, the 64 kbps 48 kHz MP3 sent before lost 2 minutes of speech
# (including the opening mantra) and 13% of the text; 16 kHz lossless FLAC lost
# nothing. Long recordings whose FLAC would be too large go as 16 kHz 64 kbps MP3,
# which lost nothing either and matched the lossless text 98%.
SAMPLE_RATE = 16000
MAX_LOSSLESS_MB = 300
AUDIO_TYPES = {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg"}

# Speech detection on 10 ms loudness frames.
FRAME_SEC = 0.01
SPEECH_RANGE_DB = 18            # frames this far below a stretch's loud level are pauses
MIN_PAUSE_SEC = 0.25            # shorter dips are stops inside words, not pauses
MIN_VOICE_SEC = 0.1             # shorter blips are clicks, not speech
# Unrecognised speech: a gap between phrases this long, with this much audio within
# GAP_RANGE_DB of the recording's speech level, is re-transcribed.
MIN_GAP_SEC = 2.0
MIN_MISSING_SPEECH_SEC = 2.0
GAP_RANGE_DB = 20
RETRY_SPLIT_SILENCE_SEC = 2.0   # retries send the sound between silences this long...
RETRY_PADS_SEC = (1.0, 3.0)     # ...with this much lead-in/out, one try per value
# Re-timing (align_words). Measured against reference word times on 8 minutes of the
# same discourse: words within 1 s of when they are spoken rose from 80% to 91%.
ALIGN_PRIOR_WEIGHT = 0.3        # pull towards Azure's own word start, per second off...
ALIGN_PRIOR_CAP_SEC = 1.5       # ...capped, because Azure is sometimes 5 s off
EMPTY_SPEECH_COST = 2.0         # per second of speech left without words
TRUSTED_SILENCE_SEC = 1.5       # Azure's word starting right where a silence this long ends...
ONSET_TOLERANCE_SEC = 0.3       # ...within this much is kept there (e.g. speech after a slow chant)
# Pace alone can't tell which side of a pause a word was said on (it predicts a
# stretch's syllables only to ~16%), so each cue's audio is transcribed on its own and
# words go to the cue whose clip heard them. Measured on 8.5 minutes of a discourse
# against reference word times: pauses with a word on the wrong side fell from 20% to
# 2%, and words on screen while spoken rose from 95% to 99.8%. One request per cue;
# Azure allows 600 requests a minute per resource and bills per second of audio.
CLIP_WORKERS = 6
MIN_EVIDENCE_LETTERS = 8        # letters a clip must share with words to move them there

DEFAULT_REGION = "centralindia"
API_VERSION = "2025-10-15"      # adds phrase lists and MAI-Transcribe to fast transcription
REQUEST_TIMEOUT_SEC = 1800

# Azure's display format writes spoken numbers as digits, including the article ఒక
# ("a"): "1 చేతిలో 1 కలాన్ని". Single digits before a word are spelled out again.
NUMBER_WORDS = {"1": "ఒక", "2": "రెండు", "3": "మూడు", "4": "నాలుగు", "5": "ఐదు",
                "6": "ఆరు", "7": "ఏడు", "8": "ఎనిమిది", "9": "తొమ్మిది"}
PERSON_WORDS = {"2": "ఇద్దరు", "3": "ముగ్గురు", "4": "నలుగురు", "5": "ఐదుగురు"}  # "2రు" = ఇద్దరు

ProgressFn = Callable[[str], None]
Segment = Tuple[float, float]


def _noop(_msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Audio conversion
# --------------------------------------------------------------------------- #
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # bundled static ffmpeg (used on hosts without ffmpeg)

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as err:  # pragma: no cover - depends on host
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from err


def _run_ffmpeg(args: Sequence[str]) -> None:
    proc = subprocess.run([ffmpeg_exe(), "-v", "error", "-nostdin", *args], capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip()[-400:]
        raise RuntimeError(f"Could not read audio file: {tail or 'ffmpeg failed'}")


def make_compact_audio(src: Path, dest: Path) -> None:
    """Small mono MP3 streamed by the in-app editor (not sent to Azure: too lossy)."""
    _run_ffmpeg(["-y", "-i", str(src), "-vn", "-ac", "1", "-b:a", "64k", str(dest)])


def make_speech_audio(src: Path, dest_dir: Path) -> Path:
    """The file sent to Azure Speech: 16 kHz mono FLAC, or MP3 if that would be too big."""
    flac = Path(dest_dir) / "speech.flac"
    _run_ffmpeg(["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(flac)])
    if flac.stat().st_size <= MAX_LOSSLESS_MB * 1024 * 1024:
        return flac
    flac.unlink()
    mp3 = Path(dest_dir) / "speech.mp3"
    _run_ffmpeg(["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-b:a", "64k", str(mp3)])
    return mp3


def cut_audio(src: Path, dest: Path, start: float, end: float) -> None:
    """A lossless 16 kHz clip of [start, end] (dest should end in .flac)."""
    _run_ffmpeg(["-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
                 "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(dest)])


def audio_levels(path: Path) -> np.ndarray:
    """Loudness (dBFS) of every 10 ms of the audio, decoded in chunks to bound memory."""
    frame = int(SAMPLE_RATE * FRAME_SEC) * 2  # bytes of 16-bit mono
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    levels, pending = [], b""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        while chunk := proc.stdout.read(frame * 6000):
            pending += chunk
            usable = len(pending) - len(pending) % frame
            samples = np.frombuffer(pending[:usable], dtype="<i2").astype(np.float32) / 32768
            pending = pending[usable:]
            power = (samples.reshape(-1, frame // 2) ** 2).mean(axis=1)
            levels.append(10 * np.log10(power + 1e-12))
        err = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Could not read audio file: {err[-400:] or 'ffmpeg failed'}")
    return np.concatenate(levels) if levels else np.zeros(0, dtype=np.float32)


def speech_regions(levels: np.ndarray, start: float, end: float,
                   min_pause: float = MIN_PAUSE_SEC) -> List[Segment]:
    """Stretches of speech in [start, end]: frames within SPEECH_RANGE_DB of the span's
    loud level, joined across dips shorter than min_pause. Quiet at the very start or
    end of the span is never speech, however short (a clip cut mid-pause)."""
    a, b = int(start / FRAME_SEC), int(end / FRAME_SEC)
    span = levels[a:b]
    if len(span) < 10:
        return [(start, end)]
    loud = span >= np.percentile(span, 90) - SPEECH_RANGE_DB
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(loud.astype(np.int8))) + 1, [len(loud)]])
    regions: List[List[int]] = []
    joined = False
    for i, j in zip(cuts[:-1], cuts[1:]):
        if loud[i] or (0 < i and j < len(loud) and (j - i) * FRAME_SEC < min_pause):
            if joined:
                regions[-1][1] = j
            else:
                regions.append([i, j])
            joined = True
        else:
            joined = False
    out = [(start + i * FRAME_SEC, start + j * FRAME_SEC) for i, j in regions if (j - i) * FRAME_SEC >= MIN_VOICE_SEC]
    return out or [(start, end)]


def _loud_seconds(levels: np.ndarray, start: float, end: float, threshold: float) -> float:
    return float((levels[int(start / FRAME_SEC):int(end / FRAME_SEC)] >= threshold).sum()) * FRAME_SEC


# --------------------------------------------------------------------------- #
# Azure AI Speech fast transcription
# --------------------------------------------------------------------------- #
def _speech_config() -> Tuple[str, str, str]:
    load_dotenv()
    load_dotenv(".env.local")
    key = os.getenv("AZURE_SPEECH_KEY")
    if not key:
        raise RuntimeError(
            "Audio transcription needs an Azure Speech resource. Set AZURE_SPEECH_KEY and "
            "AZURE_SPEECH_REGION (see .env.example)."
        )
    endpoint = (os.getenv("AZURE_SPEECH_ENDPOINT") or "").rstrip("/")
    if not endpoint:
        region = os.getenv("AZURE_SPEECH_REGION") or DEFAULT_REGION
        endpoint = f"https://{region}.api.cognitive.microsoft.com"
    return key, endpoint, (os.getenv("AZURE_SPEECH_MODEL") or "").strip()


def _phrases(context_hint: str) -> List[str]:
    parts = [p.strip() for chunk in context_hint.splitlines() for p in chunk.replace(";", ",").split(",")]
    return [p for p in parts if p][:500]


def build_definition(model: str, context_hint: str = "") -> Dict[str, Any]:
    definition: Dict[str, Any] = {"profanityFilterMode": "None"}
    if model:  # MAI-Transcribe (preview) runs through the same API's enhanced mode
        definition["locales"] = ["te"]
        definition["enhancedMode"] = {
            "enabled": True,
            "model": model,
            "modelOptions": {"timestamps": "word", "transcribeStyle": "verbatim"},
        }
    else:
        definition["locales"] = ["te-IN"]
    phrases = _phrases(context_hint)
    if phrases:
        definition["phraseList"] = {"phrases": phrases}
    return definition


def _error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error") or body
        msg = err.get("message") or err.get("code") or json.dumps(body)[:300]
    except ValueError:
        msg = resp.text[:300] or resp.reason
    hint = {
        401: " Check AZURE_SPEECH_KEY, and that AZURE_SPEECH_REGION matches the resource's region.",
        403: " Check that the Speech resource is on the Standard (S0) tier; fast transcription is not "
             "available on Free (F0).",
        404: " Check AZURE_SPEECH_REGION / AZURE_SPEECH_ENDPOINT (use centralindia; southindia has no Speech).",
    }.get(resp.status_code, "")
    return f"Azure Speech error {resp.status_code}: {str(msg).rstrip('.')}.{hint}"


def request_transcription(audio_path: Path, definition: Dict[str, Any]) -> Dict[str, Any]:
    key, endpoint, _ = _speech_config()
    url = f"{endpoint}/speechtotext/transcriptions:transcribe"
    for attempt in range(4):
        try:
            with open(audio_path, "rb") as audio:
                resp = requests.post(
                    url,
                    params={"api-version": API_VERSION},
                    headers={"Ocp-Apim-Subscription-Key": key},
                    files={
                        "audio": (Path(audio_path).name, audio,
                                  AUDIO_TYPES.get(Path(audio_path).suffix.lower(), "application/octet-stream")),
                        "definition": (None, json.dumps(definition, ensure_ascii=False)),
                    },
                    timeout=REQUEST_TIMEOUT_SEC,
                )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)
            continue
        # Microsoft asks clients to retry 429s: the service may still be scaling up.
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(float(resp.headers.get("Retry-After") or 5 * 2 ** attempt))
            continue
        if not resp.ok:
            raise RuntimeError(_error_message(resp))
        return resp.json()
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Response -> phrases of words
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    start: float
    end: float
    text: str
    confidence: Optional[float]
    phrase_end: bool  # Azure closes a phrase at a pause or sentence end
    recovered: bool = False  # came from re-sending a stretch Azure first missed


@dataclass
class Phrase:
    start: float
    end: float
    words: List[Word]


def _letters(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LMN")


def phrases_from_response(data: Dict[str, Any], offset: float = 0.0, recovered: bool = False) -> List[Phrase]:
    """Phrases with words. A phrase without letters ("." or "...") is how Azure marks
    audio it failed to recognise, so it is left out and found again as missing speech."""
    out: List[Phrase] = []
    for phrase in data.get("phrases") or []:
        if phrase.get("channel", 0) != 0 or not _letters(phrase.get("text") or ""):
            continue
        conf = phrase.get("confidence") or None  # MAI-Transcribe returns 0: no score
        start = offset + phrase["offsetMilliseconds"] / 1000
        end = start + phrase.get("durationMilliseconds", 0) / 1000
        items = [w for w in phrase.get("words") or [] if w["text"].strip()]
        words: List[Word] = []
        if items:
            for k, w in enumerate(items):
                w_start = offset + w["offsetMilliseconds"] / 1000
                w_end = w_start + w.get("durationMilliseconds", 0) / 1000
                words.append(Word(w_start, w_end, w["text"], conf, k == len(items) - 1, recovered))
        else:  # No word timings: spread the phrase over its words by length.
            tokens = (phrase.get("text") or "").split()
            total = sum(len(tok) for tok in tokens) or 1
            t = start
            for k, tok in enumerate(tokens):
                d = (end - start) * len(tok) / total
                words.append(Word(t, t + d, tok, conf, k == len(tokens) - 1, recovered))
                t += d
        if words:
            out.append(Phrase(start, end, words))
    out.sort(key=lambda p: p.start)
    return out


# --------------------------------------------------------------------------- #
# Recovering speech Azure returned no text for
# --------------------------------------------------------------------------- #
def missing_spans(phrases: List[Phrase], levels: np.ndarray, lo: float, hi: float,
                  threshold: float) -> List[Segment]:
    """Gaps in [lo, hi] between recognised phrases that still contain speech."""
    spans, cursor = [], lo
    for start, end in [(p.start, p.end) for p in phrases] + [(hi, hi)]:
        if start - cursor >= MIN_GAP_SEC and _loud_seconds(levels, cursor, start, threshold) >= MIN_MISSING_SPEECH_SEC:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    return spans


def _transcribe_span(audio_path: Path, clip: Path, lo: float, hi: float, keep: Segment,
                     definition: Dict[str, Any]) -> List[Phrase]:
    """Transcribe [lo, hi] on its own; keep the phrases centred inside `keep`."""
    cut_audio(audio_path, clip, lo, hi)
    phrases = phrases_from_response(request_transcription(clip, definition), offset=lo, recovered=True)
    return [p for p in phrases if keep[0] <= (p.start + p.end) / 2 <= keep[1]]


def _overlap(left: List[Word], right: List[Word]) -> int:
    """How many words that end `left` repeat at the start of `right` (one word only
    if it is long enough not to be a coincidence)."""
    for k in range(min(len(left), len(right)), 0, -1):
        if [_letters(w.text) for w in left[-k:]] == [_letters(w.text) for w in right[:k]]:
            if k > 1 or len(_letters(right[0].text)) >= 6:
                return k
    return 0


def _drop_repeats(phrases: List[Phrase]) -> List[Phrase]:
    """A re-sent window overlaps its neighbours, so a recovered phrase can repeat words
    already transcribed on either side; those words are dropped from the recovered one."""
    out: List[Phrase] = []
    for i, p in enumerate(phrases):
        words = p.words
        if words[0].recovered:
            if out:
                words = words[_overlap(out[-1].words, words):]
            if words and i + 1 < len(phrases):
                words = words[:len(words) - _overlap(words, phrases[i + 1].words)]
        if words:
            out.append(Phrase(p.start, p.end, words))
    return out


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def recover_missing(audio_path: Path, phrases: List[Phrase], levels: np.ndarray, duration: float,
                    definition: Dict[str, Any], progress: ProgressFn = _noop,
                    ) -> Tuple[List[Phrase], List[Segment]]:
    """Re-send each stretch of speech that came back without text: first on its own,
    then (for what is still missing) each piece of sound between longer silences with
    some lead-in, since Azure's failures depend on where a clip starts and ends.
    Returns all phrases and the stretches that stayed empty."""
    inside = np.zeros(len(levels), dtype=bool)
    for p in phrases:
        inside[int(p.start / FRAME_SEC):int(p.end / FRAME_SEC)] = True
    speech_level = np.percentile(levels[inside] if inside.any() else levels, 90) if len(levels) else 0.0
    threshold = speech_level - GAP_RANGE_DB
    spans = missing_spans(phrases, levels, 0.0, duration, threshold)
    if not spans:
        return phrases, []
    progress(f"Azure returned no text for {len(spans)} stretch(es) with speech "
             f"({sum(b - a for a, b in spans):.0f} s): {', '.join(f'{_fmt(a)}-{_fmt(b)}' for a, b in spans)}. "
             "Transcribing them again on their own...")
    found: List[Phrase] = []
    still: List[Segment] = []
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.flac"
        for a, b in spans:
            try:
                got = _transcribe_span(audio_path, clip, a, b, (a, b), definition)
                todo = missing_spans(got, levels, a, b, threshold)
                for pad in RETRY_PADS_SEC:
                    left: List[Segment] = []
                    for c, d in todo:
                        for s, e in speech_regions(levels, c, d, RETRY_SPLIT_SILENCE_SEC):
                            more = _transcribe_span(audio_path, clip, max(0.0, s - pad), min(duration, e + pad),
                                                    (s, e), definition)
                            got += more
                            left += missing_spans(more, levels, s, e, threshold)
                    todo = left
                still += todo
            except (RuntimeError, requests.RequestException) as err:
                progress(f"Could not re-transcribe {_fmt(a)}-{_fmt(b)}: {err}")
                got = []
                still.append((a, b))
            found += got
    progress(f"Recovered {len(found)} phrase(s); "
             + (f"{len(still)} stretch(es) still have no text and are marked for the reviewer: "
                f"{', '.join(f'{_fmt(a)}-{_fmt(b)}' for a, b in still)}." if still else "no stretch is left without text."))
    return _drop_repeats(sorted(phrases + found, key=lambda p: p.start)), still


# --------------------------------------------------------------------------- #
# Re-timing words onto the speech heard in the audio
# --------------------------------------------------------------------------- #
def _weight(text: str) -> float:
    """Spoken length of a word, in letters (a digit is read as a short word)."""
    letters = _letters(text)
    return max(1.0, len(letters) + 2.0 * sum(ch.isdigit() for ch in letters))


def _align_unit(words: List[Word], start: float, end: float, levels: np.ndarray, rate: float) -> List[Word]:
    """Split [start, end] into speech regions at its pauses and give each region a run
    of consecutive words (possibly none), choosing the split that best matches each
    region's length to its words at the speaker's rate while staying near Azure's own
    word times. Words then share their region in proportion to their length."""
    regions = speech_regions(levels, start, end)
    n, m = len(words), len(regions)
    cum = np.concatenate([[0.0], np.cumsum([_weight(w.text) for w in words])])
    azure = np.array([w.start for w in words])
    cost = np.full((m + 1, n + 1), np.inf)
    cost[0, 0] = 0.0
    back = np.zeros((m + 1, n + 1), dtype=int)
    for r, (rs, re_) in enumerate(regions, 1):
        span = re_ - rs
        for k2 in range(n + 1):
            best, arg = cost[r - 1, k2] + EMPTY_SPEECH_COST * span, k2
            for k1 in range(k2 - 1, -1, -1):
                if not np.isfinite(cost[r - 1, k1]):
                    continue
                letters = cum[k2] - cum[k1]
                expected = letters / rate
                if expected > 2 * span + 4:
                    break
                fit = (span - expected) ** 2 / (0.5 + 0.25 * span)
                starts = rs + (cum[k1:k2] - cum[k1]) / letters * span
                drift = np.minimum(np.abs(starts - azure[k1:k2]), ALIGN_PRIOR_CAP_SEC).sum()
                c = cost[r - 1, k1] + fit + ALIGN_PRIOR_WEIGHT * drift
                if c < best:
                    best, arg = c, k1
            cost[r, k2], back[r, k2] = best, arg
    if not np.isfinite(cost[m, n]):
        return words
    out: List[Word] = list(words)
    k2 = n
    for r in range(m, 0, -1):
        k1 = back[r, k2]
        if k2 > k1:
            rs, re_ = regions[r - 1]
            scale = (re_ - rs) / (cum[k2] - cum[k1])
            for k in range(k1, k2):
                w = words[k]
                out[k] = Word(rs + (cum[k] - cum[k1]) * scale, rs + (cum[k + 1] - cum[k1]) * scale,
                              w.text, w.confidence, w.phrase_end, w.recovered)
        k2 = k1
    return out


def _split_at_silences(start: float, end: float, words: List[Word],
                       levels: np.ndarray) -> List[Tuple[float, float, List[Word]]]:
    """Split a segment where Azure starts a word right as speech resumes after a long
    silence: those word starts are reliable, and keeping them stops the aligner from
    pulling words across the silence when the pace changes (a slow chant, then talk)."""
    regions = speech_regions(levels, start, end, TRUSTED_SILENCE_SEC)
    pieces, k0, seg_start = [], 0, start
    for (_, silence_start), (silence_end, _) in zip(regions, regions[1:]):
        for k in range(k0 + 1, len(words)):
            if abs(words[k].start - silence_end) <= ONSET_TOLERANCE_SEC and words[k - 1].start < silence_start:
                pieces.append((seg_start, silence_start, words[k0:k]))
                k0, seg_start = k, silence_end
                break
    pieces.append((seg_start, end, words[k0:]))
    return pieces


def _segments(phrases: List[Phrase]) -> List[Tuple[float, float, List[Word]]]:
    units: List[Tuple[float, float, List[Word]]] = []
    for p in phrases:  # back-to-back phrases are one segment split at punctuation
        if units and p.start - units[-1][1] < 0.05:
            units[-1] = (units[-1][0], max(units[-1][1], p.end), units[-1][2] + p.words)
        else:
            units.append((p.start, p.end, list(p.words)))
    return units


def speaking_rate(phrases: List[Phrase], levels: np.ndarray) -> float:
    """Letters per second of speech: the median over segments, because a segment's own
    rate is skewed by words Azure dropped."""
    rates = []
    for start, end, ws in _segments(phrases):
        speech = sum(e - s for s, e in speech_regions(levels, start, end))
        rates.append(sum(_weight(w.text) for w in ws) / max(speech, 0.5))
    return float(np.median(rates)) if rates else 10.0


def align_words(phrases: List[Phrase], levels: np.ndarray) -> List[Word]:
    """Azure's te-IN word timestamps are unreliable (runs squeezed into 0.08 s, single
    words stretched over pauses), but its segment edges hold. Within each segment,
    words are re-timed onto the speech regions found in the audio."""
    units = _segments(phrases)
    if not len(levels):
        return [w for _, _, ws in units for w in ws]
    rate = speaking_rate(phrases, levels)
    pieces = [piece for start, end, ws in units for piece in _split_at_silences(start, end, ws, levels)]
    return [w for start, end, ws in pieces for w in _align_unit(ws, start, end, levels, rate)]


# --------------------------------------------------------------------------- #
# Words -> cues
# --------------------------------------------------------------------------- #
def _is_break(prev: Word, nxt: Word) -> bool:
    return (nxt.start - prev.end >= SENTENCE_PAUSE_SEC or prev.phrase_end
            or prev.text.endswith(SENTENCE_END))


def _best_cut(cur: List[Word]) -> int:
    """Where to split a cue that hit a hard limit: the longest pause, preferring
    phrase or sentence ends, leaving at least a second on the left."""
    best, best_score = len(cur), -1.0
    for i in range(1, len(cur)):
        if cur[i - 1].end - cur[0].start < 1.0:
            continue
        score = cur[i].start - cur[i - 1].end
        if cur[i - 1].phrase_end or cur[i - 1].text.endswith(SENTENCE_END + (",",)):
            score += 0.5
        if score > best_score:
            best, best_score = i, score
    return best


def _too_long(cur: List[Word], w: Word) -> bool:
    return (w.end - cur[0].start > MAX_SPEECH_SEC
            or sum(len(x.text) + 1 for x in cur) + len(w.text) > MAX_CUE_CHARS)


def _cut_if_too_long(cues: List[List[Word]], cur: List[Word], w: Word) -> List[Word]:
    # The words left after one cut can still be too long: cut again.
    while cur and _too_long(cur, w):
        cut = _best_cut(cur)
        cues.append(cur[:cut])
        cur = cur[cut:]
    return cur


def _fits(words: List[Word]) -> bool:
    return (words[-1].end - words[0].start <= MAX_SPEECH_SEC
            and sum(len(w.text) + 1 for w in words) - 1 <= MAX_CUE_CHARS)


def _split_point(words: List[Word]) -> int:
    """Where to cut words that are too long for one cue: at a real pause if there is one
    (so the clips checked against Azure are cut in silence), else at a sentence end, else
    at a comma or phrase end, keeping the halves balanced; each half must fit in a cue if
    any cut allows that."""
    total = words[-1].end - words[0].start or 1.0
    best, best_score = 0, -np.inf
    for i in range(1, len(words)):
        left, right = words[:i], words[i:]
        gap = words[i].start - words[i - 1].end
        score = (1.0 + 2.0 * min(gap, 1.0)) if gap >= MIN_PAUSE_SEC else 0.0
        score += (1.0 if words[i - 1].text.endswith(SENTENCE_END)
                  else 0.5 if words[i - 1].text.endswith(",") or words[i - 1].phrase_end else 0.0)
        score -= 0.3 * abs((left[-1].end - left[0].start) - (right[-1].end - right[0].start)) / total
        score += 10.0 * (_fits(left) + _fits(right))
        if score > best_score:
            best, best_score = i, score
    return best


def join_touching(groups: List[List[Word]]) -> List[List[Word]]:
    """Cues with no pause between them were cut mid-speech (by the length limits, or where
    a clip was cut), sometimes mid-phrase ("ఉండే భగవంతుడి | వల్ల"). Join them, and cut
    again only where too long, at the best break the words offer."""
    out: List[List[Word]] = []
    for g in groups:
        if out and g[0].start - out[-1][-1].end < MIN_PAUSE_SEC:
            todo = [out.pop() + g]
            while todo:
                piece = todo.pop(0)
                if _fits(piece) or len(piece) < 2:
                    out.append(piece)
                else:
                    cut = _split_point(piece)
                    todo[:0] = [piece[:cut], piece[cut:]]
        else:
            out.append(g)
    return out


def group_words(words: List[Word]) -> List[List[Word]]:
    cues: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            if gap > MAX_MERGE_GAP_SEC or (_is_break(cur[-1], w) and cur[-1].end - cur[0].start >= TARGET_MIN_SEC):
                cues.append(cur)
                cur = []
            else:
                cur = _cut_if_too_long(cues, cur, w)
        cur.append(w)
    if cur:
        cues.append(cur)
    return cues


# --------------------------------------------------------------------------- #
# Which cue each word belongs to: each cue's audio transcribed on its own
# --------------------------------------------------------------------------- #
def _clip_spans(groups: List[List[Word]], missing: List[Segment], duration: float) -> List[Segment]:
    """Each cue's clip runs to the middle of the pause on either side; the first and last
    reach the ends of the file, so speech there that the long job missed is heard too.
    Stretches with no recognised text are neighbours too: clips stop where they start."""
    items = sorted([(g[0].start, g[-1].end, k) for k, g in enumerate(groups)]
                   + [(a, b, None) for a, b in missing], key=lambda item: item[0])
    clips: List[Segment] = [(0.0, 0.0)] * len(groups)
    for i, (s, e, k) in enumerate(items):
        if k is None:
            continue
        lo = (items[i - 1][1] + s) / 2 if i else 0.0
        hi = (e + items[i + 1][0]) / 2 if i + 1 < len(items) else duration
        clips[k] = (max(0.0, min(lo, s)), min(duration, max(hi, e)))
    return clips


def _transcribe_clips(audio_path: Path, clips: List[Segment],
                      definition: Dict[str, Any]) -> List[Optional[List[Word]]]:
    """Words heard in each clip (None where the request failed)."""
    with tempfile.TemporaryDirectory() as tmp:
        def one(item: Tuple[int, Segment]) -> Optional[List[Word]]:
            k, (lo, hi) = item
            path = Path(tmp) / f"clip{k}.flac"
            try:
                cut_audio(audio_path, path, lo, hi)
                phrases = phrases_from_response(request_transcription(path, definition), offset=lo, recovered=True)
                return [w for p in phrases for w in p.words]
            except (RuntimeError, requests.RequestException):
                return None
            finally:
                path.unlink(missing_ok=True)

        with ThreadPoolExecutor(CLIP_WORKERS) as pool:
            return list(pool.map(one, enumerate(clips)))


def _keep_in_order(ids: List[Optional[int]]) -> List[Optional[int]]:
    """Words come in clip order, so keep the longest non-decreasing run of clip ids and
    forget the matches that contradict it."""
    tails: List[int] = []
    tail_pos: List[int] = []
    prev: Dict[int, Optional[int]] = {}
    for k, c in enumerate(ids):
        if c is None:
            continue
        i = bisect.bisect_right(tails, c)
        prev[k] = tail_pos[i - 1] if i else None
        if i == len(tails):
            tails.append(c)
            tail_pos.append(k)
        else:
            tails[i], tail_pos[i] = c, k
    keep = set()
    k = tail_pos[-1] if tail_pos else None
    while k is not None:
        keep.add(k)
        k = prev[k]
    return [c if k in keep else None for k, c in enumerate(ids)]


def _match_words(words: List[Word], origin: List[int], heard: List[Optional[List[Word]]]
                 ) -> Tuple[List[Optional[int]], List[Word], List[Word]]:
    """Clip id of each word (None if no clip clearly heard it), and words heard only in
    the first clip before everything else / only in the last clip after everything else.

    A short clip sometimes mishears a word as one said just before it ("భగవద్గీతలో" as
    "భాగవత్ గీత"), so one stray match must not move words. Consecutive words that a
    clip heard (exactly or spelled a little differently) form a run; a run moves its
    words to that clip only with at least MIN_EVIDENCE_LETTERS matching letters, or when
    the clips just before and after it agree word for word (a one-word cue). Weaker runs
    can still confirm the cue a word is already in."""
    mine = [_letters(w.text) for w in words]
    theirs, clip_of, heard_words = [], [], []
    for k, ws in enumerate(heard):
        for w in ws or []:
            theirs.append(_letters(w.text))
            clip_of.append(k)
            heard_words.append(w)
    vote: List[Optional[Tuple[int, int]]] = [None] * len(words)  # (clip, matching letters)
    exact_block = [-1] * len(words)  # which run of exact word matches a word is in
    head: List[Word] = []
    tail: List[Word] = []
    opcodes = difflib.SequenceMatcher(None, mine, theirs, autojunk=False).get_opcodes()
    for n_op, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            for o in range(i2 - i1):
                vote[i1 + o] = (clip_of[j1 + o], len(mine[i1 + o]))
                exact_block[i1 + o] = n_op
        elif tag == "replace":  # spelled differently: vote letter by letter
            a = [(k, ch) for k in range(i1, i2) for ch in mine[k]]
            b = [(clip_of[j], ch) for j in range(j1, j2) for ch in theirs[j]]
            votes: Dict[int, Dict[int, int]] = {}
            matcher = difflib.SequenceMatcher(None, [ch for _, ch in a], [ch for _, ch in b], autojunk=False)
            for block in matcher.get_matching_blocks():
                for o in range(block.size):
                    tally = votes.setdefault(a[block.a + o][0], {})
                    clip = b[block.b + o][0]
                    tally[clip] = tally.get(clip, 0) + 1
            for k, tally in votes.items():
                clip, n = max(tally.items(), key=lambda item: item[1])
                if 2 * n >= len(mine[k]):
                    vote[k] = (clip, n)
        elif tag == "insert" and i1 == 0 and clip_of[j1] == 0:
            head = heard_words[j1:j2]
        elif tag == "insert" and i1 == len(words) and clip_of[j2 - 1] == len(heard) - 1:
            tail = heard_words[j1:j2]

    ids: List[Optional[int]] = [None] * len(words)
    k = 0
    while k < len(words):
        if vote[k] is None:
            k += 1
            continue
        end = k + 1  # the run: consecutive words voting for the same clip
        while end < len(words) and vote[end] is not None and vote[end][0] == vote[k][0]:
            end += 1
        clip = vote[k][0]
        letters = sum(vote[i][1] for i in range(k, end))
        block = exact_block[k]
        between = (block >= 0 and k > 0 and end < len(words) and exact_block[k - 1] == block == exact_block[end]
                   and all(exact_block[i] == block for i in range(k, end)))
        for i in range(k, end):
            if letters >= MIN_EVIDENCE_LETTERS or between or origin[i] == clip:
                ids[i] = clip
        k = end
    return _keep_in_order(ids), head, tail


def assign_by_clips(audio_path: Path, groups: List[List[Word]], missing: List[Segment],
                    levels: np.ndarray, duration: float, rate: float, definition: Dict[str, Any],
                    progress: ProgressFn = _noop) -> List[List[Word]]:
    """Transcribe each cue's audio on its own and move every word into the cue whose
    clip heard it; then re-time the words inside their cue and re-cut overlong cues.
    Words a clip did not hear stay between the cues of their neighbours."""
    if not groups:
        return groups
    words = [w for g in groups for w in g]
    clips = _clip_spans(groups, missing, duration)
    progress(f"Checking which words belong to each of the {len(clips)} subtitles "
             "(transcribing each one's audio on its own)...")
    heard = _transcribe_clips(audio_path, clips, definition)
    failed = sum(1 for h in heard if h is None)
    origin = [k for k, g in enumerate(groups) for _ in g]
    ids, head, tail = _match_words(words, origin, heard)

    # Words no clip clearly heard stay between the cues of their neighbours: in their
    # own cue if that is allowed, else in the clip their estimated start falls in.
    known = [k for k, c in enumerate(ids) if c is not None]
    if not known:
        progress("Could not check the subtitles against their own audio; keeping the timing as estimated.")
        return groups
    for k, c in enumerate(ids):
        if c is not None:
            continue
        i = bisect.bisect_left(known, k)
        lo = ids[known[i - 1]] if i else 0
        hi = ids[known[i]] if i < len(known) else len(clips) - 1
        t = words[k].start
        ids[k] = (origin[k] if lo <= origin[k] <= hi else
                  min(range(lo, hi + 1), key=lambda j: 0.0 if clips[j][0] <= t <= clips[j][1]
                      else min(abs(t - clips[j][0]), abs(t - clips[j][1]))))
        if k and ids[k] < ids[k - 1]:  # keep the words in order
            ids[k] = ids[k - 1]

    placed: List[List[Word]] = [[] for _ in clips]
    for w, c in zip(words, ids):
        placed[c].append(w)
    placed[0][:0] = head
    placed[-1].extend(tail)

    out: List[List[Word]] = []
    for (lo, hi), g in zip(clips, placed):
        if not g:
            continue
        regions = speech_regions(levels, lo, hi)
        g = _align_unit(g, regions[0][0], regions[-1][1], levels, rate)
        # The clip's sound is these words, so the cue covers it to its first and last
        # syllable (re-timing can leave a short tail of the last word outside).
        speech_start, speech_end = regions[0][0], regions[-1][1]
        if speech_start < g[0].start <= speech_start + MAX_MERGE_GAP_SEC:
            g[0] = Word(speech_start, g[0].end, g[0].text, g[0].confidence, g[0].phrase_end, g[0].recovered)
        if speech_end - MAX_MERGE_GAP_SEC <= g[-1].end < speech_end:
            g[-1] = Word(g[-1].start, speech_end, g[-1].text, g[-1].confidence, g[-1].phrase_end, g[-1].recovered)
        cur: List[Word] = []
        for w in g:
            if cur:
                cur = _cut_if_too_long(out, cur, w)
            cur.append(w)
        out.append(cur)
    progress(f"Checked {len(clips) - failed} of {len(clips)} subtitles against their own audio"
             + (f" ({failed} request(s) failed; those keep the estimated timing)" if failed else "")
             + (f"; added {len(head) + len(tail)} word(s) heard only at the start or end of the file"
                if head or tail else "") + ".")
    return out


def pad_cues(cues: List[Segment], duration: float) -> List[Segment]:
    """Add a little lead-in/out without overlapping neighbours; enforce min/max duration."""
    out: List[Segment] = []
    for i, (start, end) in enumerate(cues):
        prev_end = out[-1][1] if out else 0.0
        next_start = cues[i + 1][0] if i + 1 < len(cues) else duration
        s = max(prev_end, start - LEAD_IN_SEC, 0.0)
        e = min(next_start - 0.05, end + LEAD_OUT_SEC, duration, s + MAX_CUE_SEC)
        if e - s < MIN_CUE_SEC:
            e = min(next_start - 0.05, s + MIN_CUE_SEC, duration)
            s = max(prev_end, min(s, e - MIN_CUE_SEC), 0.0)  # no room after: start earlier
        out.append((round(s, 3), round(max(e, s + 0.1), 3)))
    return out


def _placeholder_spans(levels: np.ndarray, start: float, end: float) -> List[Segment]:
    """Subtitle-sized pieces of a stretch with no recognised text, cut at its pauses,
    so the reviewer can play each piece and type what is said."""
    pieces: List[List[float]] = []
    for s, e in speech_regions(levels, start, end):
        if pieces and e - pieces[-1][0] <= MAX_SPEECH_SEC:
            pieces[-1][1] = e
            continue
        parts = max(1, int(np.ceil((e - s) / MAX_SPEECH_SEC)))
        step = (e - s) / parts
        pieces += [[s + k * step, s + (k + 1) * step] for k in range(parts)]
    return [(s, e) for s, e in pieces]


def tidy_text(text: str) -> str:
    text = text.replace("�", "")
    # A zero-width (non-)joiner only belongs right after a virama; Azure sometimes puts
    # one before it ("త‌్రి"), which renders as a broken letter with a dotted circle.
    text = re.sub("(?<!్)[‌‍]", "", text)
    text = re.sub(r"(?<![\w.,])1(?=ట)", "ఒక", text)  # ఒకటి/ఒకటే written "1టి"/"1టే"
    text = re.sub(r"(?<![\w.,])([2-5])రు", lambda m: PERSON_WORDS[m.group(1)], text)
    text = re.sub(r"(?<![\w.,])([1-9])(?=\s+[ఀ-౿])", lambda m: NUMBER_WORDS[m.group(1)], text)
    return " ".join(text.split())


def build_cues(groups: List[List[Word]], phrases: List[Phrase], missing: List[Segment],
               levels: np.ndarray, duration: float) -> List[Dict[str, Any]]:
    """Cues {start, end, text, confidence, review}. review is "missing" (no text was
    recognised: blank cue for the reviewer to fill or delete), "recovered" (text from
    re-sending the stretch), "low" (weak recognition) or None."""
    flat = [w for g in groups for w in g]
    for w, nxt in zip(flat, flat[1:]):  # a number whose noun starts the next cue
        if w.text in NUMBER_WORDS and re.match("[ఀ-౿]", nxt.text):
            w.text = NUMBER_WORDS[w.text]
    items: List[Tuple[float, float, Optional[List[Word]]]] = [(g[0].start, g[-1].end, g) for g in groups]
    for a, b in missing:
        items += [(s, e, None) for s, e in _placeholder_spans(levels, a, b)]
    items.sort(key=lambda item: item[0])
    confs = [p.words[0].confidence for p in phrases if p.words[0].confidence is not None and not p.words[0].recovered]
    low = min(LOW_CONFIDENCE, float(np.percentile(confs, 100 * LOW_CONFIDENCE_SHARE))) if confs else LOW_CONFIDENCE
    cues = []
    for (start, end), (_, _, group) in zip(pad_cues([(s, e) for s, e, _ in items], duration), items):
        if group is None:
            cues.append({"start": start, "end": end, "text": "", "confidence": None, "review": "missing"})
            continue
        scores = [w.confidence for w in group if w.confidence is not None]
        conf = round(min(scores), 3) if scores else None
        review = ("recovered" if any(w.recovered for w in group)
                  else "low" if conf is not None and conf < low else None)
        cues.append({
            "start": start,
            "end": end,
            "text": pipeline.wrap_subtitle(tidy_text(" ".join(w.text for w in group))),
            "confidence": conf,
            "review": review,
        })
    return cues


def transcribe_audio(
    audio_path: Path,
    *,
    context_hint: str = "",
    progress: ProgressFn = _noop,
) -> List[Dict[str, Any]]:
    """Speech audio file (see make_speech_audio) -> list of cues:
    {start, end (seconds), text, confidence, review}."""
    _, endpoint, model = _speech_config()
    audio_path = Path(audio_path)
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    host = endpoint.split("//")[-1].split(".")[0]
    progress(f"Sending {size_mb:.1f} MB to Azure Speech ({model or 'standard Telugu model'}, {host})...")
    definition = build_definition(model, context_hint)
    data = request_transcription(audio_path, definition)

    levels = audio_levels(audio_path)
    duration = (data.get("durationMilliseconds") or 0) / 1000 or len(levels) * FRAME_SEC
    phrases, missing = recover_missing(audio_path, phrases_from_response(data), levels, duration,
                                       definition, progress)
    if not phrases:
        raise RuntimeError("Azure Speech returned no Telugu speech for this audio.")
    progress("Timing the words to the speech in the audio...")
    groups = join_touching(group_words(align_words(phrases, levels)))  # clips are then cut in pauses
    groups = join_touching(assign_by_clips(audio_path, groups, missing, levels, duration,
                                           speaking_rate(phrases, levels), definition, progress))
    cues = build_cues(groups, phrases, missing, levels, duration)
    counts = {flag: sum(1 for c in cues if c["review"] == flag) for flag in ("missing", "recovered", "low")}
    progress(f"Transcription done: {duration / 60:.1f} min of audio -> {len(cues)} cues; flagged for review: "
             f"{counts['low']} low-confidence, {counts['recovered']} re-transcribed, "
             f"{counts['missing']} with no recognised text.")
    return cues
