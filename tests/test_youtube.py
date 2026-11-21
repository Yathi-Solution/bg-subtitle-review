"""Unit tests for youtube.py: original-track selection and small parsers (no network)."""

import pytest

import youtube
from youtube import YouTubeError, pick_original_audio


def fmt(fid, lang=None, note="medium", ext="m4a", abr=128.0, pref=-1, **extra):
    return {"format_id": fid, "language": lang, "format_note": note, "ext": ext, "abr": abr,
            "vcodec": "none", "acodec": "mp4a.40.2" if ext == "m4a" else "opus",
            "language_preference": pref, **extra}


VIDEO = {"format_id": "137", "vcodec": "avc1", "acodec": "none", "ext": "mp4", "format_note": "1080p"}


def test_single_track_prefers_m4a_and_skips_drc():
    formats = [VIDEO, fmt("140-drc", note="medium, DRC", abr=130), fmt("251", ext="webm", abr=140),
               fmt("140", abr=129), fmt("139", note="low", abr=48)]
    assert pick_original_audio(formats)["format_id"] == "140"


def test_original_beats_auto_dubbed_even_when_lang_matches_a_dub():
    # The shape yt-dlp gives a video with YouTube auto-dubbing (seen on a real video).
    formats = [
        fmt("140-0", "te", "Telugu, medium"),                       # a dub in the preferred language
        fmt("140-1", "hi", "Hindi dubbed-auto, medium"),
        fmt("140-2", "en-US", "English (US) original (default), medium", pref=10),
        fmt("251-2", "en-US", "English (US) original (default), medium", ext="webm", abr=150, pref=10),
    ]
    assert pick_original_audio(formats, "te")["format_id"] == "140-2"


def test_telugu_original_is_chosen():
    formats = [fmt("140-0", "te", "Telugu original (default), medium", pref=10),
               fmt("140-1", "en", "English dubbed-auto, medium")]
    assert pick_original_audio(formats, "te")["format_id"] == "140-0"


def test_original_note_without_preference_still_counts():
    formats = [fmt("140-0", "en", "English dubbed-auto"), fmt("140-1", "te", "Telugu (original)")]
    assert pick_original_audio(formats, "")["format_id"] == "140-1"


def test_only_dubbed_tracks_is_refused():
    with pytest.raises(YouTubeError, match="dubbed"):
        pick_original_audio([fmt("140-0", "en", "English dubbed-auto"), fmt("140-1", "hi", "Hindi dubbed")])


def test_several_unmarked_languages_use_lang_or_refuse():
    formats = [fmt("140-0", "en", "English"), fmt("140-1", "te", "Telugu")]
    assert pick_original_audio(formats, "te")["format_id"] == "140-1"
    with pytest.raises(YouTubeError, match="several audio languages"):
        pick_original_audio(formats, "ta")


def test_descriptive_and_drm_tracks_are_ignored():
    formats = [fmt("140-0", "te-desc", "Telugu descriptive", pref=-10, abr=300),
               fmt("140-1", "te", "Telugu", has_drm=True, abr=300), fmt("140-2", "te", "Telugu")]
    assert pick_original_audio(formats, "te")["format_id"] == "140-2"


def test_damaged_copy_is_last_resort():
    formats = [fmt("140", note="medium, DAMAGED", abr=200), fmt("251", note="medium", ext="webm", abr=100)]
    assert pick_original_audio(formats)["format_id"] == "251"


def test_no_audio_only_stream():
    with pytest.raises(YouTubeError, match="no downloadable audio"):
        pick_original_audio([VIDEO])


def test_track_label():
    assert youtube.track_label(fmt("140-2", "en-US", "English (US) original")) == "140-2 · en-US · English (US) original"


@pytest.mark.parametrize("iso, seconds", [("PT1H2M3S", 3723), ("PT45M", 2700), ("PT59S", 59),
                                          ("P1DT1S", 86401), ("PT0S", 0), ("", None), ("junk", None)])
def test_parse_duration(iso, seconds):
    assert youtube.parse_duration(iso) == seconds


@pytest.mark.parametrize("value, expected", [
    ("PLabc123", "PLabc123"),
    ("https://www.youtube.com/playlist?list=PLabc123", "PLabc123"),
    ("https://www.youtube.com/watch?v=xyz&list=PLabc123&index=2", "PLabc123"),
    ("www.youtube.com/playlist?list=PLabc123", "PLabc123"),
    ("  PLabc123  ", "PLabc123"),
])
def test_playlist_id_from(value, expected):
    assert youtube.playlist_id_from(value) == expected


@pytest.mark.parametrize("message, expected", [
    ("ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies", "refused the download"),
    ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "private"),
    ("ERROR: [youtube] abc: Join this channel to get access to members-only content", "members-only"),
    ("ERROR: [youtube] abc: Video unavailable", "unavailable"),
    ("ERROR: HTTP Error 429: Too Many Requests", "rate-limiting"),
    ("ERROR: something else", "could not download the audio: something else"),
])
def test_explain_download_error(message, expected):
    assert expected in youtube.explain_download_error(message)


def test_api_key_is_sent_as_header_not_in_url(monkeypatch):
    seen = {}

    class Res:
        status_code = 200

        @staticmethod
        def json():
            return {"items": []}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return Res()

    monkeypatch.setattr(youtube.requests, "get", fake_get)
    with pytest.raises(YouTubeError, match="not found"):
        youtube.fetch_playlist("PLx", "SECRET")
    assert seen["headers"] == {"X-Goog-Api-Key": "SECRET"}
    assert "SECRET" not in seen["url"] and "SECRET" not in str(seen["params"])
