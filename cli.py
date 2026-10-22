#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py — command-line entry point for the lean subtitle review pipeline.

Examples:
    python cli.py transcribe episode.mp3 --formats sbv,srt --out-dir out_v4_1_lean
    python cli.py translate "CH 05 _EP 11.sbv" --out-dir out_v4_1_lean
    python cli.py validate-workbook "CH 05 _EP 11.sbv" out_v4_1_lean/CH_05__EP_11_master_review.xlsx
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

import pipeline


def cmd_transcribe(args: argparse.Namespace) -> None:
    import transcribe

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in ("sbv", "srt")]
    if bad or not formats:
        sys.exit(f"--formats must be sbv, srt or both (got {args.formats!r})")

    source = Path(args.source)
    with tempfile.TemporaryDirectory() as tmp:
        speech = transcribe.make_speech_audio(source, Path(tmp))
        cues = transcribe.transcribe_audio(speech, context_hint=args.hint, progress=lambda msg: print(msg))
    entries = [
        (pipeline.seconds_to_srt_ts(c["start"]), pipeline.seconds_to_srt_ts(c["end"]), c["text"])
        for c in cues if c["text"]
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nDONE — review the Telugu before translating:")
    for fmt in formats:
        path = out_dir / f"{pipeline.safe_stem(source.name)}_telugu.{fmt}"
        path.write_text(pipeline.format_subtitles(entries, fmt), encoding="utf-8")
        print(f"Telugu {fmt.upper()}: {path}")
    missing = [c for c in cues if not c["text"]]
    if missing:
        spans = ", ".join(f"{pipeline.seconds_to_srt_ts(c['start'])}-{pipeline.seconds_to_srt_ts(c['end'])}" for c in missing)
        print(f"No text recognised (left out of the files; listen and add them): {spans}")
    numbered = [c for c in cues if c["text"]]  # cue numbers as in the written files
    for flag, label in (("recovered", "Re-transcribed cues to check"), ("low", "Low-confidence cues to check")):
        nums = [str(i) for i, c in enumerate(numbered, 1) if c["review"] == flag]
        if nums:
            print(f"{label}: {', '.join(nums)}")


def cmd_translate(args: argparse.Namespace) -> None:
    source = Path(args.source)
    skill_text = pipeline.load_skill_text(Path(args.skill) if args.skill else None)
    artifacts = pipeline.translate_to_files(
        source_text=source.read_text(encoding="utf-8-sig"),
        source_name=source.name,
        out_dir=Path(args.out_dir),
        skill_text=skill_text,
        model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        progress=lambda msg: print(msg),
    )
    print("\nDONE")
    print(f"Discourse brief: {artifacts['brief']}")
    print(f"Master review workbook: {artifacts['workbook']}")
    print(f"Raw JSON: {artifacts['raw']}")


def cmd_validate_workbook(args: argparse.Namespace) -> None:
    cues = pipeline.parse_source(Path(args.source))
    df = pd.read_excel(args.workbook)

    ok = True
    if len(df) != len(cues):
        print(f"FAIL cue count: workbook {len(df)} vs source {len(cues)}")
        ok = False

    for cue in cues:
        row = df[df["Cue Number"] == cue.number]
        if row.empty:
            print(f"FAIL missing cue {cue.number}")
            ok = False
            continue
        row = row.iloc[0]
        if str(row["Timecode"]).strip() != cue.timecode:
            print(f"FAIL timecode mismatch cue {cue.number}")
            ok = False
        if str(row["Telugu Cue"]).strip() != cue.telugu.strip():
            print(f"WARN Telugu text differs cue {cue.number}")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telugu audio/subtitle -> English review workbook automation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transcribe", help="Create Telugu SBV/SRT from an audio file")
    p.add_argument("source", help="Audio file (.mp3, .m4a, .wav, ...)")
    p.add_argument("--formats", default="sbv,srt", help="sbv, srt or sbv,srt")
    p.add_argument("--out-dir", default="out_v4_1_lean")
    p.add_argument("--hint", default="", help="Names/terms that appear in the audio")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("translate", help="Create review workbook from Telugu SBV/SRT")
    p.add_argument("source")
    p.add_argument("--out-dir", default="out_v4_1_lean")
    p.add_argument("--skill", default=pipeline.DEFAULT_SKILL_FILE)
    p.add_argument("--model", default=None, help="Azure deployment name (overrides env)")
    p.add_argument("--chunk-size", type=int, default=pipeline.DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=pipeline.DEFAULT_OVERLAP)
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("validate-workbook", help="Validate workbook against source")
    p.add_argument("source")
    p.add_argument("workbook")
    p.set_defaults(func=cmd_validate_workbook)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
