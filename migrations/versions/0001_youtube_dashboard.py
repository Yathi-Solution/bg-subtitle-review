"""YouTube playlist dashboard: videos, job text, subtitle versions, events, syncs.

Revision ID: 0001
Revises:
Create Date: 2026-11-21

After adding a migration, bump db.EXPECTED_REVISION to its revision ID.
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE youtube_videos (
            video_id            text PRIMARY KEY,
            playlist_id         text NOT NULL,
            position            integer,
            title               text NOT NULL,
            published_at        timestamptz,
            duration_sec        integer,
            thumbnail_url       text,
            privacy_status      text,
            in_playlist         boolean NOT NULL DEFAULT true,
            -- videos.list contentDetails.caption: "captions are available" (any kind,
            -- any language). A hint only; a human sets original_status.
            yt_caption_flag     boolean NOT NULL DEFAULT false,
            original_status     text NOT NULL DEFAULT 'unknown'
                                CHECK (original_status IN ('unknown', 'confirmed', 'not_original')),
            original_by         text,
            original_at         timestamptz,
            workflow_status     text NOT NULL DEFAULT 'not_started'
                                CHECK (workflow_status IN ('not_started', 'queued', 'downloading', 'transcribing',
                                       'telugu_review', 'telugu_approved', 'translating', 'english_review',
                                       'completed', 'failed')),
            telugu_approved_by  text,
            telugu_approved_at  timestamptz,
            english_approved_by text,
            english_approved_at timestamptz,
            current_job_id      text,
            last_error          text,
            audio_track         text,
            first_seen_at       timestamptz NOT NULL DEFAULT now(),
            last_synced_at      timestamptz,
            updated_at          timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX youtube_videos_playlist_idx ON youtube_videos (playlist_id, position)")

    # The permanent copy of a job's text; jobs/<id>/ on disk is a 7-day cache that
    # can be rebuilt from this row (audio is downloaded again).
    op.execute("""
        CREATE TABLE job_artifacts (
            job_id          text PRIMARY KEY,
            video_id        text NOT NULL REFERENCES youtube_videos (video_id) ON DELETE CASCADE,
            meta            jsonb NOT NULL,
            source          jsonb,
            review          jsonb,
            brief           text,
            translation_raw jsonb,
            updated_at      timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX job_artifacts_video_idx ON job_artifacts (video_id)")

    # Original (from the channel) and generated subtitles side by side. Drafts are
    # updated in place; approved and original files get a new version each time.
    op.execute("""
        CREATE TABLE subtitle_files (
            id          bigserial PRIMARY KEY,
            video_id    text NOT NULL REFERENCES youtube_videos (video_id) ON DELETE CASCADE,
            source_type text NOT NULL CHECK (source_type IN ('generated', 'youtube_original')),
            language    text NOT NULL CHECK (language IN ('te', 'en')),
            format      text NOT NULL CHECK (format IN ('srt', 'sbv')),
            stage       text NOT NULL CHECK (stage IN ('draft', 'approved', 'original')),
            version     integer NOT NULL DEFAULT 1,
            content     text NOT NULL,
            sha256      text NOT NULL,
            job_id      text,
            created_by  text,
            created_at  timestamptz NOT NULL DEFAULT now(),
            is_current  boolean NOT NULL DEFAULT true,
            CHECK ((source_type = 'youtube_original') = (stage = 'original'))
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX subtitle_files_current_uq
            ON subtitle_files (video_id, source_type, language, format, stage) WHERE is_current
    """)
    op.execute("""
        CREATE UNIQUE INDEX subtitle_files_version_uq
            ON subtitle_files (video_id, source_type, language, format, stage, version)
    """)

    # Audit log: the app has one shared password and no user accounts.
    op.execute("""
        CREATE TABLE video_events (
            id         bigserial PRIMARY KEY,
            video_id   text REFERENCES youtube_videos (video_id) ON DELETE CASCADE,
            event      text NOT NULL,
            actor      text,
            detail     jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX video_events_video_idx ON video_events (video_id, created_at)")

    op.execute("""
        CREATE TABLE sync_runs (
            id             bigserial PRIMARY KEY,
            playlist_id    text NOT NULL,
            playlist_title text,
            actor          text,
            started_at     timestamptz NOT NULL DEFAULT now(),
            finished_at    timestamptz,
            total          integer,
            new_count      integer,
            removed_count  integer,
            error          text
        )
    """)
    op.execute("CREATE INDEX sync_runs_playlist_idx ON sync_runs (playlist_id, started_at)")


def downgrade() -> None:
    for table in ("sync_runs", "video_events", "subtitle_files", "job_artifacts", "youtube_videos"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
