"""YouTubePlaylistSource — list recent videos from a YouTube playlist via yt-dlp.

Uses --flat-playlist (no audio download) to enumerate videos, then filters by
upload_date as a proxy for "added in the last N hours". YouTube playlists do not
expose per-item addition timestamps, so upload_date is the best available signal.
The pipeline's episode-ID dedup (episode_id = "yt-{video_id}") prevents any video
from being processed more than once regardless of how often it appears.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone

from podcastbrief.adapters._yt_dlp import yt_dlp_path
from podcastbrief.core.models import Episode

log = logging.getLogger(__name__)


class YouTubePlaylistSource:
    """List YouTube playlist videos uploaded in the last N hours.

    Requires yt-dlp on $PATH (or in ~/.local/bin / /opt/homebrew/bin).
    Install: pip install yt-dlp  OR  scripts/install.sh (already handles it).
    """

    def __init__(self, playlist_url: str) -> None:
        self.playlist_url = playlist_url

    def list_recent_episodes(self, *, hours: int = 24) -> list[Episode]:
        yt = yt_dlp_path()
        if not yt:
            log.error(
                "yt-dlp not found — YouTube playlist skipped. "
                "Install with: pip install yt-dlp  OR run scripts/install.sh"
            )
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        # yt-dlp accepts YYYYMMDD for --dateafter
        dateafter = cutoff.strftime("%Y%m%d")

        cmd = [
            yt,
            "--flat-playlist",
            "--dump-single-json",
            "--no-warnings",
            "--dateafter", dateafter,
            "--extractor-args", "youtube:skip=dash,hls",
            self.playlist_url,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            log.error("yt-dlp timed out fetching YouTube playlist")
            return []

        if proc.returncode != 0:
            log.warning("yt-dlp playlist error: %s", (proc.stderr or "")[:400])
            return []

        try:
            data = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("yt-dlp JSON parse error: %s", exc)
            return []

        entries = data.get("entries") or []
        channel_name = data.get("uploader") or data.get("channel") or "YouTube"

        out: list[Episode] = []
        for entry in entries:
            if not entry:
                continue
            video_id = entry.get("id") or ""
            if not video_id:
                continue

            title = (entry.get("title") or video_id).strip()
            uploader = (entry.get("uploader") or entry.get("channel") or channel_name).strip()
            duration_s = int(entry.get("duration") or 0)

            upload_date_str = entry.get("upload_date") or ""
            added_at = datetime.now(timezone.utc)
            if upload_date_str and len(upload_date_str) == 8:
                try:
                    added_at = datetime(
                        int(upload_date_str[:4]),
                        int(upload_date_str[4:6]),
                        int(upload_date_str[6:8]),
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    pass

            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            out.append(
                Episode(
                    episode_id=f"yt-{video_id}",
                    name=title,
                    show_name=uploader,
                    added_at=added_at,
                    duration_ms=duration_s * 1000,
                    spotify_url=yt_url,
                    audio_url=yt_url,
                )
            )

        log.info("YouTube playlist: %d video(s) in last %dh", len(out), hours)
        return out
