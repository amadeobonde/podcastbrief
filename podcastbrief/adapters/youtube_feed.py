"""YouTube feed resolver and audio downloader.

YouTubeFeedResolver  — implements FeedResolver for episodes whose audio_url is a
                       YouTube watch URL (set by YouTubePlaylistSource).
DirectAudioFeedResolver — generic resolver for any source that already embedded
                          the audio URL directly on Episode.audio_url (e.g. RSS).
youtube_download_audio  — downloads YouTube audio to bytes via yt-dlp.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from podcastbrief.adapters._yt_dlp import yt_dlp_path
from podcastbrief.core.models import AudioRef, Episode

log = logging.getLogger(__name__)


class YouTubeFeedResolver:
    """FeedResolver that treats Episode.audio_url as a YouTube watch URL."""

    def find_audio(self, episode: Episode) -> AudioRef:
        url = episode.audio_url or episode.spotify_url
        if not url or ("youtube.com" not in url and "youtu.be" not in url):
            raise ValueError(f"No YouTube URL on episode: {episode.name!r}")
        return AudioRef(
            url=url,
            title=episode.name,
            pub_date=episode.added_at.strftime("%Y-%m-%d") if episode.added_at else None,
            show_name=episode.show_name,
        )


class DirectAudioFeedResolver:
    """FeedResolver for sources that already carry the audio URL on Episode.audio_url.

    Used by RssFeedSource and AppleMusicSource (when a direct enclosure URL is
    available). Falls back to episode.spotify_url if audio_url is empty.
    """

    def find_audio(self, episode: Episode) -> AudioRef:
        url = episode.audio_url or episode.spotify_url
        if not url:
            raise ValueError(f"No audio URL on episode: {episode.name!r}")
        return AudioRef(
            url=url,
            title=episode.name,
            pub_date=episode.added_at.strftime("%Y-%m-%d") if episode.added_at else None,
            show_name=episode.show_name,
        )


def youtube_download_audio(audio_ref: AudioRef, *, timeout: int = 600) -> bytes:
    """Download audio from a YouTube URL via yt-dlp and return MP3 bytes."""
    yt = yt_dlp_path()
    if not yt:
        raise RuntimeError(
            "yt-dlp not found. Install with: pip install yt-dlp  "
            "OR run scripts/install.sh"
        )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cmd = [
            yt,
            "-x", "--audio-format", "mp3",
            "--no-playlist",
            "--no-warnings",
            "-o", str(td_path / "%(id)s.%(ext)s"),
            audio_ref.url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"yt-dlp failed for {audio_ref.url}: {exc.stderr.decode()[:300]}"
            ) from exc

        mp3s = list(td_path.glob("*.mp3"))
        if not mp3s:
            raise RuntimeError(f"yt-dlp produced no MP3 for: {audio_ref.url}")
        return mp3s[0].read_bytes()
