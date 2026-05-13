"""AppleMusicSource — list tracks from a public Apple Music playlist.

Requires an Apple Music developer token (JWT signed with a MusicKit key).
Public playlists are accessible via the Apple Music catalog API without a
user music token — only the developer token is needed.

Setup (one-time):
  1. Apple Developer account → Certificates, IDs & Profiles → Keys → + Key
     Check "MusicKit" → download the .p8 private key
  2. pip install PyJWT cryptography
  3. Generate the JWT:

       python - <<'EOF'
       import jwt, time, pathlib
       private_key = pathlib.Path("AuthKey_XXXXXXXXXX.p8").read_text()
       token = jwt.encode(
           {"iss": "YOUR_TEAM_ID", "iat": int(time.time()), "exp": int(time.time()) + 15_777_000},
           private_key,
           algorithm="ES256",
           headers={"kid": "YOUR_KEY_ID"},
       )
       print(token)
       EOF

  4. Paste the token into APPLE_MUSIC_DEV_TOKEN in .env.

Audio: Apple Music tracks are DRM-protected. For podcast episodes found in
Apple Music, ItunesRssFeed (the default FeedResolver) is used to locate the
open RSS enclosure URL. For non-podcast music tracks this source is not useful.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import httpx

from podcastbrief.core.models import Episode

log = logging.getLogger(__name__)

_API_BASE = "https://api.music.apple.com/v1"


def _parse_playlist_id(url: str) -> tuple[str, str]:
    """Extract (storefront, playlist_id) from an Apple Music playlist URL."""
    pattern = r"music\.apple\.com/([a-z]{2})/playlist/[^/]*/?(pl\.[^?#/\s]+)"
    m = re.search(pattern, url, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    # Shorter form without slug: /us/playlist/pl.xxx
    pattern2 = r"music\.apple\.com/([a-z]{2})/playlist/(pl\.[^?#/\s]+)"
    m2 = re.search(pattern2, url, re.IGNORECASE)
    if m2:
        return m2.group(1), m2.group(2)
    raise ValueError(f"Cannot parse Apple Music playlist URL: {url!r}")


class AppleMusicSource:
    """List tracks from a public Apple Music playlist via the MusicKit catalog API.

    For podcast content, pair this with the default ItunesRssFeed resolver so the
    pipeline can locate the open RSS audio enclosure. For music tracks the pipeline
    will attempt the same iTunes lookup, which may fail if no podcast RSS exists —
    those episodes will be skipped with a warning.
    """

    def __init__(self, playlist_url: str, developer_token: str) -> None:
        self.playlist_url = playlist_url
        self.developer_token = developer_token.strip()
        self.storefront = "us"
        self.playlist_id = ""
        if playlist_url:
            try:
                self.storefront, self.playlist_id = _parse_playlist_id(playlist_url)
            except ValueError as exc:
                log.error("Apple Music URL parse failed: %s", exc)

    def list_recent_episodes(self, *, hours: int = 24) -> list[Episode]:
        if not self.developer_token:
            log.warning("APPLE_MUSIC_DEV_TOKEN not set — Apple Music source skipped")
            return []
        if not self.playlist_id:
            log.warning("Apple Music playlist ID could not be parsed — skipping")
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        headers = {"Authorization": f"Bearer {self.developer_token}"}
        url = f"{_API_BASE}/catalog/{self.storefront}/playlists/{self.playlist_id}"

        try:
            resp = httpx.get(
                url,
                headers=headers,
                params={"include": "tracks"},
                timeout=30,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Apple Music API %s: %s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            return []
        except Exception as exc:
            log.error("Apple Music request failed: %s", exc)
            return []

        playlist_data = (resp.json().get("data") or [{}])[0]
        tracks = (
            playlist_data
            .get("relationships", {})
            .get("tracks", {})
            .get("data") or []
        )

        out: list[Episode] = []
        for track in tracks:
            attrs = track.get("attributes") or {}
            track_id = track.get("id") or ""
            name = (attrs.get("name") or "").strip()
            if not name or not track_id:
                continue

            artist = (attrs.get("artistName") or attrs.get("albumName") or "Apple Music").strip()
            duration_ms = int(attrs.get("durationInMillis") or 0)

            release_str = attrs.get("releaseDate") or ""
            added_at = datetime.now(timezone.utc)
            if release_str:
                try:
                    added_at = datetime.fromisoformat(
                        release_str.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except ValueError:
                    pass

            if added_at < cutoff:
                continue

            apple_url = (
                attrs.get("url")
                or f"https://music.apple.com/{self.storefront}/album/{track_id}"
            )

            out.append(
                Episode(
                    episode_id=f"am-{track_id}",
                    name=name,
                    show_name=artist,
                    added_at=added_at,
                    duration_ms=duration_ms,
                    spotify_url=apple_url,
                    audio_url="",  # resolved via ItunesRssFeed
                )
            )

        log.info("Apple Music playlist: %d track(s) in last %dh", len(out), hours)
        return out
