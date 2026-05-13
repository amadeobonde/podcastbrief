"""RssFeedSource — subscribe to podcast RSS feeds and surface new episodes.

Accepts a list of RSS/Atom feed URLs (any podcast platform publishes one).
Filters by pub_date so only episodes published in the last N hours are returned.
Audio enclosure URL is stored on Episode.audio_url; use DirectAudioFeedResolver
+ the standard HTTP download_audio to fetch the file.

This single adapter covers every major platform:
  Spotify       → open.spotify.com/show/…  (copy RSS from Spotify)
  Apple Podcasts → feeds.feedburner.com/…  (each show has a public RSS)
  Pocket Casts  → pca.st/…
  Overcast      → overcast.fm/…
  Any indie feed → paste the direct RSS URL
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from podcastbrief.core.models import Episode

log = logging.getLogger(__name__)


def _parse_rfc2822(date_str: str | None) -> datetime | None:
    """Try RFC 2822 (RSS standard), then ISO 8601 fallback."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).astimezone(timezone.utc)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _itunes_duration_to_seconds(dur: str) -> int:
    """Parse iTunes duration: HH:MM:SS, MM:SS, or plain seconds."""
    if not dur:
        return 0
    try:
        parts = str(dur).strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


class RssFeedSource:
    """List podcast episodes from RSS feeds published in the last N hours.

    feed_urls: list of RSS/Atom feed URLs — one per podcast show you want
               briefed. Paste the feed URL (not the website URL).
    """

    def __init__(self, feed_urls: list[str]) -> None:
        self.feed_urls = [u.strip() for u in feed_urls if u.strip()]

    def list_recent_episodes(self, *, hours: int = 24) -> list[Episode]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out: list[Episode] = []
        for feed_url in self.feed_urls:
            try:
                episodes = self._fetch_feed(feed_url, cutoff=cutoff)
                out.extend(episodes)
                log.debug("RSS %s: %d episode(s)", feed_url, len(episodes))
            except Exception as exc:
                log.warning("RSS feed error (%s): %s", feed_url, exc)
        log.info("RSS feeds total: %d episode(s) in last %dh", len(out), hours)
        return out

    def _fetch_feed(self, feed_url: str, *, cutoff: datetime) -> list[Episode]:
        resp = httpx.get(feed_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        show_name = (
            (feed.feed.get("title") or "").strip()
            or feed_url.split("/")[-1]
            or feed_url
        )

        out: list[Episode] = []
        for entry in feed.entries:
            pub_date = _parse_rfc2822(
                entry.get("published") or entry.get("updated")
            )
            # If we can't parse a date, include the episode (conservative).
            if pub_date and pub_date < cutoff:
                continue

            # Find audio enclosure.
            audio_url = ""
            for link in entry.get("links", []):
                link_type = link.get("type") or ""
                if link.get("rel") == "enclosure" and (
                    "audio" in link_type or link_type == ""
                ):
                    audio_url = link.get("href", "")
                    break
            if not audio_url and entry.get("enclosures"):
                audio_url = entry.enclosures[0].get("href", "")
            if not audio_url:
                continue  # No audio — skip (news articles, not podcasts)

            guid = (entry.get("id") or entry.get("link") or audio_url).strip()
            episode_id = "rss-" + hashlib.md5(
                (feed_url + guid).encode("utf-8")
            ).hexdigest()[:16]

            duration_s = _itunes_duration_to_seconds(
                entry.get("itunes_duration") or ""
            )
            title = (entry.get("title") or "").strip() or guid
            listen_url = (entry.get("link") or "").strip()

            out.append(
                Episode(
                    episode_id=episode_id,
                    name=title,
                    show_name=show_name,
                    added_at=pub_date or datetime.now(timezone.utc),
                    duration_ms=duration_s * 1000,
                    spotify_url=listen_url,
                    audio_url=audio_url,
                )
            )

        return out
