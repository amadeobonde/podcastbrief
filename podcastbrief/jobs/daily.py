from __future__ import annotations
import logging
from podcastbrief.core.config import Settings
from podcastbrief.core.pipeline import Pipeline
from podcastbrief.adapters.spotify_source import SpotifySource
from podcastbrief.adapters.itunes_rss_feed import ItunesRssFeed, download_audio
from podcastbrief.adapters.whisper_http import WhisperHttpTranscriber
from podcastbrief.adapters.ollama_gemma import OllamaGemma
from podcastbrief.adapters.artwork_rss import ItunesArtworkProvider
from podcastbrief.adapters.gotenberg_renderer import GotenbergRenderer
from podcastbrief.adapters.telegram_notifier import TelegramNotifier
from podcastbrief.adapters.fs_notes import FilesystemNoteStore
from podcastbrief.adapters.spotify_recommender import SpotifyEpisodeRecommender
from podcastbrief.adapters.yahoo_enricher import YahooFinanceEnricher
from podcastbrief.adapters.fred_enricher import FREDEnricher
from podcastbrief.adapters.wikipedia_enricher import WikipediaEnricher
from podcastbrief.adapters.rss_news_enricher import RSSNewsEnricher

log = logging.getLogger(__name__)


def _build_extra_bundles(s: Settings) -> list[tuple]:
    """Build (source, feed_resolver, audio_downloader) tuples for non-Spotify sources."""
    bundles = []

    # YouTube playlist via yt-dlp — legally clean, huge educational library.
    if s.youtube_playlist_url:
        from podcastbrief.adapters.youtube_source import YouTubePlaylistSource
        from podcastbrief.adapters.youtube_feed import YouTubeFeedResolver, youtube_download_audio
        log.info("YouTube playlist source enabled: %s", s.youtube_playlist_url)
        bundles.append((
            YouTubePlaylistSource(s.youtube_playlist_url),
            YouTubeFeedResolver(),
            youtube_download_audio,
        ))

    # RSS podcast feed subscriptions — covers every platform simultaneously.
    if s.rss_podcast_feed_list:
        from podcastbrief.adapters.rss_source import RssFeedSource
        from podcastbrief.adapters.youtube_feed import DirectAudioFeedResolver
        log.info("RSS podcast source enabled: %d feed(s)", len(s.rss_podcast_feed_list))
        bundles.append((
            RssFeedSource(s.rss_podcast_feed_list),
            DirectAudioFeedResolver(),
            download_audio,  # standard HTTP download
        ))

    # Apple Music playlist (requires developer token).
    if s.apple_music_playlist_url and s.apple_music_dev_token:
        from podcastbrief.adapters.apple_music_source import AppleMusicSource
        log.info("Apple Music source enabled: %s", s.apple_music_playlist_url)
        # Apple Music tracks → ItunesRssFeed resolves to open RSS audio URL.
        bundles.append((
            AppleMusicSource(s.apple_music_playlist_url, s.apple_music_dev_token),
            ItunesRssFeed(),
            download_audio,
        ))

    return bundles


def build_pipeline(s: Settings) -> Pipeline:
    spotify = SpotifySource(
        client_id=s.spotify_client_id,
        client_secret=s.spotify_client_secret,
        refresh_token=s.spotify_refresh_token,
        playlist_id=s.spotify_playlist_id,
    )
    enrichers = [
        YahooFinanceEnricher(),
        FREDEnricher(api_key=s.fred_api_key),
        WikipediaEnricher(),
        RSSNewsEnricher(feeds=s.rss_feed_list),
    ]
    return Pipeline(
        source=spotify,
        feed=ItunesRssFeed(),
        transcriber=WhisperHttpTranscriber(
            base_url=s.whisper_url, timeout_seconds=s.whisper_timeout_seconds
        ),
        llm=OllamaGemma(host=s.ollama_host, model=s.llm_model),
        images=ItunesArtworkProvider(),
        renderer=GotenbergRenderer(base_url=s.gotenberg_url),
        notifier=TelegramNotifier(bot_token=s.telegram_bot_token),
        notes=FilesystemNoteStore(base_dir=s.notes_dir),
        recommender=SpotifyEpisodeRecommender(token_getter=spotify._access_token),
        chat_ids=s.chat_id_list,
        audio_downloader=download_audio,
        pdf_out_dir=s.pdf_out_dir,
        enrichers=enrichers,
        notes_dir=s.notes_dir,
        extra_source_bundles=_build_extra_bundles(s),
    )


def run_daily(*, dry_run: bool = False, hours: int = 24) -> int:
    from podcastbrief.core.config import load_settings

    s = load_settings()
    logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))
    pipe = build_pipeline(s)
    return pipe.run_daily(hours=hours, dry_run=dry_run)
