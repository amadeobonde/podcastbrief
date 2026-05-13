"""Handle on-demand audio in the bot: file uploads and YouTube URLs.

Branching by duration:
- <= 5 minutes: treat as a conversational message. Transcribe via Whisper,
  run RagBot in voice mode for a tight spoken-style answer, reply with both
  the transcribed question and an OGG/Opus voice note.
- > 5 minutes: treat as a full podcast. Run the entire pipeline end-to-end
  (Pass 1, Pass 2, enrich, Pass 3, render PDF, send PDF). Dedup logic kicks
  in via the standard NoteStore.save (any prior note with the same episode
  id is replaced).

For YouTube, we use yt-dlp via subprocess (no Python binding required, the
binary is shipped via scripts/install.sh or via uv/pip). Audio is downloaded
to a tempfile, parsed for duration, and routed.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from podcastbrief.adapters._yt_dlp import yt_dlp_path as _yt_dlp_path_shared
from podcastbrief.bot.rag import RagBot
from podcastbrief.bot.voice import VoiceProcessor
from podcastbrief.core.models import AudioRef, Episode
from podcastbrief.core.pipeline import Pipeline

log = logging.getLogger(__name__)


SHORT_THRESHOLD_S = 300  # ≤5 min ⇒ conversational reply, otherwise full PDF brief


_YT_RX = re.compile(
    r"(?:https?://)?(?:www\.)?(youtube\.com/watch\?[\w=&-]+|youtu\.be/[\w-]+|youtube\.com/shorts/[\w-]+)",
    re.IGNORECASE,
)


def is_youtube_url(text: str) -> bool:
    return bool(_YT_RX.search(text or ""))


def _yt_dlp_path() -> str | None:
    return _yt_dlp_path_shared()


def _download_youtube_audio(url: str, tmpdir: Path) -> tuple[Path, dict]:
    yt = _yt_dlp_path()
    if not yt:
        raise RuntimeError(
            "yt-dlp not found. Install with: pip install yt-dlp  OR  "
            "curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos -o ~/.local/bin/yt-dlp && chmod +x ~/.local/bin/yt-dlp"
        )
    out_template = str(tmpdir / "%(id)s.%(ext)s")
    info_json = tmpdir / "info.json"
    subprocess.run(
        [
            yt,
            "-x", "--audio-format", "mp3",
            "--no-playlist",
            "--no-warnings",
            "--write-info-json",
            "-o", out_template,
            url,
        ],
        check=True,
        capture_output=True,
    )
    # Locate the produced mp3 and info JSON.
    mp3s = list(tmpdir.glob("*.mp3"))
    infos = list(tmpdir.glob("*.info.json"))
    if not mp3s:
        raise RuntimeError("yt-dlp finished but produced no audio file.")
    info = {}
    if infos:
        try:
            info = json.loads(infos[0].read_text(encoding="utf-8"))
        except Exception:
            info = {}
    return mp3s[0], info


def _ffprobe_duration_s(audio_path: Path) -> int:
    """Robust duration probe — uses the bundled ffmpeg as ffprobe."""
    # ffmpeg can print duration to stderr.
    candidates = [
        os.path.expanduser("~/.local/bin/ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ]
    ffmpeg = next((c for c in candidates if c and Path(c).is_file()), None)
    if not ffmpeg:
        return 0
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", str(audio_path), "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        err = proc.stderr or ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", err)
        if not m:
            return 0
        h, mn, sc = (int(g) for g in m.groups())
        return h * 3600 + mn * 60 + sc
    except Exception:
        return 0


async def _download_telegram_audio(update: Update, tmpdir: Path) -> tuple[Path, int, str, str]:
    """Returns (path, duration_s, title, mime). Works for audio, voice, video,
    and document.audio attachments."""
    msg = update.message
    attach = None
    for getter in (lambda: msg.audio, lambda: msg.video, lambda: msg.voice, lambda: msg.document):
        try:
            a = getter()
            if a:
                attach = a
                break
        except Exception:
            continue
    if attach is None:
        raise RuntimeError("Message has no audio/video/document attachment.")
    file = await attach.get_file()
    suffix = ".m4a"
    file_path = file.file_path or ""
    if file_path.endswith(".mp3"):
        suffix = ".mp3"
    elif file_path.endswith(".ogg") or file_path.endswith(".oga"):
        suffix = ".ogg"
    elif file_path.endswith(".mp4") or file_path.endswith(".m4a"):
        suffix = ".m4a"
    elif file_path.endswith(".wav"):
        suffix = ".wav"
    out = tmpdir / f"upload{suffix}"
    await file.download_to_drive(str(out))
    duration_s = getattr(attach, "duration", None) or 0
    if not duration_s:
        duration_s = _ffprobe_duration_s(out)
    title = getattr(attach, "title", None) or getattr(attach, "file_name", None) or "upload"
    return out, int(duration_s), str(title), suffix


async def handle_youtube_or_upload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    url_or_file_id: str | None,
    voice: VoiceProcessor,
    rag: RagBot,
    pipe: Pipeline,
    source: str,
) -> None:
    msg = update.message
    if not msg:
        return
    user_id = str(msg.from_user.id) if msg.from_user else "anon"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        try:
            if source == "youtube_url" and url_or_file_id:
                await msg.reply_text("Pulling audio from YouTube…")
                audio_path, yt_info = await asyncio.to_thread(
                    _download_youtube_audio, url_or_file_id, td_path
                )
                title = str(yt_info.get("title") or "YouTube clip")
                pub_date = str(yt_info.get("upload_date") or "")  # YYYYMMDD
                show_name = str(yt_info.get("uploader") or "YouTube")
                duration_s = int(yt_info.get("duration") or _ffprobe_duration_s(audio_path) or 0)
                episode_id = f"yt-{yt_info.get('id') or audio_path.stem}"
            else:
                await msg.reply_text("Processing uploaded audio…")
                audio_path, duration_s, title, _suffix = await _download_telegram_audio(update, td_path)
                show_name = "Telegram Upload"
                pub_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                episode_id = f"tg-{msg.message_id}-{user_id}"
        except Exception as e:
            log.exception("Upload prep failed: %s", e)
            await msg.reply_text(f"Couldn't fetch the audio: {e}")
            return

        log.info("Upload routed: source=%s duration=%ss title=%r", source, duration_s, title)
        audio_bytes = audio_path.read_bytes()

        # SHORT path: treat as a conversational query, voice reply.
        if duration_s and duration_s <= SHORT_THRESHOLD_S:
            try:
                await msg.reply_chat_action("typing")
                transcript = await asyncio.to_thread(
                    pipe.transcriber.transcribe, audio_bytes, filename=audio_path.name
                )
                question = (transcript.text or "").strip()
                if not question:
                    await msg.reply_text("Couldn't make out the audio.")
                    return
                answer = await asyncio.to_thread(
                    rag.answer, user_id=user_id, question=question, mode="voice"
                )
                await msg.reply_chat_action("record_voice")
                audio_out, mime, fname = await asyncio.to_thread(voice.synthesize, answer)
                import io as _io
                buf = _io.BytesIO(audio_out)
                buf.name = fname
                if mime == "audio/ogg":
                    await msg.reply_voice(voice=buf, caption=question[:1024])
                else:
                    await msg.reply_audio(audio=buf, title="Reply", caption=question[:1024])
            except Exception as e:
                log.exception("Short-form upload reply failed: %s", e)
                await msg.reply_text(f"Couldn't answer: {e}")
            return

        # LONG path: full pipeline → PDF
        await msg.reply_text(
            f"That's a {duration_s // 60}-minute piece — processing as a full brief. "
            f"This takes a few minutes; the PDF will land here when it's done."
        )

        ep = Episode(
            episode_id=episode_id,
            name=title,
            show_name=show_name,
            added_at=datetime.now(timezone.utc),
            duration_ms=duration_s * 1000,
            spotify_url="",
        )
        audio_ref = AudioRef(url="", title=title, pub_date=pub_date, show_name=show_name)
        try:
            # Run the blocking pipeline off the event loop in a thread.
            await asyncio.to_thread(
                _drive_pipeline_from_audio,
                pipe, ep=ep, audio_ref=audio_ref, audio_bytes=audio_bytes,
            )
            await msg.reply_text("Done — PDF sent above.")
        except Exception as e:
            log.exception("Full-pipeline upload failed: %s", e)
            await msg.reply_text(f"Pipeline failed: {e}")


def _drive_pipeline_from_audio(
    pipe: Pipeline, *, ep: Episode, audio_ref: AudioRef, audio_bytes: bytes
) -> None:
    """Replay the Pipeline._process_episode flow against pre-downloaded audio.

    Uses the feed/downloader override parameters so we don't need to monkey-patch
    the pipeline object. The rest of the pipeline runs unchanged — Whisper,
    Pass 1, Pass 2, enrichment, Pass 3, render, notify, save.
    """
    from podcastbrief.adapters.youtube_feed import DirectAudioFeedResolver

    shim_feed = DirectAudioFeedResolver()
    # Inject the already-resolved AudioRef by storing its URL on the episode.
    ep.audio_url = audio_ref.url

    pipe._process_episode(
        ep,
        dry_run=False,
        force=True,
        feed=shim_feed,
        audio_downloader=lambda _ref: audio_bytes,
    )
