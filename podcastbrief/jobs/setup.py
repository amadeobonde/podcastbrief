"""Interactive first-run setup wizard for Podcast Brief.

Run:  podcastbrief setup
      (or: .venv/bin/podcastbrief setup  from the repo root)

What it does in order:
  1. Checks + installs system dependencies (Ollama, Docker, yt-dlp, ffmpeg)
  2. Pulls Gemma 4 E4B via Ollama if not already present
  3. Starts Whisper + Gotenberg Docker containers if not running
  4. Collects Spotify, Telegram, YouTube, RSS, and Apple Music credentials
  5. Runs Spotify OAuth to obtain a refresh token (if Spotify configured)
  6. Writes (or updates) .env in the current working directory

After running setup, start the service with:
  podcastbrief serve       # scheduler + Telegram bot in one process
  podcastbrief run-daily   # one-off daily brief run
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    click.echo(click.style("  ✓ ", fg="green") + msg)


def _warn(msg: str) -> None:
    click.echo(click.style("  ! ", fg="yellow") + msg)


def _err(msg: str) -> None:
    click.echo(click.style("  ✗ ", fg="red") + msg)


def _section(title: str) -> None:
    width = 60
    click.echo()
    click.echo(click.style("─" * width, fg="cyan"))
    click.echo(click.style(f"  {title}", fg="cyan", bold=True))
    click.echo(click.style("─" * width, fg="cyan"))


def _run(cmd: list[str], *, check: bool = True, capture: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _local_bin() -> Path:
    p = Path.home() / ".local" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Dependency checks + installation
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_ollama() -> bool:
    """Check Ollama is installed and running. Return True if ready."""
    if not _cmd_exists("ollama"):
        if platform.system() == "Darwin":
            _warn(
                "Ollama not found. Install the one-click .pkg from https://ollama.com/download\n"
                "  then re-run  podcastbrief setup"
            )
        else:
            _warn("Installing Ollama (Linux)…")
            try:
                subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True, check=True, timeout=300,
                )
                _ok("Ollama installed")
            except Exception as exc:
                _err(f"Ollama install failed: {exc}")
                return False
    else:
        try:
            v = _run(["ollama", "--version"]).stdout.strip().splitlines()[0]
            _ok(f"Ollama: {v}")
        except Exception:
            _ok("Ollama: found")
    return True


def _ensure_gemma(model: str = "gemma4:e4b") -> None:
    """Pull the Gemma model if it's not already local."""
    try:
        result = _run(["ollama", "list"])
        if model in result.stdout:
            _ok(f"{model}: already pulled")
            return
    except Exception:
        pass
    click.echo(f"  Pulling {model} (~9.6 GB, one-time download)…")
    try:
        subprocess.run(["ollama", "pull", model], check=True, timeout=3600)
        _ok(f"{model} ready")
    except Exception as exc:
        _warn(f"Could not pull {model}: {exc}\n  Run  ollama pull {model}  manually and re-run setup.")


def _ensure_docker() -> bool:
    """Check Docker is available. Return True if ready."""
    if not _cmd_exists("docker"):
        if platform.system() == "Darwin":
            _warn(
                "Docker not found. Install Docker Desktop from\n"
                "  https://www.docker.com/products/docker-desktop\n"
                "  then re-run  podcastbrief setup"
            )
        else:
            _warn(
                "Docker not found. Install Docker Engine for your distro\n"
                "  (https://docs.docker.com/engine/install/) then re-run setup."
            )
        return False
    try:
        v = _run(["docker", "--version"]).stdout.strip()
        _ok(f"Docker: {v}")
    except Exception:
        _ok("Docker: found")
    return True


def _ensure_containers(repo_root: Path) -> None:
    """Start Whisper + Gotenberg via docker compose if not already running."""
    try:
        running = _run(["docker", "ps", "--format", "{{.Names}}"]).stdout
    except Exception:
        _warn("Could not check Docker container status.")
        return

    need_start = (
        "podcastbrief-whisper" not in running
        or "podcastbrief-gotenberg" not in running
    )
    if not need_start:
        _ok("Whisper + Gotenberg containers already running")
        return

    compose_file = repo_root / "docker-compose.yml"
    if not compose_file.exists():
        _warn(f"docker-compose.yml not found at {repo_root} — skipping container start")
        return

    click.echo("  Starting Whisper + Gotenberg via docker compose…")
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=str(repo_root),
            check=True,
            timeout=180,
        )
        _ok("Whisper + Gotenberg started")
    except Exception as exc:
        _warn(f"docker compose up failed: {exc}\n  Run  docker compose up -d  from {repo_root} manually.")


def _ensure_ytdlp() -> None:
    """Install yt-dlp standalone binary if not found."""
    if _cmd_exists("yt-dlp"):
        try:
            v = _run(["yt-dlp", "--version"]).stdout.strip()
            _ok(f"yt-dlp: {v}")
        except Exception:
            _ok("yt-dlp: found")
        return

    local_bin = _local_bin()
    dest = local_bin / "yt-dlp"
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        url = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
    elif system == "Linux":
        url = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"
    else:
        _warn("yt-dlp: unsupported platform — install manually: pip install yt-dlp")
        return

    click.echo(f"  Installing yt-dlp to {dest}…")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        dest.chmod(0o755)
        _ok(f"yt-dlp installed at {dest}")
    except Exception as exc:
        _warn(f"yt-dlp install failed: {exc}\n  Install manually: pip install yt-dlp")


def _ensure_ffmpeg() -> None:
    """Best-effort: report ffmpeg status (voice replies need it)."""
    if _cmd_exists("ffmpeg"):
        try:
            v = _run(["ffmpeg", "-version"]).stdout.splitlines()[0]
            _ok(f"ffmpeg: {v}")
        except Exception:
            _ok("ffmpeg: found")
    else:
        _warn(
            "ffmpeg not found — voice replies will fall back to M4A.\n"
            "  macOS: brew install ffmpeg  OR  see scripts/install.sh for a static binary.\n"
            "  Linux: sudo apt install ffmpeg"
        )


# ──────────────────────────────────────────────────────────────────────────────
# .env helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_env(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a {KEY: value} dict (ignores comments/blanks)."""
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _write_env(env_path: Path, values: dict[str, str]) -> None:
    """Write key=value pairs to .env, preserving existing comments/structure."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                updated.add(key)
                continue
        new_lines.append(line)

    # Append any keys that weren't already in the file.
    remaining = {k: v for k, v in values.items() if k not in updated}
    if remaining:
        new_lines.append("")
        new_lines.append("# Added by podcastbrief setup")
        for k, v in remaining.items():
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _extract_playlist_id(url: str) -> str:
    """Extract Spotify playlist ID from a URL or return the raw string."""
    m = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", url)
    return m.group(1) if m else url.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main wizard
# ──────────────────────────────────────────────────────────────────────────────

def run_setup(*, env_path: Path, repo_root: Path) -> None:
    click.clear()
    click.echo(click.style("🎙  Podcast Brief — Setup Wizard", bold=True, fg="cyan"))
    click.echo(click.style("=" * 44, fg="cyan"))
    click.echo(
        "\nThis wizard will:\n"
        "  • Check + install required tools (Ollama, Docker, yt-dlp, ffmpeg)\n"
        "  • Pull Gemma 4 E4B and start Whisper / Gotenberg containers\n"
        "  • Collect your credentials and playlist / feed URLs\n"
        "  • Run Spotify OAuth to get a refresh token (if Spotify is configured)\n"
        "  • Write your .env\n\n"
        "Press Ctrl+C to quit at any time."
    )

    existing = _read_env(env_path)

    # ── 1. DEPENDENCIES ────────────────────────────────────────────────────────
    _section("1 / 5  Dependencies")

    ollama_ok = _ensure_ollama()
    if ollama_ok:
        llm_model = click.prompt(
            "  Gemma model", default=existing.get("LLM_MODEL", "gemma4:e4b")
        )
        _ensure_gemma(llm_model)
    else:
        llm_model = existing.get("LLM_MODEL", "gemma4:e4b")

    docker_ok = _ensure_docker()
    if docker_ok:
        _ensure_containers(repo_root)

    _ensure_ytdlp()
    _ensure_ffmpeg()

    # ── 2. TELEGRAM ────────────────────────────────────────────────────────────
    _section("2 / 5  Telegram Bot")
    click.echo(
        "  Create a bot via @BotFather → /newbot.\n"
        "  Get your chat ID by messaging @userinfobot.\n"
    )
    telegram_token = click.prompt(
        "  Bot token (TELEGRAM_BOT_TOKEN)",
        default=existing.get("TELEGRAM_BOT_TOKEN", ""),
        show_default=bool(existing.get("TELEGRAM_BOT_TOKEN")),
    )
    telegram_chats = click.prompt(
        "  Chat IDs, comma-separated (TELEGRAM_CHAT_IDS)",
        default=existing.get("TELEGRAM_CHAT_IDS", ""),
        show_default=bool(existing.get("TELEGRAM_CHAT_IDS")),
    )

    # ── 3. SOURCES ─────────────────────────────────────────────────────────────
    _section("3 / 5  Content Sources")

    # Spotify
    click.echo("  ── Spotify (optional) ──")
    spotify_id = click.prompt(
        "  Client ID (SPOTIFY_CLIENT_ID)",
        default=existing.get("SPOTIFY_CLIENT_ID", ""),
        show_default=bool(existing.get("SPOTIFY_CLIENT_ID")),
    )
    spotify_secret = click.prompt(
        "  Client secret (SPOTIFY_CLIENT_SECRET)",
        default=existing.get("SPOTIFY_CLIENT_SECRET", ""),
        show_default=bool(existing.get("SPOTIFY_CLIENT_SECRET")),
        hide_input=True, prompt_suffix=": ",
    ) if spotify_id else ""
    spotify_playlist_raw = click.prompt(
        "  Playlist URL or ID (SPOTIFY_PLAYLIST_ID)",
        default=existing.get("SPOTIFY_PLAYLIST_ID", ""),
        show_default=bool(existing.get("SPOTIFY_PLAYLIST_ID")),
    ) if spotify_id else ""
    spotify_playlist_id = _extract_playlist_id(spotify_playlist_raw) if spotify_playlist_raw else ""

    # YouTube
    click.echo("\n  ── YouTube (optional) ──")
    yt_playlist = click.prompt(
        "  Playlist URL (YOUTUBE_PLAYLIST_URL, blank to skip)",
        default=existing.get("YOUTUBE_PLAYLIST_URL", ""),
        show_default=bool(existing.get("YOUTUBE_PLAYLIST_URL")),
    )

    # RSS Podcast feeds
    click.echo("\n  ── RSS Podcast Feeds (optional) ──")
    click.echo("  Paste comma-separated RSS feed URLs — one per show.")
    click.echo("  Covers Spotify, Apple Podcasts, Pocket Casts, Overcast, any indie feed.")
    rss_podcast = click.prompt(
        "  RSS podcast feeds (RSS_PODCAST_FEEDS, blank to skip)",
        default=existing.get("RSS_PODCAST_FEEDS", ""),
        show_default=bool(existing.get("RSS_PODCAST_FEEDS")),
    )

    # Apple Music
    click.echo("\n  ── Apple Music (optional, requires developer token) ──")
    click.echo("  See: https://developer.apple.com/documentation/applemusicapi/generating_developer_tokens")
    am_playlist = click.prompt(
        "  Playlist URL (APPLE_MUSIC_PLAYLIST_URL, blank to skip)",
        default=existing.get("APPLE_MUSIC_PLAYLIST_URL", ""),
        show_default=bool(existing.get("APPLE_MUSIC_PLAYLIST_URL")),
    )
    am_token = ""
    if am_playlist:
        am_token = click.prompt(
            "  Developer JWT token (APPLE_MUSIC_DEV_TOKEN)",
            default=existing.get("APPLE_MUSIC_DEV_TOKEN", ""),
            show_default=False,
        )

    # ── 4. OPTIONAL ENRICHMENT ─────────────────────────────────────────────────
    _section("4 / 5  Optional Enrichment")
    click.echo("  FRED API key enables macro-indicator charts in every brief.")
    click.echo("  Free, instant: https://fred.stlouisfed.org/docs/api/api_key.html")
    fred_key = click.prompt(
        "  FRED API key (blank to skip)",
        default=existing.get("FRED_API_KEY", ""),
        show_default=bool(existing.get("FRED_API_KEY")),
    )

    # ── 5. SPOTIFY OAUTH ───────────────────────────────────────────────────────
    spotify_refresh_token = existing.get("SPOTIFY_REFRESH_TOKEN", "")
    if spotify_id and spotify_secret and not spotify_refresh_token:
        _section("5 / 5  Spotify OAuth")
        click.echo(
            "  Before proceeding, make sure this redirect URI is registered\n"
            "  in your Spotify Developer Dashboard for this app:\n\n"
            "    http://127.0.0.1:3000/discovery\n\n"
            "  (Stop the Gotenberg container briefly if port 3000 is taken:\n"
            "   docker stop podcastbrief-gotenberg  then restart after OAuth)\n"
        )
        if click.confirm("  Run Spotify OAuth now to get a refresh token?", default=True):
            try:
                # Temporarily set env vars so auth_spotify can pick them up.
                os.environ["SPOTIFY_CLIENT_ID"] = spotify_id
                os.environ["SPOTIFY_CLIENT_SECRET"] = spotify_secret
                from podcastbrief.jobs.auth_spotify import run_spotify_auth
                spotify_refresh_token = run_spotify_auth()
                _ok("Spotify OAuth successful — refresh token obtained")
            except Exception as exc:
                _warn(
                    f"OAuth failed: {exc}\n"
                    "  Run  podcastbrief auth-spotify --write-env  later to complete it."
                )
    else:
        _section("5 / 5  Spotify OAuth")
        if spotify_refresh_token:
            _ok("SPOTIFY_REFRESH_TOKEN already set — skipping OAuth")
        elif not spotify_id:
            _ok("Spotify not configured — skipping OAuth")

    # ── WRITE .ENV ─────────────────────────────────────────────────────────────
    _section("Writing .env")

    values: dict[str, str] = {}

    # Spotify
    if spotify_id:
        values["SPOTIFY_CLIENT_ID"] = spotify_id
    if spotify_secret:
        values["SPOTIFY_CLIENT_SECRET"] = spotify_secret
    if spotify_refresh_token:
        values["SPOTIFY_REFRESH_TOKEN"] = spotify_refresh_token
    if spotify_playlist_id:
        values["SPOTIFY_PLAYLIST_ID"] = spotify_playlist_id

    # Telegram
    if telegram_token:
        values["TELEGRAM_BOT_TOKEN"] = telegram_token
    if telegram_chats:
        values["TELEGRAM_CHAT_IDS"] = telegram_chats

    # LLM
    values["LLM_MODEL"] = llm_model
    values["OLLAMA_HOST"] = existing.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    # Services (keep existing overrides or use defaults)
    values["WHISPER_URL"] = existing.get("WHISPER_URL", "http://localhost:9000")
    values["WHISPER_TIMEOUT_SECONDS"] = existing.get("WHISPER_TIMEOUT_SECONDS", "1800")
    values["GOTENBERG_URL"] = existing.get("GOTENBERG_URL", "http://localhost:3000")

    # New sources
    if yt_playlist:
        values["YOUTUBE_PLAYLIST_URL"] = yt_playlist
    if rss_podcast:
        values["RSS_PODCAST_FEEDS"] = rss_podcast
    if am_playlist:
        values["APPLE_MUSIC_PLAYLIST_URL"] = am_playlist
    if am_token:
        values["APPLE_MUSIC_DEV_TOKEN"] = am_token

    # Enrichment
    if fred_key:
        values["FRED_API_KEY"] = fred_key

    # Preserve paths and log level
    values["NOTES_DIR"] = existing.get("NOTES_DIR", "./podcast_notes")
    values["PDF_OUT_DIR"] = existing.get("PDF_OUT_DIR", "./briefs")
    values["LOG_LEVEL"] = existing.get("LOG_LEVEL", "INFO")

    _write_env(env_path, values)
    _ok(f".env written to {env_path}")

    # ── NEXT STEPS ─────────────────────────────────────────────────────────────
    click.echo()
    click.echo(click.style("✓  Setup complete!", fg="green", bold=True))
    click.echo(
        "\nNext steps:\n\n"
        "  Test the pipeline (one-off daily run):\n"
        "    podcastbrief run-daily\n\n"
        "  Start the scheduler + Telegram bot:\n"
        "    podcastbrief serve\n\n"
        "  (macOS) Install as a launchd service that survives reboots:\n"
        "    ./scripts/install-launchd.sh\n\n"
        "  Re-run setup at any time to update credentials:\n"
        "    podcastbrief setup\n"
    )
