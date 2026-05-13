from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
import click

from podcastbrief.core.config import load_settings


@click.group()
def cli() -> None:
    """Podcast Brief — modular daily podcast → morning brief pipeline."""


@cli.command("run-daily")
@click.option("--dry-run", is_flag=True, help="Skip notify and persist; useful for end-to-end check.")
@click.option("--hours", default=24, type=int, help="Look-back window for new episodes.")
def run_daily_cmd(dry_run: bool, hours: int) -> None:
    from podcastbrief.jobs.daily import run_daily

    n = run_daily(dry_run=dry_run, hours=hours)
    click.echo(f"Processed {n} episode(s).")


@cli.command("cleanup-notes")
def cleanup_notes_cmd() -> None:
    from podcastbrief.jobs.cleanup import run_cleanup

    n = run_cleanup()
    click.echo(f"Cleared {n} note(s).")


@cli.command("auth-spotify")
@click.option(
    "--write-env",
    is_flag=True,
    help="Append/replace SPOTIFY_REFRESH_TOKEN in ./.env automatically.",
)
@click.option(
    "--env-path",
    type=click.Path(path_type=Path),
    default=Path(".env"),
    help="Path to .env file (only used with --write-env).",
)
def auth_spotify_cmd(write_env: bool, env_path: Path) -> None:
    """One-time OAuth flow to obtain a Spotify refresh token.

    Uses the registered redirect URI http://127.0.0.1:3000/discovery — make sure
    that URI is in your Spotify Developer Dashboard for this app.
    """
    from podcastbrief.jobs.auth_spotify import (
        REDIRECT_URI,
        run_spotify_auth,
        update_env_file,
    )

    click.echo(f"Required redirect URI: {REDIRECT_URI}")
    token = run_spotify_auth()
    click.echo("\n=== SUCCESS ===\n")
    if write_env:
        update_env_file(env_path, token)
        click.echo(f"Wrote SPOTIFY_REFRESH_TOKEN to {env_path}")
    else:
        click.echo("Add this line to your .env file:\n")
        click.echo(f"SPOTIFY_REFRESH_TOKEN={token}\n")


@cli.command("run-bot")
def run_bot_cmd() -> None:
    """Run the Telegram RAG bot (long-running)."""
    from podcastbrief.jobs.bot import run_bot

    run_bot()


@cli.command("test-brief")
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--show", required=True, help="Show name (for context).")
@click.option("--title", required=True, help="Episode title.")
@click.option("--artwork", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=Path("./test_brief.pdf"))
def test_brief_cmd(audio_path: Path, show: str, title: str, artwork: Path | None, out: Path) -> None:
    """Run extraction + interrogation + render against a local audio file. Writes PDF to --out."""
    from podcastbrief.adapters.whisper_http import WhisperHttpTranscriber
    from podcastbrief.adapters.ollama_gemma import OllamaGemma
    from podcastbrief.adapters.gotenberg_renderer import GotenbergRenderer
    from podcastbrief.briefing.extractor import extract_structure
    from podcastbrief.briefing.interrogator import interrogate
    from podcastbrief.briefing.schemas import RenderInput

    s = load_settings()
    logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))

    audio_bytes = audio_path.read_bytes()
    art_bytes = artwork.read_bytes() if artwork else None

    whisper = WhisperHttpTranscriber(
        base_url=s.whisper_url, timeout_seconds=s.whisper_timeout_seconds
    )
    llm = OllamaGemma(host=s.ollama_host, model=s.llm_model)
    renderer = GotenbergRenderer(base_url=s.gotenberg_url)

    transcript = whisper.transcribe(audio_bytes, filename=audio_path.name)
    structure = extract_structure(
        llm=llm, transcript=transcript, show_name=show, episode_title=title, artwork_png=art_bytes
    )
    brief = interrogate(llm=llm, structure=structure, transcript=transcript)
    render_in = RenderInput(
        brief=brief,
        show_name=show,
        episode_title=title,
        runtime="",
        pub_date=None,
        spotify_url=None,
        artwork_png=art_bytes,
        suggestions=[],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    pdf = renderer.render(render_in)
    out.write_bytes(pdf)
    click.echo(f"Wrote {out} ({len(pdf)} bytes)")


@cli.command("setup")
@click.option(
    "--env-path",
    type=click.Path(path_type=Path),
    default=Path(".env"),
    help="Path to write the .env file (default: ./.env).",
)
def setup_cmd(env_path: Path) -> None:
    """Interactive first-run wizard: checks dependencies, pulls models, writes .env.

    Re-run at any time to update credentials or add new content sources.
    """
    from podcastbrief.jobs.setup import run_setup

    # Resolve repo root as two levels up from this file's directory.
    repo_root = Path(__file__).resolve().parent.parent
    run_setup(env_path=env_path.resolve(), repo_root=repo_root)


@cli.command("serve")
def serve_cmd() -> None:
    """Run scheduler (daily 02:00, monthly 1st 03:00) + Telegram bot in one process."""
    import asyncio
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    from podcastbrief.jobs.daily import run_daily
    from podcastbrief.jobs.cleanup import run_cleanup
    from podcastbrief.jobs.bot import run_bot

    s = load_settings()
    logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))

    sched = BackgroundScheduler()
    sched.add_job(run_daily, CronTrigger(hour=2, minute=0), id="daily")
    sched.add_job(run_cleanup, CronTrigger(day=1, hour=3, minute=0), id="cleanup")
    sched.start()
    click.echo("Scheduler started. Running Telegram bot in foreground.")
    try:
        run_bot()
    finally:
        sched.shutdown()


if __name__ == "__main__":
    cli()
