from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from podcastbrief.core.models import BriefArtifacts, Episode
from podcastbrief.briefing.extractor import extract_structure
from podcastbrief.briefing.interrogator import interrogate
from podcastbrief.briefing.schemas import BriefFinal, RenderInput
from podcastbrief.core.enrichment import run_enrichers, write_annotations_to_cache
from podcastbrief.ports.enricher import Enricher, EnrichmentResult
from podcastbrief.ports.feed import FeedResolver
from podcastbrief.ports.images import ImageProvider
from podcastbrief.ports.llm import LLM
from podcastbrief.ports.notes import NoteStore
from podcastbrief.ports.notifier import Notifier
from podcastbrief.ports.recommender import Recommender
from podcastbrief.ports.renderer import BriefRenderer
from podcastbrief.ports.source import PodcastSource
from podcastbrief.ports.transcriber import Transcriber

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    source: PodcastSource
    feed: FeedResolver
    transcriber: Transcriber
    llm: LLM
    images: ImageProvider
    renderer: BriefRenderer
    notifier: Notifier
    notes: NoteStore
    recommender: Recommender
    chat_ids: list[str]
    audio_downloader: callable  # (AudioRef) -> bytes
    pdf_out_dir: Path | None = None
    enrichers: list[Enricher] | None = None
    notes_dir: Path | None = None
    # Additional (source, feed_resolver, audio_downloader) bundles.
    # Each tuple drives an independent content source (YouTube, RSS, Apple Music…)
    # that runs alongside the primary Spotify source in run_daily.
    extra_source_bundles: list[tuple] = field(default_factory=list)

    def run_daily(self, *, hours: int = 24, dry_run: bool = False) -> int:
        # Collect all (source, feed, downloader) bundles — primary first.
        bundles = [(self.source, self.feed, self.audio_downloader)]
        bundles.extend(self.extra_source_bundles)

        already = self._existing_episode_ids()
        n_done = 0
        any_found = False

        for source, feed, downloader in bundles:
            try:
                episodes = source.list_recent_episodes(hours=hours)
            except Exception as exc:
                log.exception("Source listing failed (%s): %s", type(source).__name__, exc)
                continue

            if not episodes:
                continue
            any_found = True

            fresh = [ep for ep in episodes if ep.episode_id not in already]
            skipped = len(episodes) - len(fresh)
            if skipped:
                log.info(
                    "%s: dedup skipping %d episode(s) already in vault",
                    type(source).__name__, skipped,
                )

            for ep in fresh:
                try:
                    self._process_episode(ep, feed=feed, audio_downloader=downloader, dry_run=dry_run)
                    already.add(ep.episode_id)  # prevent cross-source duplicates
                    n_done += 1
                except Exception as exc:
                    log.exception(
                        "Episode failed: %s — %s: %s", ep.show_name, ep.name, exc
                    )
                    if not dry_run:
                        self._notify_all(
                            f"⚠️ Failed to process: {ep.show_name} — {ep.name}\n{exc}"
                        )

        if not any_found:
            log.info("No new episodes in last %dh", hours)
            self._notify_all("No new podcast episodes today.")
            return 0

        if not dry_run and n_done > 0:
            self._notify_all(f"All {n_done} podcast(s) summarized and sent out.")
        return n_done

    def run_latest(self, *, dry_run: bool = False) -> Episode | None:
        """Reprocess the most-recently-added episode across all sources.

        Checks the primary source first, then extra bundles, and picks the
        overall most-recent episode. Always runs end-to-end (fresh download,
        fresh Whisper, fresh Gemma passes, fresh PDF). Used by /run command.
        """
        bundles = [(self.source, self.feed, self.audio_downloader)]
        bundles.extend(self.extra_source_bundles)

        best_ep = None
        best_feed = self.feed
        best_dl = self.audio_downloader

        for source, feed, downloader in bundles:
            try:
                eps = source.list_recent_episodes(hours=24 * 365)
            except Exception as exc:
                log.warning("run_latest: source %s failed: %s", type(source).__name__, exc)
                continue
            if not eps:
                continue
            candidate = max(eps, key=lambda e: e.added_at)
            if best_ep is None or candidate.added_at > best_ep.added_at:
                best_ep = candidate
                best_feed = feed
                best_dl = downloader

        if best_ep is None:
            log.info("All playlists/feeds are empty; nothing to run.")
            return None
        log.info("/run latest: %s — %s", best_ep.show_name, best_ep.name)
        self._process_episode(
            best_ep, feed=best_feed, audio_downloader=best_dl,
            dry_run=dry_run, force=True,
        )
        return best_ep

    def _enrich_brief(
        self,
        *,
        brief: BriefFinal,
        episode_id: str,
        episode_pub_date: str | None,
        accent_hex: str,
        force: bool,
    ) -> EnrichmentResult:
        if not self.enrichers or not self.notes_dir:
            return EnrichmentResult()
        try:
            result = run_enrichers(
                self.enrichers,
                notes_dir=self.notes_dir,
                episode_id=episode_id,
                market_entities=brief.market_entities,
                macro_indicators=brief.macro_indicators,
                named_entities=brief.named_entities,
                episode_pub_date=episode_pub_date,
                accent_hex=accent_hex,
                force=force,
            )
        except Exception as e:
            log.warning("Enrichment failed: %s", e)
            return EnrichmentResult()

        # Pass 3 — Gemma 4 grounding annotations across modalities.
        try:
            from podcastbrief.briefing.grounder import ground_enrichment
            result = ground_enrichment(llm=self.llm, brief=brief, enrichment=result)
            write_annotations_to_cache(self.notes_dir, episode_id, result)
        except Exception as e:
            log.warning("Grounding annotations failed: %s", e)
        return result

    @staticmethod
    def _accent_for(artwork: bytes | None) -> str:
        """Best-effort dominant-color sampling for chart styling. Falls back to
        the brand accent if Pillow isn't happy."""
        if not artwork:
            return "#6c63ff"
        try:
            from podcastbrief.adapters.gotenberg_renderer import _dominant_color_hex
            return _dominant_color_hex(artwork)
        except Exception:
            return "#6c63ff"

    def _existing_episode_ids(self) -> set[str]:
        ids: set[str] = set()
        for note in self.notes.list_recent(limit=500):
            meta = note.metadata or {}
            eid = str(meta.get("episode_id") or "").strip()
            if not eid:
                # Backward compat: older notes only stored the spotify URL.
                # Extract the trailing episode ID segment.
                spotify_url = str(meta.get("spotify") or "")
                if "/episode/" in spotify_url:
                    eid = spotify_url.rstrip("/").rsplit("/episode/", 1)[-1].split("?")[0]
            if eid:
                ids.add(eid)
        return ids

    def _process_episode(
        self,
        ep: Episode,
        *,
        dry_run: bool,
        force: bool = False,
        feed: FeedResolver | None = None,
        audio_downloader: callable | None = None,
    ) -> None:
        _feed = feed if feed is not None else self.feed
        _downloader = audio_downloader if audio_downloader is not None else self.audio_downloader

        log.info("Processing: %s — %s (force=%s)", ep.show_name, ep.name, force)

        audio_ref = _feed.find_audio(ep)
        audio_bytes = _downloader(audio_ref)
        transcript = self.transcriber.transcribe(
            audio_bytes, filename=f"{ep.episode_id}.mp3", force=force
        )
        artwork = self.images.artwork(ep)

        structure = extract_structure(
            llm=self.llm,
            transcript=transcript,
            show_name=ep.show_name,
            episode_title=audio_ref.title,
            artwork_png=artwork,
        )
        brief = interrogate(llm=self.llm, structure=structure, transcript=transcript)
        # Carry transcript language through to the rest of the pipeline.
        brief.language = (transcript.language or "en").split("-")[0].lower()

        rec_query = " ".join(brief.topics[:3]) or " ".join(brief.go_deeper[:2]) or ep.show_name
        suggestions = self.recommender.similar(query=rec_query, limit=5)

        # ---- Enrichment (item 2) + grounding annotations (item 3) ----
        enrichment = self._enrich_brief(
            brief=brief,
            episode_id=ep.episode_id,
            episode_pub_date=audio_ref.pub_date,
            accent_hex=self._accent_for(artwork),
            force=force,
        )

        render_in = RenderInput(
            brief=brief,
            show_name=ep.show_name,
            episode_title=audio_ref.title,
            runtime=_fmt_runtime(ep.duration_ms),
            pub_date=audio_ref.pub_date,
            spotify_url=ep.spotify_url,
            artwork_png=artwork,
            suggestions=[
                {"title": s.title, "show": s.show, "url": s.url} for s in suggestions
            ],
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            enrichment=enrichment,
        )

        pdf_bytes = self.renderer.render(render_in)
        artifacts = self._build_artifacts(brief, render_in, pdf_bytes, transcript_text=transcript.text)

        if dry_run:
            log.info("[dry-run] Would notify and persist: %s", artifacts.file_stem)
            if self.pdf_out_dir:
                self._save_pdf_locally(artifacts)
            return

        if self.pdf_out_dir:
            self._save_pdf_locally(artifacts)

        for chat_id in self.chat_ids:
            self.notifier.send_document(
                chat_id=chat_id,
                data=artifacts.pdf_bytes,
                filename=f"{artifacts.file_stem}.pdf",
                caption=f"{ep.show_name} — {audio_ref.title}",
            )

        self.notes.save(
            file_stem=artifacts.file_stem,
            body=artifacts.markdown,
            metadata={
                "title": audio_ref.title,
                "show": ep.show_name,
                "date": audio_ref.pub_date or "",
                "spotify": ep.spotify_url,
                "episode_id": ep.episode_id,
                "language": brief.language,
                "processed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "topics": brief.topics,
            },
        )

    def _build_artifacts(
        self,
        brief: BriefFinal,
        render_in: RenderInput,
        pdf_bytes: bytes,
        *,
        transcript_text: str,
    ) -> BriefArtifacts:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe_title = re.sub(r"[^a-zA-Z0-9 ]", "", render_in.episode_title)
        safe_title = re.sub(r"\s+", "-", safe_title)[:60].strip("-")
        stem = f"{date}_{safe_title}" if safe_title else f"{date}_episode"

        md_lines: list[str] = []
        md_lines.append(f"# {render_in.episode_title}")
        md_lines.append(f"**Show:** {render_in.show_name}")
        if render_in.pub_date:
            md_lines.append(f"**Date:** {render_in.pub_date}")
        if render_in.spotify_url:
            md_lines.append(f"**Spotify:** {render_in.spotify_url}")
        md_lines.append("")
        md_lines.append(f"## Headline\n{brief.headline}\n")
        md_lines.append(f"## TL;DR\n{brief.tldr}\n")
        if brief.thesis:
            md_lines.append(f"## Thesis\n{brief.thesis}\n")
        if brief.why_it_matters:
            md_lines.append("## Why It Matters")
            md_lines.extend(f"- {b}" for b in brief.why_it_matters)
            md_lines.append("")
        if brief.pull_quotes:
            md_lines.append("## Key Quotes")
            for q in brief.pull_quotes:
                ts = f" `{q.timestamp}`" if q.timestamp else ""
                md_lines.append(f"> {q.text}\n> — **{q.speaker}**{ts}")
                if q.context:
                    md_lines.append(f"> *{q.context}*")
                md_lines.append("")
        if brief.by_the_numbers:
            md_lines.append("## By The Numbers")
            for d in brief.by_the_numbers:
                src = f" ({d.source})" if d.source else ""
                md_lines.append(f"- **{d.stat}** — {d.label}{src} · {d.why_relevant}")
            md_lines.append("")
        if brief.predictions:
            md_lines.append("## Predictions")
            md_lines.extend(f"- {p}" for p in brief.predictions)
            md_lines.append("")
        if brief.counterpoints:
            md_lines.append("## Counterpoints")
            md_lines.extend(f"- {c}" for c in brief.counterpoints)
            md_lines.append("")
        if brief.resources_mentioned:
            md_lines.append("## Resources Mentioned")
            for r in brief.resources_mentioned:
                note = f" — {r.note}" if r.note else ""
                md_lines.append(f"- *{r.kind}* **{r.name}**{note}")
            md_lines.append("")
        if brief.action_items:
            md_lines.append("## Action Items")
            md_lines.extend(f"- [ ] {a}" for a in brief.action_items)
            md_lines.append("")
        if render_in.suggestions:
            md_lines.append("## Similar Episodes")
            for s in render_in.suggestions:
                md_lines.append(f"- [{s['title']}]({s['url']}) — {s['show']}")
            md_lines.append("")
        if brief.topics:
            md_lines.append("## Topics\n" + " ".join(f"#{t}" for t in brief.topics) + "\n")
        md_lines.append("## Transcript\n```\n" + transcript_text[:10000] + "\n```\n")

        markdown = "\n".join(md_lines)
        return BriefArtifacts(pdf_bytes=pdf_bytes, markdown=markdown, file_stem=stem)

    def _save_pdf_locally(self, artifacts: BriefArtifacts) -> None:
        if not self.pdf_out_dir:
            return
        self.pdf_out_dir.mkdir(parents=True, exist_ok=True)
        path = self.pdf_out_dir / f"{artifacts.file_stem}.pdf"
        path.write_bytes(artifacts.pdf_bytes)
        artifacts.pdf_path = path
        log.info("Saved PDF: %s", path)

    def _notify_all(self, text: str) -> None:
        for chat_id in self.chat_ids:
            try:
                self.notifier.send_text(chat_id=chat_id, text=text)
            except Exception as e:
                log.warning("Notify chat %s failed: %s", chat_id, e)


def _fmt_runtime(ms: int) -> str:
    if not ms:
        return ""
    secs = ms // 1000
    h, secs = divmod(secs, 3600)
    m, _ = divmod(secs, 60)
    return f"{h}h {m}m" if h else f"{m}m"
