"""Shared yt-dlp binary locator used by youtube_source and youtube_feed."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def yt_dlp_path() -> str | None:
    """Return path to yt-dlp binary, or None if not found."""
    candidates = [
        os.path.expanduser("~/.local/bin/yt-dlp"),
        "/opt/homebrew/bin/yt-dlp",
        "/usr/local/bin/yt-dlp",
    ]
    for p in candidates:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return shutil.which("yt-dlp")
