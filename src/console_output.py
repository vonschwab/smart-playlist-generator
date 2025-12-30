"""
Console output formatting for Playlist Generator.

Provides beautiful, organized console output with clear visual hierarchy.
All user-facing output should go through this module.
"""
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Box-drawing characters for visual structure
BOX_H = "─"  # Horizontal line
BOX_V = "│"  # Vertical line
BOX_TL = "┌"  # Top-left corner
BOX_TR = "┐"  # Top-right corner
BOX_BL = "└"  # Bottom-left corner
BOX_BR = "┘"  # Bottom-right corner
BOX_T = "┬"  # T-junction (top)
BOX_B = "┴"  # T-junction (bottom)
BOX_L = "├"  # T-junction (left)
BOX_R = "┤"  # T-junction (right)
BOX_X = "┼"  # Cross

# Section markers
BULLET = "•"
ARROW = "→"
CHECK = "✓"
CROSS = "✗"
STAR = "★"

# Width for formatted output
WIDTH = 70


def _safe_print(text: str = "") -> None:
    """Print with UTF-8 encoding safety."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Unicode
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


def header(title: str, subtitle: str = "") -> None:
    """
    Print a major section header with box drawing.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PLAYLIST GENERATOR                                                  │
    │  Generating playlist for: Radiohead                                  │
    └──────────────────────────────────────────────────────────────────────┘
    """
    inner_width = WIDTH - 2

    _safe_print()
    _safe_print(BOX_TL + BOX_H * inner_width + BOX_TR)
    _safe_print(f"{BOX_V}  {title:<{inner_width - 3}}{BOX_V}")
    if subtitle:
        _safe_print(f"{BOX_V}  {subtitle:<{inner_width - 3}}{BOX_V}")
    _safe_print(BOX_BL + BOX_H * inner_width + BOX_BR)


def section(title: str) -> None:
    """
    Print a section divider with title.

    ─── CONFIGURATION ─────────────────────────────────────────────────────
    """
    title_part = f" {title} "
    padding = WIDTH - len(title_part) - 3
    _safe_print()
    _safe_print(BOX_H * 3 + title_part + BOX_H * max(0, padding))


def subsection(title: str) -> None:
    """
    Print a lighter subsection header.

    ▸ Track Selection
    """
    _safe_print(f"\n  {ARROW} {title}")


def divider() -> None:
    """Print a simple divider line."""
    _safe_print(BOX_H * WIDTH)


def blank() -> None:
    """Print a blank line."""
    _safe_print()


def info(label: str, value: Any, indent: int = 0) -> None:
    """
    Print a labeled value.

      Mode:        dynamic
      Tracks:      30
    """
    prefix = "  " * indent
    _safe_print(f"{prefix}  {label + ':':<14} {value}")


def stat(label: str, value: Any, total: Optional[int] = None, indent: int = 0) -> None:
    """
    Print a statistic with optional percentage.

      Sonic:       25 tracks (83%)
    """
    prefix = "  " * indent
    if total and total > 0:
        pct = (value / total) * 100
        _safe_print(f"{prefix}  {label + ':':<14} {value} ({pct:.0f}%)")
    else:
        _safe_print(f"{prefix}  {label + ':':<14} {value}")


def bullet(text: str, indent: int = 0) -> None:
    """Print a bullet point."""
    prefix = "  " * indent
    _safe_print(f"{prefix}  {BULLET} {text}")


def track_line(index: int, artist: str, title: str, marker: str = "") -> None:
    """
    Print a formatted track line.

    01. Radiohead - Karma Police
    """
    track_str = f"{index:02d}. {artist} - {title}"
    if marker:
        track_str = f"{track_str} [{marker}]"
    _safe_print(f"    {track_str}")


def success(message: str) -> None:
    """Print a success message."""
    _safe_print(f"\n  {CHECK} {message}")


def error(message: str) -> None:
    """Print an error message."""
    _safe_print(f"\n  {CROSS} {message}")


def warning(message: str) -> None:
    """Print a warning message."""
    _safe_print(f"\n  ! {message}")


def progress(current: int, total: int, label: str = "") -> None:
    """
    Print a progress indicator.

    [████████████░░░░░░░░] 60% Processing tracks...
    """
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = (current / total * 100) if total > 0 else 0

    status = f"[{bar}] {pct:3.0f}%"
    if label:
        status = f"{status} {label}"

    # Use carriage return for in-place updates
    sys.stdout.write(f"\r  {status}")
    sys.stdout.flush()
    if current >= total:
        _safe_print()  # New line when complete


def table_row(cols: List[Tuple[str, int]], separator: str = " ") -> None:
    """Print a table row with fixed-width columns."""
    parts = []
    for text, width in cols:
        if len(text) > width:
            text = text[:width-1] + "…"
        parts.append(f"{text:<{width}}")
    _safe_print("    " + separator.join(parts))


class PlaylistReport:
    """
    Structured report for a generated playlist.

    Collects all relevant data and presents it in a unified, beautiful format.
    """

    def __init__(self, title: str, artist_name: str = ""):
        self.title = title
        self.artist_name = artist_name
        self.tracks: List[Dict[str, Any]] = []
        self.duration_ms: int = 0
        self.mode: str = "dynamic"
        self.dry_run: bool = False
        self.export_path: Optional[str] = None
        self.plex_exported: bool = False
        self.stats: Dict[str, Any] = {}
        self.edge_scores: Dict[str, float] = {}
        self.start_time: datetime = datetime.now()

    def set_tracks(self, tracks: List[Dict[str, Any]]) -> None:
        """Set the track list and compute derived stats."""
        self.tracks = tracks
        self.duration_ms = sum(t.get('duration', 0) for t in tracks)

        # Compute unique artists
        artists = set()
        for t in tracks:
            if t.get('artist'):
                artists.add(t['artist'])
        self.stats['unique_artists'] = len(artists)

        # Compute seed artist stats if applicable
        if self.artist_name:
            seed_norm = self.artist_name.strip().lower()
            seed_count = sum(
                1 for t in tracks
                if (t.get('artist') or '').strip().lower() == seed_norm
            )
            self.stats['seed_tracks'] = seed_count
            self.stats['seed_percentage'] = (seed_count / len(tracks) * 100) if tracks else 0

    def set_edge_scores(self, scores: Dict[str, float]) -> None:
        """Set transition quality scores."""
        self.edge_scores = scores

    def print_header(self) -> None:
        """Print the report header."""
        if self.dry_run:
            header("PLAYLIST PREVIEW", f"Seed: {self.artist_name}" if self.artist_name else "")
        else:
            header("PLAYLIST GENERATED", f"Seed: {self.artist_name}" if self.artist_name else "")

    def print_config(self) -> None:
        """Print configuration section."""
        section("CONFIGURATION")
        info("Mode", self.mode)
        info("Target", f"{len(self.tracks)} tracks")
        if self.artist_name:
            info("Seed Artist", self.artist_name)

    def print_tracklist(self, show_all: bool = False) -> None:
        """Print the track listing."""
        section("TRACKLIST")

        if not self.tracks:
            bullet("No tracks generated")
            return

        # Show first 10, last 3, or all if requested
        if show_all or len(self.tracks) <= 15:
            for i, t in enumerate(self.tracks, 1):
                artist = t.get('artist', 'Unknown')
                title = t.get('title', 'Unknown')
                track_line(i, artist, title)
        else:
            # First 8
            for i, t in enumerate(self.tracks[:8], 1):
                artist = t.get('artist', 'Unknown')
                title = t.get('title', 'Unknown')
                track_line(i, artist, title)

            # Ellipsis
            _safe_print(f"    ... ({len(self.tracks) - 11} more tracks) ...")

            # Last 3
            for i, t in enumerate(self.tracks[-3:], len(self.tracks) - 2):
                artist = t.get('artist', 'Unknown')
                title = t.get('title', 'Unknown')
                track_line(i, artist, title)

    def print_statistics(self) -> None:
        """Print playlist statistics."""
        section("STATISTICS")

        duration_min = self.duration_ms / 1000 / 60
        info("Duration", f"{duration_min:.1f} minutes")
        info("Tracks", len(self.tracks))
        info("Artists", self.stats.get('unique_artists', 0))

        if self.artist_name and 'seed_tracks' in self.stats:
            seed_count = self.stats['seed_tracks']
            seed_pct = self.stats['seed_percentage']
            info("Seed Tracks", f"{seed_count} ({seed_pct:.0f}%)")

        # Edge score summary if available
        if self.edge_scores:
            blank()
            subsection("Transition Quality")
            for label, value in self.edge_scores.items():
                if isinstance(value, float):
                    info(label, f"{value:.3f}", indent=1)

    def print_export(self) -> None:
        """Print export information."""
        if self.dry_run:
            section("DRY RUN")
            warning("No files were created")
            bullet("Run without --dry-run to export playlist")
            return

        section("EXPORT")

        if self.export_path:
            success(f"M3U: {self.export_path}")

        if self.plex_exported:
            success("Plex: Playlist synced")

    def print_footer(self) -> None:
        """Print the report footer."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        blank()
        divider()
        _safe_print(f"  Generated in {elapsed:.1f}s at {datetime.now().strftime('%H:%M:%S')}")
        divider()
        blank()

    def print_full(self, show_all_tracks: bool = False) -> None:
        """Print the complete report."""
        self.print_header()
        self.print_config()
        self.print_tracklist(show_all=show_all_tracks)
        self.print_statistics()
        self.print_export()
        self.print_footer()


class BatchReport:
    """Report for batch playlist generation."""

    def __init__(self, count: int):
        self.target_count = count
        self.created: List[Dict[str, Any]] = []
        self.failed: List[str] = []
        self.start_time: datetime = datetime.now()

    def add_created(self, title: str, track_count: int, duration_min: float) -> None:
        """Record a successfully created playlist."""
        self.created.append({
            'title': title,
            'tracks': track_count,
            'duration': duration_min
        })

    def add_failed(self, reason: str) -> None:
        """Record a failed playlist."""
        self.failed.append(reason)

    def print_summary(self) -> None:
        """Print batch generation summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        header("BATCH GENERATION COMPLETE", f"{len(self.created)}/{self.target_count} playlists")

        if self.created:
            section("CREATED PLAYLISTS")

            total_tracks = 0
            total_duration = 0.0

            for p in self.created:
                bullet(f"{p['title']} ({p['tracks']} tracks, {p['duration']:.0f} min)")
                total_tracks += p['tracks']
                total_duration += p['duration']

            section("TOTALS")
            info("Playlists", len(self.created))
            info("Tracks", total_tracks)
            info("Duration", f"{total_duration:.0f} minutes")

        if self.failed:
            section("FAILED")
            for reason in self.failed:
                bullet(reason)

        blank()
        divider()
        _safe_print(f"  Completed in {elapsed:.1f}s")
        divider()
        blank()


def print_startup_banner() -> None:
    """Print the application startup banner."""
    _safe_print()
    _safe_print("  ┌─────────────────────────────────────┐")
    _safe_print("  │     AI PLAYLIST GENERATOR           │")
    _safe_print("  │     Sonic + Genre Intelligence      │")
    _safe_print("  └─────────────────────────────────────┘")
    _safe_print()


def print_initialization(config_items: List[Tuple[str, str]]) -> None:
    """Print initialization status."""
    section("INITIALIZATION")
    for label, status in config_items:
        info(label, status)


def print_error_and_exit(message: str, details: Optional[str] = None) -> None:
    """Print an error message and exit."""
    blank()
    error(message)
    if details:
        _safe_print(f"    {details}")
    blank()
    sys.exit(1)
