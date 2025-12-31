"""
M3U Playlist Exporter - Exports playlists to M3U format for foobar2000
"""
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class M3UExporter:
    """Exports playlists to M3U format"""

    def __init__(self, export_path: str):
        """
        Initialize M3U exporter

        Args:
            export_path: Directory to save M3U files
        """
        self.export_path = Path(export_path)
        self._ensure_export_directory()
        logger.info(f"Initialized M3U exporter: {self.export_path}")

    def _ensure_export_directory(self):
        """Create export directory if it doesn't exist"""
        try:
            self.export_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Export directory ready: {self.export_path}")
        except Exception as e:
            logger.error(f"Failed to create export directory: {e}")
            raise

    def export_playlist(self, title: str, tracks: List[Dict[str, Any]], library_client, *, sonic_variant: str = "raw") -> str:
        """
        Export a playlist to M3U format

        Args:
            title: Playlist title (used as filename)
            tracks: List of track dictionaries with rating_key
            library_client: LibraryClient instance to fetch file paths

        Returns:
            Path to created M3U file
        """
        logger.info(f"Exporting playlist: {title}")

        # Sanitize filename
        variant_suffix = f"_sonic-{sonic_variant}" if sonic_variant and sonic_variant != "raw" else ""
        safe_filename = self._sanitize_filename(title + variant_suffix)
        m3u_path = self.export_path / f"{safe_filename}.m3u8"

        # Get file paths for all tracks
        track_paths = []
        for track in tracks:
            # Try to get file_path directly from track (local mode)
            file_path = track.get('file_path')

            # Fallback to fetching from library_client (legacy mode)
            if not file_path and library_client:
                rating_key = track.get('rating_key')
                if rating_key:
                    file_path = library_client.get_track_file_path(rating_key)

            if file_path:
                track_paths.append({
                    'path': file_path,
                    'artist': track.get('artist', 'Unknown'),
                    'title': track.get('title', 'Unknown'),
                    'duration': track.get('duration', 0)
                })

        if not track_paths:
            logger.warning(f"No valid file paths found for playlist: {title}")
            return None

        # Write M3U8 file (UTF-8 encoded extended M3U)
        try:
            with open(m3u_path, 'w', encoding='utf-8') as f:
                # Write M3U header
                f.write('#EXTM3U\n')

                # Write each track
                for track_info in track_paths:
                    # Duration in seconds (stored in milliseconds)
                    duration_sec = track_info['duration'] // 1000 if track_info['duration'] else -1

                    # Write EXTINF line with track info
                    f.write(f"#EXTINF:{duration_sec},{track_info['artist']} - {track_info['title']}\n")
                    f.write(f"#EXTVARIANT:{sonic_variant}\n")

                    # Write file path
                    f.write(f"{track_info['path']}\n")

            logger.info(f"Exported {len(track_paths)} tracks to: {m3u_path}")
            return str(m3u_path)

        except Exception as e:
            logger.error(f"Failed to write M3U file: {e}")
            return None

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters

        Args:
            filename: Original filename

        Returns:
            Safe filename
        """
        # Replace invalid Windows filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')

        return filename

    def cleanup_old_playlists(self, keep_prefixes: List[str] = None):
        """
        Clean up old M3U files from export directory

        Args:
            keep_prefixes: List of filename prefixes to keep (e.g., ['Auto'])
        """
        if not keep_prefixes:
            logger.debug("No cleanup - keep_prefixes not specified")
            return

        try:
            deleted_count = 0
            for m3u_file in self.export_path.glob('*.m3u*'):
                # Check if file should be kept
                should_keep = any(m3u_file.stem.startswith(prefix) for prefix in keep_prefixes)

                if not should_keep:
                    m3u_file.unlink()
                    logger.debug(f"Deleted old M3U: {m3u_file.name}")
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old M3U files")

        except Exception as e:
            logger.error(f"Failed to cleanup old playlists: {e}")


# Example usage
if __name__ == "__main__":
    logger.info("M3U Exporter module loaded")
