# -*- coding: utf-8 -*-
"""
AI Playlist Generator - Main Application
Automatically generates AI-powered playlists based on listening history
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import sys
import argparse

from src.config_loader import Config
from src.local_library_client import LocalLibraryClient
from src.openai_client import OpenAIClient
from src.playlist_generator import PlaylistGenerator
from src.lastfm_client import LastFMClient
from src.track_matcher import TrackMatcher
from src.m3u_exporter import M3UExporter
from src.metadata_client import MetadataClient
from src.similarity.sonic_variant import resolve_sonic_variant
from src.similarity.sonic_variant import resolve_sonic_variant
from src.plex_exporter import PlexExporter


class PlaylistApp:
    """Main application orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml", ds_mode_override: Optional[str] = None):
        # Load configuration
        self.config = Config(config_path)
        self.ds_mode_override = ds_mode_override
        
        # Set up logging
        self._setup_logging()
        
        # Initialize library client
        self.logger.info("Initializing Playlist Generator")
        self.library = LocalLibraryClient(db_path="data/metadata.db")

        self.openai = OpenAIClient(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        # Initialize Last.FM client (history-only). Only create if credentials exist.
        self.lastfm = None
        if self.config.lastfm_api_key and self.config.lastfm_username:
            self.lastfm = LastFMClient(
                api_key=self.config.lastfm_api_key,
                username=self.config.lastfm_username
            )
        else:
            self.logger.info("Last.FM credentials missing; skipping Last.FM client (history will be local-only).")

        # Use the same metadata DB path as the library client to ensure normalized columns are present
        self.matcher = TrackMatcher(
            self.library,
            library_id=None,
            db_path=self.config.library_database_path,
        )
        sonic_cfg = self.config.get('playlists', 'sonic', default={}) or {}
        self.sonic_variant = resolve_sonic_variant(sonic_cfg.get("sim_variant"))

        # Initialize metadata database client (optional, for enhanced genre matching)
        self.metadata = None
        try:
            db_path = self.config.get('metadata', 'database_path', default='data/metadata.db')
            self.metadata = MetadataClient(db_path)
            self.logger.info(f"Metadata database connected: {db_path}")
        except Exception as e:
            self.logger.warning(f"Metadata database not available: {e}")
            self.logger.info("Dynamic mode will use local metadata as fallback")

        self.generator = PlaylistGenerator(
            self.library,
            self.config,
            lastfm_client=self.lastfm,
            track_matcher=self.matcher,
            metadata_client=self.metadata
        )

        # Initialize M3U exporter (always on with E:\PLAYLISTS fallback)
        export_path = self.config.get('playlists', 'm3u_export_path', default="E:\\PLAYLISTS")
        export_enabled = self.config.get('playlists', 'export_m3u', default=True)
        if not export_enabled:
            self.logger.warning("M3U export disabled in config; overriding to ensure playlists are written to disk.")

        self.m3u_exporter = None
        if export_path:
            try:
                self.m3u_exporter = M3UExporter(export_path)
                self.logger.info(f"M3U export enabled (always on): {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize M3U exporter at {export_path}: {e}")

        # Initialize Plex exporter (optional)
        plex_enabled = bool(self.config.get('plex', 'enabled', default=False))
        plex_base_url = self.config.get('plex', 'base_url', default=None)
        plex_token = os.getenv("PLEX_TOKEN") or self.config.get('plex', 'token', default=None)
        plex_section = self.config.get('plex', 'music_section', default=None)
        plex_verify_ssl = bool(self.config.get('plex', 'verify_ssl', default=True))
        plex_replace = bool(self.config.get('plex', 'replace_existing', default=True))
        plex_path_map = self.config.get('plex', 'path_map', default=None)

        self.plex_exporter = None
        if plex_enabled:
            if plex_base_url and plex_token:
                try:
                    self.plex_exporter = PlexExporter(
                        plex_base_url,
                        plex_token,
                        music_section=plex_section,
                        verify_ssl=plex_verify_ssl,
                        replace_existing=plex_replace,
                        path_map=plex_path_map,
                    )
                    self.logger.info("Plex export enabled")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Plex exporter: {e}")
            else:
                self.logger.warning("Plex export enabled but base_url/token not configured")
    
    def _setup_logging(self):
        """Configure logging"""
        log_level = self.config.get('logging', 'level', default='INFO')
        log_file = self.config.get('logging', 'file', default='playlist_generator.log')
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler with UTF-8 encoding and error handling
        import sys
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        # Set stream encoding to handle Unicode properly
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Set logging for other modules (using src. prefix for new package structure)
        for module in [
            'src.openai_client',
            'src.playlist_generator',
            'src.playlist.reporter',
            'src.playlist.filtering',
            'src.playlist.scoring',
            'src.playlist.diversity',
            'src.playlist.ordering',
            'src.playlist.history_analyzer',
            'src.playlist.candidate_generator',
            'src.lastfm_client',
            'src.track_matcher',
            'src.m3u_exporter',
            'src.plex_exporter',
            'src.metadata_client',
            'src.metadata_builder',
            'src.metadata_updater',
        ]:
            mod_logger = logging.getLogger(module)
            mod_logger.setLevel(getattr(logging, log_level))
            mod_logger.addHandler(console_handler)
            mod_logger.addHandler(file_handler)
    
    def cleanup_old_playlists(self) -> int:
        """
        Delete old auto-generated playlists
        
        Returns:
            Number of playlists deleted
        """
        self.logger.info("Cleaning up old playlists")
        
        name_prefix = self.config.get('playlists', 'name_prefix', default='Auto:')
        max_age_days = self.config.get('playlists', 'max_age_days', default=14)
        
        # Get all auto-generated playlists
        playlists = self.library.get_playlists(name_prefix=name_prefix)
        
        if not playlists:
            self.logger.info("No auto-generated playlists found")
            return 0
        
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = int(cutoff.timestamp())
        
        deleted_count = 0
        for playlist in playlists:
            # Check if playlist is too old
            created_at = playlist.get('created_at', 0)
            
            if created_at < cutoff_timestamp:
                playlist_id = playlist['id']
                playlist_title = playlist['title']
                
                try:
                    self.library.delete_playlist(playlist_id)
                    self.logger.info(f"Deleted old playlist: {playlist_title}")
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete playlist {playlist_title}: {e}")
        
        self.logger.info(f"Cleanup complete: {deleted_count} playlists deleted")
        return deleted_count
    
    def generate_playlists(self, dry_run: bool = False, dynamic: bool = False) -> List[Dict[str, str]]:
        """
        Main workflow: Generate new playlists

        Args:
            dry_run: If True, skip creating playlists and exporting M3U
            dynamic: If True, enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of created playlist metadata
        """
        self.logger.info("=" * 60)
        if dry_run:
            self.logger.info("Starting playlist generation workflow (DRY RUN - no playlists will be created)")
        else:
            self.logger.info("Starting playlist generation workflow")
        self.logger.info("=" * 60)

        # Step 1: Clean up old playlists (skip in dry-run mode)
        if not dry_run:
            self.cleanup_old_playlists()
        else:
            self.logger.info("Skipping cleanup (dry run mode)")

        # Step 2: Generate playlists with tracks and metadata
        playlist_count = self.config.get('playlists', 'count', default=3)
        playlists = self.generator.create_playlist_batch(
            playlist_count,
            dynamic=dynamic,
            ds_mode_override=self.ds_mode_override,
        )

        if not playlists:
            self.logger.warning("No playlists generated - insufficient data")
            return []

        # Step 3: Create playlists (skip in dry-run mode)
        name_prefix = self.config.get('playlists', 'name_prefix', default='Auto:')

        created_playlists = []

        for i, playlist_data in enumerate(playlists, 1):
            # Extract title and tracks from playlist data
            title = playlist_data['title']
            tracks = playlist_data['tracks']
            full_title = f"{name_prefix} {title}"

            # Calculate duration
            total_duration_ms = sum(t.get('duration', 0) for t in tracks)
            duration_minutes = total_duration_ms / 1000 / 60

            if dry_run:
                # Dry run - just log what would be created
                self.logger.info(f"Would create playlist {i}/{len(playlists)}: {full_title} ({len(tracks)} tracks, {duration_minutes:.1f} min)")
                created_playlists.append({
                    'title': full_title,
                    'id': 'dry-run',
                    'track_count': len(tracks)
                })
            else:
                self.logger.info(f"Exporting playlist {i}/{len(playlists)}: {full_title}")

                # Export to M3U (local mode)
                try:
                    if self.m3u_exporter:
                        m3u_path = self.m3u_exporter.export_playlist(full_title, tracks, self.library, sonic_variant=self.sonic_variant)
                        if m3u_path:
                            created_playlists.append({
                                'title': full_title,
                                'id': f'm3u_{i}',
                                'track_count': len(tracks)
                            })
                            self.logger.info(f"Exported: {full_title} ({len(tracks)} tracks) -> {m3u_path}")
                        else:
                            self.logger.error(f"Failed to export M3U for '{full_title}'")
                    else:
                        self.logger.warning(f"M3U exporter not configured, skipping '{full_title}'")

                except Exception as e:
                    self.logger.error(f"Failed to export playlist '{full_title}': {e}")

                # Export to Plex (optional)
                if self.plex_exporter:
                    try:
                        plex_key = self.plex_exporter.export_playlist(full_title, tracks)
                        if plex_key:
                            self.logger.info("Exported to Plex: %s (%d tracks)", full_title, len(tracks))
                        else:
                            self.logger.error("Failed to export Plex playlist '%s'", full_title)
                    except Exception as e:
                        self.logger.error("Plex export failed for '%s': %s", full_title, e)

        self.logger.info("=" * 60)
        if dry_run:
            self.logger.info(f"Dry run complete: {len(created_playlists)} playlists would be created")
        else:
            self.logger.info(f"Workflow complete: {len(created_playlists)} playlists created")
        self.logger.info("=" * 60)

        return created_playlists

    def _generate_summary_report(self, playlists: List[Dict], created_playlists: List[Dict]):
        """
        Generate and print comprehensive summary report

        Args:
            playlists: Playlist data with metadata from generator
            created_playlists: Successfully created playlists
        """
        print("\n" + "=" * 70)
        print("PLAYLIST GENERATION SUMMARY")
        print("=" * 70)

        # Basic statistics
        print(f"\nDate/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Playlists Created: {len(created_playlists)}/{len(playlists)}")

        # Cache statistics
        if hasattr(self.generator, 'artist_cache'):
            cache_stats = self.generator.artist_cache.get_cache_stats()
            print(f"\nArtist Similarity Cache:")
            print(f"  Total entries: {cache_stats['total_artists']}")
            print(f"  Fresh entries: {cache_stats['fresh_entries']}")
            print(f"  Expired entries: {cache_stats['expired_entries']}")

        # Playlist details
        if created_playlists:
            print(f"\nPlaylist Details:")
            total_tracks = 0
            all_artists = set()

            for i, (playlist_data, created) in enumerate(zip(playlists, created_playlists), 1):
                track_count = created['track_count']
                total_tracks += track_count
                artists = playlist_data.get('artists', ('Unknown', 'Unknown'))
                genres = playlist_data.get('genres', [])

                # Collect unique artists from tracks
                tracks = playlist_data.get('tracks', [])
                for track in tracks:
                    if track.get('artist'):
                        all_artists.add(track['artist'])

                print(f"\n  {i}. {created['title']}")
                print(f"     Artists: {artists[0]} + {artists[1]}")
                if genres:
                    print(f"     Genres: {', '.join(genres[:3])}")
                print(f"     Tracks: {track_count}")

            print(f"\nAggregate Statistics:")
            print(f"  Total tracks across all playlists: {total_tracks}")
            print(f"  Average tracks per playlist: {total_tracks / len(created_playlists):.1f}")
            print(f"  Unique artists featured: {len(all_artists)}")

            # Export location
            if self.m3u_exporter:
                m3u_path = self.config.get('playlists', 'm3u_export_path', 'Unknown')
                print(f"\nM3U Export Location: {m3u_path}")

        print("\n" + "=" * 70)

    def run(self, dry_run: bool = False, dynamic: bool = False):
        """Run the application"""
        try:
            # Store original playlist data with metadata
            playlist_count = self.config.get('playlists', 'count', default=3)

            # Generate playlists
            results = self.generate_playlists(dry_run=dry_run, dynamic=dynamic)

            # Get the playlist metadata from the generator
            # We need to re-run the generation to get metadata, or store it
            # For now, let's create a simplified summary
            if results:
                playlists = self.generator.create_playlist_batch(0)  # Get metadata without creating
                # Actually, we need a different approach. Let me store it during generation
                # For now, use a simpler summary

                print("\n" + "=" * 70)
                print("PLAYLIST GENERATION COMPLETE!")
                print("=" * 70)

                print(f"\nCreated {len(results)} new playlists:")
                for playlist in results:
                    print(f"  • {playlist['title']} ({playlist['track_count']} tracks)")

                # Cache stats
                if hasattr(self.generator, 'artist_cache'):
                    cache_stats = self.generator.artist_cache.get_cache_stats()
                    print(f"\nCache Statistics:")
                    print(f"  Cached artists: {cache_stats['total_artists']} ({cache_stats['fresh_entries']} fresh)")

                print("\nCheck your music player to enjoy your new playlists!")
                print("=" * 70 + "\n")
            else:
                print("\nNo playlists created. Check logs for details.")
            
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            print(f"\nError: {e}")
            print("Check the log file for details.\n")
            sys.exit(1)

    def run_single_artist(self, artist_name: str, track_count: int = 30, track_title: Optional[str] = None, dry_run: bool = False, dynamic: bool = False, verbose: bool = False, artist_only: bool = False):
        """
        Generate a single playlist for a specific artist.
        Splits business logic, presentation, and export for clarity.
        """
        try:
            self._configure_verbose_logging(verbose)
            self._log_single_artist_header(artist_name, dry_run)

            print(f"Generating playlist for: {artist_name}")
            print(f"Target tracks: {track_count}\n")

            playlist_data = self._generate_single_artist_playlist(artist_name, track_count, track_title, dynamic, verbose, artist_only)
            if not playlist_data:
                self._report_single_artist_failure(artist_name)
                return

            playlist_title = f"Auto: {artist_name}"
            total_duration_ms = sum(t.get('duration', 0) for t in playlist_data['tracks'])
            duration_minutes = total_duration_ms / 1000 / 60

            if dry_run:
                self._handle_dry_run_output(playlist_title, playlist_data, artist_name, duration_minutes)
            else:
                self._export_and_report_playlist(playlist_title, playlist_data, artist_name, duration_minutes)

        except Exception as e:
            self.logger.error(f"Error creating playlist for {artist_name}: {e}", exc_info=True)
            print(f"\nError: {e}")
            print("Check the log file for details.\n")
            sys.exit(1)

    def _configure_verbose_logging(self, verbose: bool) -> None:
        """Enable verbose logging for generator/similarity when requested."""
        if not verbose:
            return
        logging.getLogger('src.playlist_generator').setLevel(logging.DEBUG)
        logging.getLogger('src.similarity_calculator').setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers + self.logger.handlers:
            handler.setLevel(logging.DEBUG)
        self.logger.info("Verbose logging enabled")

    def _log_single_artist_header(self, artist_name: str, dry_run: bool) -> None:
        """Log banner for single-artist generation."""
        self.logger.info("=" * 60)
        if dry_run:
            self.logger.info(f"Generating playlist for artist: {artist_name} (DRY RUN)")
        else:
            self.logger.info(f"Generating playlist for artist: {artist_name}")
        self.logger.info("=" * 60)

    def _generate_single_artist_playlist(self, artist_name: str, track_count: int, track_title: Optional[str], dynamic: bool, verbose: bool, artist_only: bool) -> Optional[Dict[str, Any]]:
        """Generate playlist data for a single artist."""
        return self.generator.create_playlist_for_artist(
            artist_name,
            track_count,
            track_title=track_title,
            dynamic=dynamic,
            verbose=verbose,
            ds_mode_override=self.ds_mode_override,
            artist_only=artist_only,
        )

    def _report_single_artist_failure(self, artist_name: str) -> None:
        """User-facing messaging when no playlist could be created."""
        print(f"\nCould not create playlist for '{artist_name}'")
        print("Possible reasons:")
        print("  - Artist not found in your library")
        print("  - Artist has too few tracks")
        print("  - No similar tracks available\n")

    def _compute_seed_stats(self, playlist_data: Dict[str, Any], artist_name: str) -> Tuple[int, float]:
        """Return count and percentage of seed-artist tracks in playlist."""
        seed_norm = (artist_name or "").strip().casefold()
        seed_tracks = []
        for t in playlist_data['tracks']:
            a = (t.get('artist') or "").strip().casefold()
            if a == seed_norm:
                seed_tracks.append(t)
        seed_percentage = (len(seed_tracks) / len(playlist_data['tracks'])) * 100 if playlist_data['tracks'] else 0.0
        return len(seed_tracks), seed_percentage

    def _handle_dry_run_output(self, playlist_title: str, playlist_data: Dict[str, Any], artist_name: str, duration_minutes: float) -> None:
        """Display dry-run output without side effects."""
        track_count = len(playlist_data['tracks'])
        seed_count, seed_pct = self._compute_seed_stats(playlist_data, artist_name)

        self.logger.info(f"Would create playlist: {playlist_title} ({track_count} tracks, {duration_minutes:.1f} min)")
        print(f"\nDRY RUN - Would create: {playlist_title}")
        print(f"   Tracks: {track_count}")
        print(f"   Duration: {duration_minutes:.1f} minutes")
        print(f"   Seed artist ({artist_name}): {seed_count} tracks ({seed_pct:.1f}%)")
        print("\nDry run complete - no playlists created")

    def _export_and_report_playlist(self, playlist_title: str, playlist_data: Dict[str, Any], artist_name: str, duration_minutes: float) -> None:
        """Export playlist and show summary to the user."""
        track_count = len(playlist_data['tracks'])
        seed_count, seed_pct = self._compute_seed_stats(playlist_data, artist_name)

        print(f"\nCreated: {playlist_title} ({track_count} tracks)")
        print(f"  Duration: {duration_minutes:.1f} minutes")
        print(f"  Seed artist ({artist_name}): {seed_count} tracks ({seed_pct:.1f}%)")

        if not self.m3u_exporter:
            print("\nM3U export not configured")
            return

        m3u_path = self.m3u_exporter.export_playlist(playlist_title, playlist_data['tracks'], self.library, sonic_variant=self.sonic_variant)
        if m3u_path:
            self.logger.info(f"Exported to M3U: {m3u_path}")
            print(f"  Exported to: {m3u_path}")
            print("\nPlaylist exported! Import the M3U file to your music player.")
        else:
            print("\nFailed to export M3U file")

        if self.plex_exporter:
            try:
                plex_key = self.plex_exporter.export_playlist(playlist_title, playlist_data['tracks'])
                if plex_key:
                    self.logger.info("Exported to Plex: %s", playlist_title)
                    print("  Exported to Plex.")
                else:
                    print("  Plex export failed.")
            except Exception as e:
                self.logger.error("Plex export failed for '%s': %s", playlist_title, e)
                print("  Plex export failed.")


def main():
    """Entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate playlists based on listening history or a specific artist"
    )
    parser.add_argument(
        "--artist",
        type=str,
        help="Generate a single playlist for a specific artist (e.g., --artist \"Radiohead\")"
    )
    parser.add_argument(
        "--track",
        type=str,
        help="Optional: specify a seed track title for the artist (e.g., --track \"Life On Mars\")"
    )
    parser.add_argument(
        "--tracks",
        type=int,
        default=30,
        help="Number of tracks in the playlist (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview playlists without creating them or exporting M3U files"
    )
    parser.add_argument(
        "--artist-only",
        action="store_true",
        help="Restrict DS pipeline to the requested artist only (no discovery)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging: show detailed DS transition metrics and constraint enforcement"
    )
    parser.add_argument(
        "--ds-mode",
        choices=["narrow", "dynamic", "discover", "sonic_only"],
        help="Select DS pipeline mode (default from config playlists.ds_pipeline.mode)",
    )
    parser.add_argument(
        "--sonic-variant",
        choices=[
            "raw",
            "centered",
            "z",
            "z_clip",
            "whiten_pca",
            "robust_whiten",
            "tower_l2",
            "tower_robust",
            "tower_iqr",
            "tower_weighted",
            "tower_pca",
        ],
        help="Override sonic similarity variant for DS pipeline (env has highest priority).",
    )
    args = parser.parse_args()

    print("\nAI Playlist Generator\n")

    # Check for config file
    import os
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found")
        print("\nPlease create config.yaml with your API credentials.")
        print("See config.yaml template for reference.\n")
        sys.exit(1)

    # Run application
    try:
        if getattr(args, "sonic_variant", None):
            import os

            os.environ["SONIC_SIM_VARIANT"] = args.sonic_variant
        app = PlaylistApp(
            ds_mode_override=getattr(args, 'ds_mode', None),
        )
        # propagate explicit variant into generator (CLI > env > config)
        app.generator.sonic_variant = resolve_sonic_variant(
            explicit_variant=getattr(args, "sonic_variant", None),
            config_variant=getattr(app.generator, "sonic_variant", None),
        )
        dynamic_flag = getattr(args, "ds_mode", None) == "dynamic"
        if args.dry_run:
            print("🔍 DRY RUN MODE - No playlists will be created\n")
        if args.artist:
            # Single artist mode
            app.run_single_artist(
                args.artist,
                args.tracks,
                track_title=getattr(args, "track", None),
                dry_run=getattr(args, 'dry_run', False),
                dynamic=dynamic_flag,
                verbose=getattr(args, 'verbose', False),
                artist_only=getattr(args, 'artist_only', False),
            )
        else:
            # Normal mode - generate multiple playlists from history
            app.run(dry_run=getattr(args, 'dry_run', False), dynamic=dynamic_flag)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease check your config.yaml file.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
