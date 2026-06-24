#!/usr/bin/env python3
"""Import-light entry point for running sonic feature analysis in its own process.

Why this module exists
----------------------
On Windows, ``multiprocessing`` uses the ``spawn`` start method, so every
process-pool child re-imports the ``__main__`` module during
``multiprocessing.spawn.prepare()`` *before* it can run any task. When
``__main__`` imports numpy (and the rest of the heavy stack) at module scope --
as the GUI worker (``src/playlist_gui/worker.py``) and ``analyze_library`` do --
all N children load numpy's C extension simultaneously and can deadlock on the
Windows loader lock. This was root-caused 2026-06-23 (the Analyze Library sonic
stage hung forever; see ``project_analyze_pool_deadlock`` memory).

Running the sonic pool from THIS module makes the pool's ``__main__`` numpy-free:
its module scope imports only the standard library, so the spawn-prepare()
re-import is trivial. The heavy ``SonicFeaturePipeline`` import happens inside
``main()`` -- in the parent process only, never during a child's spawn bootstrap.

Keep this module's top level free of numpy / native / heavy imports.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sonic feature analysis (import-light entry; safe for spawn)."
    )
    parser.add_argument("--db-path", default=None, help="Path to metadata.db (default: project default)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Max tracks to analyze")
    parser.add_argument("--force", action="store_true", help="Re-analyze ALL tracks")
    parser.add_argument("--rescan-inconsistent", action="store_true", help="Re-analyze inconsistent dims")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-file", default="logs/sonic_analysis.log")
    args = parser.parse_args(argv)

    # Heavy imports happen HERE (parent process only), never at module scope.
    from src.logging_utils import configure_logging
    from scripts.update_sonic import SonicFeaturePipeline

    # Log to stdout so a parent process can stream/forward progress live.
    configure_logging(level="DEBUG" if args.verbose else "INFO", log_file=args.log_file)

    pipeline = SonicFeaturePipeline(
        db_path=args.db_path,
        use_beat_sync=False,
        use_beat3tower=True,
    )
    try:
        pipeline.run(
            limit=args.limit,
            workers=args.workers,
            force=args.force,
            rescan_inconsistent=args.rescan_inconsistent,
            progress=True,
            progress_interval=args.progress_interval,
            progress_every=args.progress_every,
            verbose_each=args.verbose,
        )
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
