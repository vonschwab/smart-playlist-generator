# scripts/growth_candidate_report.py
#!/usr/bin/env python3
"""Read-only report of graph-growth candidates over the live sidecar. No writes."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ai_genre_enrichment.graph_growth import collapse_variants, gather_growth_candidates
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    p.add_argument("--min-album-freq", type=int, default=3)
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args(argv)

    db_path = Path(args.sidecar_db)
    if not db_path.exists():
        print(f"sidecar DB not found: {db_path}", file=sys.stderr)
        return 2

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    cands = collapse_variants(
        gather_growth_candidates(store, taxonomy, min_album_freq=args.min_album_freq))
    print(f"{len(cands)} growth candidate(s) at min_album_freq={args.min_album_freq}")
    for c in cands[: args.top]:
        variants = f" (variants: {', '.join(c.variants)})" if c.variants else ""
        print(f"  {c.album_frequency:4d}  {c.term}{variants}  "
              f"~ {', '.join(c.cooccurring_tags[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
