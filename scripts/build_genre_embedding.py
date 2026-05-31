#!/usr/bin/env python3
"""
Build Genre Embedding Sidecar (Phase 3)
========================================

Reads an existing artifact NPZ, trains a PMI-SVD genre embedding from its
X_genre_raw matrix, projects every track into the dense genre space, and
saves a sidecar NPZ that load_artifact_bundle() picks up automatically.

The sidecar name is: <artifact_stem>_genre_emb_dim<DIM>.npz

Usage:
    # Quick test — dim=32, no LLM prior
    python scripts/build_genre_embedding.py --skip-prior --dim 32

    # Production default — dim=64, Anthropic prior
    python scripts/build_genre_embedding.py --provider anthropic

    # Use OpenAI for prior
    python scripts/build_genre_embedding.py --provider openai

The load_artifact_bundle() function looks for the default sidecar (dim=64) at
load time.  To change which dim is loaded, set artifact.genre_emb_dim in
config.yaml (future config key, not yet wired).

Exit codes: 0 success, 1 error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.genre.blend import blend_with_prior
from src.genre.llm_client import make_llm_client
from src.genre.llm_prior import build_llm_prior
from src.genre.pmi_svd import project_tracks, train_pmi_svd
from src.logging_utils import configure_logging
from src.playlist.genre_idf import compute_genre_idf

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
CACHE_DIR = ROOT / "data" / "genre_llm_cache"


def sidecar_path(artifact_path: Path, dim: int) -> Path:
    return artifact_path.parent / f"{artifact_path.stem}_genre_emb_dim{dim}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PMI-SVD genre embedding sidecar")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "dry-run"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Mock LLM — no API calls")
    parser.add_argument("--skip-prior", action="store_true", help="Skip LLM prior step")
    parser.add_argument("--alpha-at-zero", type=float, default=0.9)
    parser.add_argument("--half-life", type=float, default=25.0)
    parser.add_argument(
        "--remove-top-components", type=int, default=2,
        help="All-but-the-top: drop this many leading PCs to remove PPMI anisotropy "
             "(0 = legacy behavior). Default 2.",
    )
    parser.add_argument(
        "--no-idf-projection", action="store_true",
        help="Disable IDF-weighting of the track projection (legacy behavior). "
             "By default hub genres are IDF-down-weighted so a single common tag "
             "(e.g. 'rock') does not dominate a track's dense position.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    if not args.artifact.exists():
        logger.error("Artifact not found: %s", args.artifact)
        return 1

    out_path = sidecar_path(args.artifact, args.dim)

    # --- Load artifact ---
    logger.info("Loading artifact: %s", args.artifact)
    data = np.load(args.artifact, allow_pickle=True)
    X_genre_raw: np.ndarray = data["X_genre_raw"]
    genre_vocab: list[str] = [str(g) for g in data["genre_vocab"]]
    track_ids: np.ndarray = data["track_ids"]
    N, V = X_genre_raw.shape
    logger.info("  %d tracks × %d genres", N, V)

    if V < args.dim:
        logger.error("vocab size %d < dim %d; reduce --dim", V, args.dim)
        return 1

    # --- Train PMI-SVD ---
    logger.info(
        "Training PMI-SVD (dim=%d, smoothing=%.2f, remove_top_components=%d) ...",
        args.dim, args.smoothing, args.remove_top_components,
    )
    corpus_emb = train_pmi_svd(
        X_genre_raw,
        dim=args.dim,
        smoothing=args.smoothing,
        random_state=args.random_state,
        remove_top_components=args.remove_top_components,
    )  # (V, dim)

    # --- LLM prior ---
    support = (X_genre_raw > 0).sum(axis=0).astype(np.int32)

    if args.skip_prior:
        genre_emb = corpus_emb
    else:
        use_dry_run = args.dry_run or args.provider == "dry-run"
        llm_client = make_llm_client(
            provider="dry-run" if use_dry_run else args.provider,
            model=args.model,
            dry_run=use_dry_run,
        )
        prior_cache = CACHE_DIR / f"llm_prior_{llm_client.provider}__{llm_client.model}_dim{args.dim}.json"
        logger.info("Building LLM prior (provider=%s) ...", llm_client.provider)
        prior_emb = build_llm_prior(
            genre_vocab, corpus_emb, support, llm_client,
            prior_cache, n_anchors=min(args.dim, V),
        )
        logger.info("Blending (alpha_at_zero=%.2f, half_life=%.0f) ...", args.alpha_at_zero, args.half_life)
        genre_emb = blend_with_prior(
            corpus_emb, prior_emb, support,
            alpha_at_zero=args.alpha_at_zero,
            half_life=args.half_life,
        )

    # --- Project tracks ---
    idf = None
    if not args.no_idf_projection:
        idf = compute_genre_idf(X_genre_raw=X_genre_raw, power=1.0, norm="max1")
        logger.info("IDF-weighting track projection (hub genres down-weighted)")
    logger.info("Projecting %d tracks to %d-dim dense genre space ...", N, args.dim)
    X_genre_dense = project_tracks(X_genre_raw, genre_emb, idf=idf)  # (N, dim)

    coverage = int((X_genre_raw > 0).any(axis=1).sum())
    logger.info("  Coverage: %d / %d tracks have at least one genre", coverage, N)

    # --- Save sidecar ---
    emb_config = {
        "dim": args.dim,
        "smoothing": args.smoothing,
        "random_state": args.random_state,
        "skip_prior": args.skip_prior,
        "provider": "dry-run" if (args.dry_run or args.provider == "dry-run") else args.provider,
        "alpha_at_zero": args.alpha_at_zero if not args.skip_prior else None,
        "half_life": args.half_life if not args.skip_prior else None,
        "remove_top_components": args.remove_top_components,
        "idf_projection": not args.no_idf_projection,
        "artifact": str(args.artifact),
        "vocab_size": V,
        "n_tracks": N,
    }

    logger.info("Saving sidecar → %s", out_path)
    np.savez(
        out_path,
        X_genre_dense=X_genre_dense,     # (N, dim) track embeddings
        genre_emb=genre_emb,             # (V, dim) genre vocabulary embedding
        genre_vocab=np.array(genre_vocab, dtype=object),  # (V,) for alignment check
        track_ids=track_ids,             # (N,) for alignment check
        emb_config=emb_config,
    )
    logger.info("Done. Sidecar: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
