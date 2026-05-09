"""DS playlist pipeline.

Public API:
    DSPipelineResult — typed result from a pipeline run.
    generate_playlist_ds — orchestrate end-to-end DS playlist generation.
    enforce_allowed_invariant — guard helper used by post-order validation.

The implementation is split across this package:
    core.py — orchestrator (generate_playlist_ds + DSPipelineResult).

Tier-1.5 splits the orchestrator further into focused modules
(bundle_restrict, embedding_setup, pier_resolver, pier_bridge_overrides,
audit_emitter, post_validation). Each lands in its own PR.
"""
from .core import (
    DSPipelineResult,
    enforce_allowed_invariant,
    generate_playlist_ds,
)

__all__ = [
    "DSPipelineResult",
    "enforce_allowed_invariant",
    "generate_playlist_ds",
]
