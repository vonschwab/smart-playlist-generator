# Anchor-based artist playlists

- `--artist` runs now lock all sampled seed tracks as anchors, spacing them using positions `round(i * (N-1)/(S-1))` and ordering anchors by best transition quality.
- Bridges between anchors are built with a beam search over the DS candidate pool; seeds are excluded from bridges and remain locked during repair.
- Logs include seed ids/titles, chosen seed order + score, anchor target/actual positions, and per-bridge stats (beam width, candidates, fallback).
- Set `PLAYLIST_DIAG_ANCHOR=1` to dump bridge paths and edge scores for debugging; existing `PLAYLIST_DIAG_RECENCY` logs recency adjustments.
