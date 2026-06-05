from src.playlist.run_audit import RunAuditContext, RunAuditEvent, write_markdown_report


def _context() -> RunAuditContext:
    return RunAuditContext(
        timestamp_utc="2026-06-04T12:00:00Z",
        run_id="test-run",
        cohesion_mode="dynamic",
        seed_track_id="seed",
        seed_artist="Seed Artist",
        dry_run=True,
        artifact_path="artifact.npz",
        sonic_variant="tower_weighted",
        allowed_ids_count=0,
    )


def test_run_audit_renders_layered_candidate_diagnostics(tmp_path):
    path = tmp_path / "audit.md"
    events = [
        RunAuditEvent(
            kind="preflight",
            ts_utc="2026-06-04T12:00:01Z",
            payload={
                "tuning": {},
                "tuning_sources": {},
                "pool_summary": {
                    "candidate_pool_stats": {
                        "pool_size": 2,
                        "layered_genre_admission": {
                            "source": "layered",
                            "applied": True,
                            "input_eligible_count": 3,
                            "admitted_count": 2,
                            "rejected_count": 1,
                            "rejection_reason_counts": {
                                "broad_only_without_leaf_support": 1,
                            },
                        },
                        "layered_genre_shadow": {
                            "enabled": True,
                            "evaluated_count": 3,
                            "legacy_disagreement_count": 1,
                            "samples": [
                                {
                                    "track_id": "broad-id",
                                    "reason": "broad_only_without_leaf_support",
                                    "family_affinity": 1.0,
                                    "niche_similarity": 0.0,
                                }
                            ],
                        },
                    }
                },
            },
        )
    ]

    write_markdown_report(context=_context(), events=events, path=path, max_bytes=100_000)

    text = path.read_text(encoding="utf-8")
    assert "### layered_genre_admission" in text
    assert "broad_only_without_leaf_support" in text
    assert "### layered_genre_shadow" in text
    assert "`broad-id`" in text


def test_run_audit_renders_layered_transition_diagnostics(tmp_path):
    path = tmp_path / "audit.md"
    events = [
        RunAuditEvent(
            kind="final_success",
            ts_utc="2026-06-04T12:00:01Z",
            payload={
                "playlist_tracks": [],
                "weakest_edges": [],
                "summary_stats": {
                    "final_playlist_size": 3,
                    "layered_transition_diagnostics": {
                        "enabled": True,
                        "edge_count": 2,
                        "explained_count": 1,
                        "unexplained_count": 1,
                        "reason_counts": {
                            "bridge_supported": 1,
                            "unexplained_family_jump": 1,
                        },
                        "samples": [
                            {
                                "edge_index": 0,
                                "from_track_id": "t0",
                                "to_track_id": "t1",
                                "reason": "bridge_supported",
                                "score": 0.75,
                            }
                        ],
                    },
                },
            },
        )
    ]

    write_markdown_report(context=_context(), events=events, path=path, max_bytes=100_000)

    text = path.read_text(encoding="utf-8")
    assert "### layered_transition_diagnostics" in text
    assert "bridge_supported" in text
    assert "unexplained_family_jump" in text
    assert "`t0`" in text
    assert "`t1`" in text
