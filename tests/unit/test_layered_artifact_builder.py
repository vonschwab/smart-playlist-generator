import numpy as np
import json
import sqlite3
from argparse import Namespace


def test_build_layered_genre_matrices_from_sidecar_assignments(tmp_path):
    from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.layered_vectors import build_layered_genre_matrices
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)

    materialize_layered_assignments(
        store,
        release_id="the clientele::strange geometry",
        artist="the clientele",
        album="strange geometry",
        report=HybridGenreReport(
            release_key="the clientele::strange geometry",
            accepted_genres=[
                FusedGenreDecision(
                    term="jangle pop",
                    confidence=0.91,
                    basis="local_metadata+lastfm_tags+taxonomy",
                    sources=["local_metadata", "lastfm_tags"],
                    reason="Corroborated specific style.",
                ),
                FusedGenreDecision(
                    term="lo-fi",
                    confidence=0.86,
                    basis="bandcamp_release+taxonomy",
                    sources=["bandcamp_release"],
                    reason="Descriptor.",
                ),
            ],
            provisional_genres=[],
            rejected_noise=[],
            needs_review=[],
        ),
        taxonomy=taxonomy,
    )
    materialize_layered_assignments(
        store,
        release_id="duster::stratosphere",
        artist="duster",
        album="stratosphere",
        report=HybridGenreReport(
            release_key="duster::stratosphere",
            accepted_genres=[
                FusedGenreDecision(
                    term="slowcore",
                    confidence=0.93,
                    basis="bandcamp_release+taxonomy",
                    sources=["bandcamp_release"],
                    reason="Strong source.",
                ),
            ],
            provisional_genres=[],
            rejected_noise=[],
            needs_review=[],
        ),
        taxonomy=taxonomy,
    )

    matrices = build_layered_genre_matrices(
        store,
        track_release_keys=[
            "the clientele::strange geometry",
            "duster::stratosphere",
            "missing::album",
        ],
        taxonomy=taxonomy,
    )

    assert matrices.X_genre_leaf_idf.shape == (3, 2)
    assert matrices.X_genre_family.shape == (3, 3)
    assert matrices.X_facet.shape == (3, 1)
    assert matrices.X_genre_bridge.shape == matrices.X_genre_leaf_idf.shape

    assert matrices.genre_leaf_vocab == ("jangle pop", "slowcore")
    assert matrices.genre_family_vocab == ("indie/alternative", "pop", "rock")
    assert matrices.genre_bridge_vocab == matrices.genre_leaf_vocab
    assert matrices.facet_vocab == ("lo-fi",)
    assert matrices.taxonomy_version == "0.2.0-expanded"
    assert len(matrices.graph_fingerprint) == 64

    jangle_idx = matrices.genre_leaf_vocab.index("jangle pop")
    slowcore_idx = matrices.genre_leaf_vocab.index("slowcore")
    rock_idx = matrices.genre_family_vocab.index("rock")
    lofi_idx = matrices.facet_vocab.index("lo-fi")

    assert matrices.X_genre_leaf_idf[0, jangle_idx] > 0
    assert matrices.X_genre_leaf_idf[1, slowcore_idx] > 0
    assert matrices.X_genre_leaf_idf[2].sum() == 0
    assert matrices.X_genre_family[0, rock_idx] > 0
    assert matrices.X_genre_family[1, rock_idx] > 0
    assert matrices.X_facet[0, lofi_idx] == np.float32(0.86)
    assert matrices.X_facet[1].sum() == 0


def test_build_ds_artifacts_can_emit_layered_vectors_when_requested(tmp_path):
    from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.analyze.artifact_builder import build_ds_artifacts

    metadata = tmp_path / "metadata.db"
    with sqlite3.connect(metadata) as conn:
        conn.executescript(
            """
            CREATE TABLE tracks (
                track_id TEXT PRIMARY KEY,
                artist TEXT,
                title TEXT,
                album TEXT,
                album_id TEXT,
                duration_ms INTEGER,
                sonic_features TEXT
            );
            CREATE TABLE track_genres (track_id TEXT, genre TEXT);
            CREATE TABLE album_genres (album_id TEXT, genre TEXT);
            CREATE TABLE artist_genres (artist TEXT, genre TEXT);
            """
        )
        conn.execute(
            """
            INSERT INTO tracks(
                track_id, artist, title, album, album_id, duration_ms, sonic_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "t1",
                "The Clientele",
                "Since K Got Over Me",
                "Strange Geometry",
                "a1",
                240000,
                json.dumps(
                    {
                        "mfcc_mean": [1.0, 2.0],
                        "chroma_mean": [0.5, 0.25],
                        "bpm": 90.0,
                        "spectral_centroid": 1200.0,
                    }
                ),
            ),
        )

    config = tmp_path / "config.yaml"
    config.write_text(
        f"library:\n  database_path: {metadata.as_posix()}\nplaylists: {{}}\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)
    materialize_layered_assignments(
        store,
        release_id="the clientele::strange geometry",
        artist="the clientele",
        album="strange geometry",
        report=HybridGenreReport(
            release_key="the clientele::strange geometry",
            accepted_genres=[
                FusedGenreDecision(
                    term="jangle pop",
                    confidence=0.91,
                    basis="local_metadata+lastfm_tags+taxonomy",
                    sources=["local_metadata", "lastfm_tags"],
                    reason="Corroborated specific style.",
                ),
                FusedGenreDecision(
                    term="lo-fi",
                    confidence=0.86,
                    basis="bandcamp_release+taxonomy",
                    sources=["bandcamp_release"],
                    reason="Descriptor.",
                ),
            ],
            provisional_genres=[],
            rejected_noise=[],
            needs_review=[],
        ),
        taxonomy=taxonomy,
    )

    out = tmp_path / "artifact.npz"
    result = build_ds_artifacts(
        db_path=str(metadata),
        config_path=str(config),
        out_path=out,
        read_only_metadata=True,
        emit_layered_vectors=True,
        layered_sidecar_db=sidecar,
    )

    artifact = np.load(out, allow_pickle=True)
    assert result.n_tracks == 1
    assert "X_genre_raw" in artifact.files
    assert "X_genre_leaf_idf" in artifact.files
    assert "X_genre_family" in artifact.files
    assert "X_genre_bridge" in artifact.files
    assert "X_facet" in artifact.files
    assert artifact["X_genre_bridge"].shape == artifact["X_genre_leaf_idf"].shape
    assert artifact["genre_leaf_vocab"].tolist() == ["jangle pop"]
    assert artifact["genre_bridge_vocab"].tolist() == ["jangle pop"]
    assert artifact["genre_family_vocab"].tolist() == ["indie/alternative", "pop", "rock"]
    assert artifact["facet_vocab"].tolist() == ["lo-fi"]
    assert artifact["genre_graph_taxonomy_version"].item() == "0.2.0-expanded"
    assert len(str(artifact["genre_graph_sidecar_fingerprint"].item())) == 64


def test_rebuild_artifacts_layered_shadow_plumbs_layered_vectors(monkeypatch, tmp_path):
    from scripts import ai_genre_enrich

    metadata = tmp_path / "metadata.db"
    with sqlite3.connect(metadata) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO tracks VALUES ('t1')")
    config = tmp_path / "config.yaml"
    config.write_text("library: {}\nplaylists: {}\n", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    active = artifacts_dir / "data_matrices_step1.npz"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"active-artifact")
    sidecar = tmp_path / "sidecar.db"

    calls = []

    def fake_build_ds_artifacts(**kwargs):
        calls.append(kwargs)
        kwargs["out_path"].write_bytes(b"layered-shadow")
        return type("Result", (), {"n_tracks": 1, "n_genres": 0})()

    monkeypatch.setattr(
        "src.analyze.artifact_builder.build_ds_artifacts",
        fake_build_ds_artifacts,
    )

    rc = ai_genre_enrich.cmd_rebuild_artifacts(Namespace(
        metadata_db=metadata,
        sidecar_db=sidecar,
        artifacts_dir=str(artifacts_dir),
        config=str(config),
        genre_sim_path=None,
        genre_source="layered_shadow",
        overwrite_shadow=False,
    ))

    shadow_artifacts = list((artifacts_dir / "shadow").glob("*/data_matrices_step1.npz"))
    assert rc == 0
    assert len(shadow_artifacts) == 1
    assert shadow_artifacts[0].read_bytes() == b"layered-shadow"
    assert calls[0]["emit_layered_vectors"] is True
    assert calls[0]["layered_sidecar_db"] == sidecar
    assert calls[0]["read_only_metadata"] is True
