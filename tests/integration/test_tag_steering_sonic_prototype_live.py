"""Live: the sonic prototype lifts Eno's ambient-pier affinity above genre-dense alone."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path  # noqa: E402

from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.artist_style import ArtistStyleConfig, cluster_artist_tracks  # noqa: E402
from src.playlist.tag_steering import (  # noqa: E402
    resolve_tag_steering_target,
    resolve_tag_sonic_prototype_rows,
    sonic_prototype_from_rows,
    sonic_global_mean,
)

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB = "data/metadata.db"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    resolved_artifact_path()
    return load_artifact_bundle(str(ARTIFACT))


def _sonic_tag_affinity(bundle, tags):
    # NOTE: ArtifactBundle exposes the active sonic space as `X_sonic` (resolved
    # from the artifact's `X_sonic_muq` key via config.yaml's
    # `artifacts.sonic_variant_override: muq` at load time) -- there is no
    # `X_sonic_muq` attribute on the bundle itself. Using `X_sonic` here is what
    # keeps this helper space-consistent with `cluster_artist_tracks`, which
    # clusters on the same `bundle.X_sonic` matrix.
    xmq = np.asarray(bundle.X_sonic, dtype=np.float64)
    t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        tags, metadata_db_path=DB, track_id_to_row=t2r,
        exclude_artist="Brian Eno", min_support=25)
    assert rows is not None, "ambient must have enough library support"
    gm = sonic_global_mean(xmq)
    proto, cohesion, _ = sonic_prototype_from_rows(xmq, rows, global_mean=gm)
    xmn = xmq / (np.linalg.norm(xmq, axis=1, keepdims=True) + 1e-12)
    return (xmn - gm) @ proto


def test_sonic_prototype_raises_ambient_pier_affinity(bundle):
    if getattr(bundle, "genre_emb", None) is None or getattr(bundle, "X_sonic", None) is None:
        pytest.skip("dense genre or MuQ sidecar absent")
    tags = ["ambient", "drone", "dark ambient", "space ambient"]
    target, mapped, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    if target is None:
        pytest.skip("ambient tags did not map")
    aff_vec = _sonic_tag_affinity(bundle, tags)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)

    _, med_base, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB)
    _, med_sonic, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=aff_vec, sonic_tag_weight=0.5)

    base_sonic = float(np.mean([aff_vec[m] for m in med_base])) if med_base else 0.0
    with_sonic = float(np.mean([aff_vec[m] for m in med_sonic])) if med_sonic else 0.0
    assert with_sonic > base_sonic, (
        f"sonic-term piers ambient affinity {with_sonic:.3f} !> genre-only {base_sonic:.3f}")


def test_no_sonic_affinity_is_byte_identical_piers(bundle):
    tags = ["ambient"]
    target, _, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)
    _, med_a, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB)
    _, med_b, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=None, sonic_tag_weight=0.5)
    assert med_a == med_b
