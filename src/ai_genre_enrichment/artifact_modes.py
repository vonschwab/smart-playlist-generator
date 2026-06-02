"""Explicit artifact genre-source modes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class GenreArtifactSource(str, Enum):
    LEGACY = "legacy"
    ENRICHED = "enriched"
    HYBRID_SHADOW = "hybrid_shadow"

    @classmethod
    def resolve(cls, value: str | None) -> "GenreArtifactSource":
        return cls(value or cls.LEGACY.value)


def make_resolver(mode: GenreArtifactSource, sidecar_db: str | Path) -> Any | None:
    if mode is GenreArtifactSource.LEGACY:
        return None
    from .genre_resolver import EnrichedGenreResolver
    return EnrichedGenreResolver(sidecar_db)


@dataclass(frozen=True)
class ShadowOutputPaths:
    root: Path
    sparse_artifact: Path
    dense_sidecar: Path
    report: Path


def signature_snapshot_identity(sidecar_db: str | Path) -> str:
    resolved = Path(sidecar_db).resolve()
    if not resolved.exists():
        return "missing-sidecar"
    uri = resolved.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("BEGIN")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        signatures = []
        if "enriched_genre_signatures" in tables:
            signatures = conn.execute(
                """
                SELECT release_key, signature_json,
                       COALESCE(enrichment_policy_version, 'legacy-v0')
                FROM enriched_genre_signatures
                ORDER BY release_key
                """
            ).fetchall()
        overrides = []
        if "ai_genre_user_overrides" in tables:
            overrides = conn.execute(
                """
                SELECT release_key, normalized_artist, normalized_album,
                       genres_add_json, genres_remove_json
                FROM ai_genre_user_overrides
                ORDER BY release_key
                """
            ).fetchall()
    return hashlib.sha256(
        json.dumps(
            {"signatures": signatures, "overrides": overrides},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_identity(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_metadata_identity(path: str | Path) -> str:
    """Hash a consistent logical SQLite snapshot, including committed WAL rows."""
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        schema_rows = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name, sql
            """
        ).fetchall()
        _update_json_digest(digest, schema_rows)
        for object_type, name, _table_name, _sql in schema_rows:
            if object_type != "table":
                continue
            _update_json_digest(digest, name)
            quoted_name = '"' + name.replace('"', '""') + '"'
            rows = [
                _json_stable_row(row)
                for row in conn.execute(f"SELECT * FROM {quoted_name}")
            ]
            _update_json_digest(digest, sorted(rows))
    return digest.hexdigest()


def temporary_shadow_artifact_path(target: str | Path) -> Path:
    target = Path(target)
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}")


def publish_shadow_artifact(
    temporary_path: str | Path,
    target_path: str | Path,
    *,
    overwrite: bool,
) -> None:
    temporary_path = Path(temporary_path)
    target_path = Path(target_path)
    try:
        if overwrite:
            os.replace(temporary_path, target_path)
        else:
            os.link(temporary_path, target_path)
            temporary_path.unlink()
    except FileExistsError as exc:
        raise ValueError(
            f"hybrid_shadow output already exists at {target_path}; "
            "pass --overwrite-shadow to replace it"
        ) from exc


def _json_stable_row(row: tuple[Any, ...]) -> str:
    return json.dumps(
        [_json_stable_value(value) for value in row],
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_stable_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


def _update_json_digest(digest: Any, value: Any) -> None:
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")


def shadow_output_paths(
    *,
    artifacts_dir: str | Path,
    policy_version: str,
    signature_snapshot: str,
    prior_snapshot: str,
    sparse_input_identity: str,
    metadata_identity: str,
    config_identity: str,
    genre_sim_identity: str,
    dense_config: dict[str, Any],
) -> ShadowOutputPaths:
    payload = {
        "genre_source": GenreArtifactSource.HYBRID_SHADOW.value,
        "policy_version": policy_version,
        "signature_snapshot": signature_snapshot,
        "prior_snapshot": prior_snapshot,
        "sparse_input_identity": sparse_input_identity,
        "metadata_identity": metadata_identity,
        "config_identity": config_identity,
        "genre_sim_identity": genre_sim_identity,
        "dense_config": dense_config,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    root = Path(artifacts_dir) / "shadow" / fingerprint
    return ShadowOutputPaths(
        root=root,
        sparse_artifact=root / "data_matrices_step1.npz",
        dense_sidecar=root / "data_matrices_step1_genre_emb_dim64.npz",
        report=root / "comparison_report.json",
    )


def shadow_input_identities(
    *,
    sidecar_db: str | Path,
    active_sparse_artifact: str | Path,
    metadata_db: str | Path,
    config_path: str | Path,
    genre_sim_path: str | Path | None,
    policy_version: str,
    prior_snapshot: str,
    dense_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy_version": policy_version,
        "signature_snapshot": signature_snapshot_identity(sidecar_db),
        "prior_snapshot": prior_snapshot,
        "sparse_input_identity": file_identity(active_sparse_artifact),
        "metadata_identity": sqlite_metadata_identity(metadata_db),
        "config_identity": file_identity(config_path),
        "genre_sim_identity": file_identity(genre_sim_path) if genre_sim_path else "none",
        "dense_config": dict(dense_config),
    }
