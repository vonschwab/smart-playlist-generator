"""User-curated artist link resolution (aliases + sibling projects).

Loaded from data/artist_aliases.yaml. Two link types:
  - alias   : same project, spelling/formatting variants -> full identity merge.
  - sibling : one person, distinct projects -> independent identities, spaced
              >= min_gap apart (enforced in the beam).

Import-cycle rule: this module lazily imports the normalizers inside
build_artist_link_map so consumer modules (identity_keys, beam, ...) can import
resolve_alias/sibling_group_of at their top level without a cycle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_ALIAS_PATH = Path(__file__).resolve().parents[2] / "data" / "artist_aliases.yaml"
VALID_TYPES = ("alias", "sibling")


@dataclass(frozen=True)
class ArtistLinkMap:
    # normalized member key (BOTH normalization families) -> merged alias group key
    alias_key: Dict[str, str] = field(default_factory=dict)
    # normalized member key (BOTH families) -> sibling group id
    sibling_key: Dict[str, str] = field(default_factory=dict)
    # merged alias group key -> member display names (for Fire multi-name popularity merge)
    alias_members: Dict[str, List[str]] = field(default_factory=dict)


_EMPTY = ArtistLinkMap()


def _normalizers():
    # Lazy: see import-cycle rule in the module docstring.
    from src.string_utils import normalize_artist_key
    from src.playlist.identity_keys import normalize_primary_artist_key
    return normalize_artist_key, normalize_primary_artist_key


def build_artist_link_map(groups: List[dict]) -> ArtistLinkMap:
    """Build the lookup from a list of {type, members} dicts.

    Each member is registered under BOTH the structural key
    (normalize_artist_key) and the semantic key (normalize_primary_artist_key),
    so a lookup resolves regardless of which normalization the caller used.
    A member may belong to at most one group (first group wins; a later group
    that reuses a member is skipped). Malformed groups are warned and skipped.
    """
    norm_struct, norm_sem = _normalizers()
    alias_key: Dict[str, str] = {}
    sibling_key: Dict[str, str] = {}
    alias_members: Dict[str, List[str]] = {}
    owner: Dict[str, int] = {}  # normalized form -> first group index that claimed it

    for gi, group in enumerate(groups or []):
        if not isinstance(group, dict):
            logger.warning("artist_aliases: group %d is not a mapping; skipped", gi)
            continue
        gtype = str(group.get("type", "")).strip().lower()
        raw_members = group.get("members")
        names = [str(m).strip() for m in raw_members] if isinstance(raw_members, list) else []
        names = [n for n in names if n]
        if gtype not in VALID_TYPES:
            logger.warning("artist_aliases: group %d has invalid type %r; skipped", gi, gtype)
            continue
        if len(names) < 2:
            logger.warning("artist_aliases: group %d (%s) needs >=2 members; skipped", gi, gtype)
            continue

        forms: List[str] = []
        for n in names:
            for form in {norm_struct(n), norm_sem(n)}:
                if form:
                    forms.append(form)
        conflict = next((f for f in forms if owner.get(f, gi) != gi), None)
        if conflict is not None:
            logger.warning(
                "artist_aliases: group %d (%s) reuses an already-linked artist; skipped", gi, gtype
            )
            continue

        group_id = f"{gtype}_group:" + "|".join(sorted(set(forms)))
        target = alias_key if gtype == "alias" else sibling_key
        for f in forms:
            owner[f] = gi
            target[f] = group_id
        if gtype == "alias":
            alias_members[group_id] = list(dict.fromkeys(names))  # dedupe, preserve order

    return ArtistLinkMap(alias_key=alias_key, sibling_key=sibling_key, alias_members=alias_members)


@lru_cache(maxsize=1)
def _cached_load(path_str: str) -> ArtistLinkMap:
    path = Path(path_str)
    if not path.exists():
        return _EMPTY
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # never let a malformed file break generation
        logger.warning("artist_aliases: failed to load %s: %s; treating as empty", path, exc)
        return _EMPTY
    groups = data.get("groups") if isinstance(data, dict) else None
    return build_artist_link_map(groups or [])


_test_override: Optional[ArtistLinkMap] = None


def get_active_map() -> ArtistLinkMap:
    if _test_override is not None:
        return _test_override
    return _cached_load(str(_DEFAULT_ALIAS_PATH))


def set_artist_link_map_for_testing(groups_or_map: Union[None, ArtistLinkMap, List[dict]]) -> None:
    """Override the active map in tests. None resets to the on-disk default."""
    global _test_override
    if groups_or_map is None:
        _test_override = None
    elif isinstance(groups_or_map, ArtistLinkMap):
        _test_override = groups_or_map
    else:
        _test_override = build_artist_link_map(groups_or_map)


def clear_cache() -> None:
    """Bust the cached on-disk map (call after the YAML changes; e.g. GUI edit)."""
    _cached_load.cache_clear()


def resolve_alias(key: str) -> str:
    m = get_active_map()
    if not m.alias_key:
        return key
    return m.alias_key.get(key, key)


def sibling_group_of(value: str) -> Optional[str]:
    m = get_active_map()
    if not m.sibling_key:
        return None
    hit = m.sibling_key.get(value)
    if hit is not None:
        return hit
    norm_struct, norm_sem = _normalizers()
    return m.sibling_key.get(norm_sem(value)) or m.sibling_key.get(norm_struct(value))


def alias_group_member_names(resolved_key: str) -> List[str]:
    return list(get_active_map().alias_members.get(resolved_key, []))
