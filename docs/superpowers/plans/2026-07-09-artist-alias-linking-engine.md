# Artist-Alias Linking — Resolution Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runtime engine that lets a user-curated `data/artist_aliases.yaml` make the playlist generator treat linked artist names as one identity — two link types: **alias** (full merge) and **sibling** (independent artists spaced ≥ `min_gap` apart).

**Architecture:** A new `src/playlist/artist_aliases.py` loads the YAML into a process-cached lookup exposing `resolve_alias(key) -> key` and `sibling_group_of(value) -> group|None`. Alias merge is applied at a handful of existing key-computation chokepoints (`_artist_indices_in_bundle`, `normalize_primary_artist_key`, `candidate_pool._normalize_artist_key`, the Fire popularity path). Sibling repulsion is a parallel `used_sibling_groups` set inside the beam, mirroring the existing `used_artists` mechanism. When the YAML is absent/empty, every hook short-circuits to a bit-for-bit no-op.

**Tech Stack:** Python 3.11+, `yaml.safe_load`, `functools.lru_cache`, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-artist-alias-linking-design.md`. This plan covers the **engine only**; the GUI panel is a separate follow-up plan (Plan 2). The engine is fully usable by hand-editing the YAML.

## Global Constraints

- **Python 3.11+**; `ruff check` (E, F) and `mypy` must stay clean — do NOT add modules to `[[tool.mypy.overrides]]`; type the code instead.
- **Empty/absent `data/artist_aliases.yaml` MUST be a bit-for-bit no-op.** Every hook short-circuits when its part of the map is empty. The existing `tests/unit/test_pier_bridge_smoke_golden.py` goldens are the regression guard and must stay green.
- **No writes to `data/metadata.db` or any NPZ artifact.** The engine is pure runtime resolution over a small git-tracked YAML.
- **YAML loading uses `yaml.safe_load`** (repo convention). Data-file path resolved via `Path(__file__).resolve().parents[2] / "data" / "artist_aliases.yaml"` (cwd-robust: CLI, worker, satellites).
- **Tests run bounded, never piped:** `python -m pytest -q -m "not slow" <path>` with the tool timeout, never `| tail`/`| head`.
- **Shared canonical checkout — commit explicit paths only.** `git add <paths>` then `git commit --only -- <paths>`; verify with `git diff --cached --name-only` first. NEVER `git add -A`/`-u`/`.` or a bare `git commit`. Current branch: `feat/artist-alias-linking`.
- **Import-cycle rule:** `src/playlist/artist_aliases.py` must have NO top-level import of any `src.*` normalizer (it lazily imports them inside `build_artist_link_map`). This lets consumer modules import `resolve_alias`/`sibling_group_of` at their top level without a cycle.

---

## File Structure

- **Create** `src/playlist/artist_aliases.py` — the resolver: YAML load, cache, `ArtistLinkMap`, `resolve_alias`, `sibling_group_of`, `alias_group_member_names`, test seam, `clear_cache`.
- **Create** `data/artist_aliases.yaml` — empty seed file (`version: 1` / `groups: []`) so the feature is discoverable and the loader has a target.
- **Create** `tests/unit/test_artist_aliases.py` — resolver unit tests.
- **Create** `tests/unit/test_artist_aliases_integration.py` — per-chokepoint behavior tests (Tasks 2–5) + sibling repulsion (Task 6).
- **Modify** `tests/conftest.py` — autouse fixture resetting the test override between tests.
- **Modify** `src/playlist/artist_style.py:20-53` — alias-merge `_artist_indices_in_bundle`.
- **Modify** `src/playlist/identity_keys.py:13-46` — alias-merge `normalize_primary_artist_key`.
- **Modify** `src/playlist/candidate_pool.py:105-108` (+ backstop sites ~1315-1345) — alias-merge structural cap.
- **Modify** `src/analyze/popularity_runner.py:232-284` — Fire popularity multi-name merge.
- **Modify** `src/playlist/pier_bridge/beam.py` — sibling repulsion (`used_sibling_groups`).
- **Modify** `.claude/skills/playlist-testing/SKILL.md` — resolve the "Smog ≟ Bill Callahan" follow-up note.

---

## Task 1: Resolver module + unit tests

**Files:**
- Create: `src/playlist/artist_aliases.py`
- Create: `data/artist_aliases.yaml`
- Create: `tests/unit/test_artist_aliases.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces:
  - `build_artist_link_map(groups: list[dict]) -> ArtistLinkMap` — pure builder.
  - `resolve_alias(key: str) -> str` — merged group key for alias members, else `key`.
  - `sibling_group_of(value: str) -> Optional[str]` — sibling group id, else `None`.
  - `alias_group_member_names(resolved_key: str) -> list[str]` — display names of an alias group (for Fire merge), else `[]`.
  - `set_artist_link_map_for_testing(groups_or_map_or_None) -> None`, `clear_cache() -> None`.
  - `ArtistLinkMap` dataclass with `.alias_key`, `.sibling_key`, `.alias_members`, `.is_empty()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_artist_aliases.py`:

```python
from src.playlist.artist_aliases import (
    build_artist_link_map,
    resolve_alias,
    sibling_group_of,
    alias_group_member_names,
    set_artist_link_map_for_testing,
)
from src.string_utils import normalize_artist_key
from src.playlist.identity_keys import normalize_primary_artist_key


def test_alias_members_collapse_in_both_key_spaces():
    m = build_artist_link_map([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    # Both normalization families must land on the same merged key.
    a_struct = m.alias_key[normalize_artist_key("Alex G")]
    b_struct = m.alias_key[normalize_artist_key("(Sandy) Alex G")]
    a_sem = m.alias_key[normalize_primary_artist_key("Alex G")]
    b_sem = m.alias_key[normalize_primary_artist_key("(Sandy) Alex G")]
    assert a_struct == b_struct == a_sem == b_sem
    assert a_struct.startswith("alias_group:")


def test_resolve_alias_uses_active_map_and_passes_through_unknowns():
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert resolve_alias(normalize_artist_key("Alex G")) == resolve_alias(normalize_artist_key("(Sandy) Alex G"))
    assert resolve_alias("someone else") == "someone else"


def test_sibling_group_of_matches_siblings_not_aliases():
    set_artist_link_map_for_testing([{"type": "sibling", "members": ["Smog", "Bill Callahan"]}])
    g1 = sibling_group_of("Smog")                       # raw display name
    g2 = sibling_group_of(normalize_primary_artist_key("Bill Callahan"))  # normalized key
    assert g1 is not None and g1 == g2
    assert resolve_alias(normalize_artist_key("Smog")) == normalize_artist_key("Smog")  # siblings do NOT alias-merge
    assert sibling_group_of("Unrelated Artist") is None


def test_alias_group_member_names_for_fire_merge():
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    resolved = resolve_alias(normalize_artist_key("Alex G"))
    assert sorted(alias_group_member_names(resolved)) == ["(Sandy) Alex G", "Alex G"]
    assert alias_group_member_names("not-a-group") == []


def test_empty_map_is_noop():
    set_artist_link_map_for_testing(None)  # reset to on-disk default (empty in test env)
    assert resolve_alias("anything") == "anything"
    assert sibling_group_of("anything") is None


def test_validation_rejects_bad_groups():
    # <2 members, invalid type, and a member duplicated across groups are all skipped.
    m = build_artist_link_map([
        {"type": "alias", "members": ["Solo Only"]},              # too few
        {"type": "bogus", "members": ["A", "B"]},                 # bad type
        {"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]},
        {"type": "sibling", "members": ["Alex G", "Someone"]},    # Alex G already claimed -> skipped
    ])
    assert m.alias_key.get(normalize_artist_key("Alex G")) is not None
    assert m.sibling_key == {}          # the conflicting sibling group was dropped
    assert normalize_artist_key("Solo Only") not in m.alias_key
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.playlist.artist_aliases'`.

- [ ] **Step 3: Create the resolver module**

Create `src/playlist/artist_aliases.py`:

```python
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

    def is_empty(self) -> bool:
        return not self.alias_key and not self.sibling_key


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
```

- [ ] **Step 4: Create the empty seed YAML**

Create `data/artist_aliases.yaml`:

```yaml
# Manual artist links. Two link types:
#   alias   - same project, spelling/formatting variants (full identity merge).
#   sibling - one person, distinct projects (independent artists, spaced >= min_gap apart).
# Edited from the GUI "Artist Links" panel, or by hand. Empty = feature inert.
version: 1
groups: []
```

- [ ] **Step 5: Add the autouse reset fixture**

In `tests/conftest.py`, add (mirrors the existing `_reset_sonic_variant_override` fixture):

```python
@pytest.fixture(autouse=True)
def _reset_artist_link_map():
    """Keep a test's artist-link override from leaking into other tests."""
    from src.playlist.artist_aliases import set_artist_link_map_for_testing
    set_artist_link_map_for_testing(None)
    yield
    set_artist_link_map_for_testing(None)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py`
Expected: PASS (6 tests).

- [ ] **Step 7: Lint/type check**

Run: `ruff check src/playlist/artist_aliases.py && mypy src/playlist/artist_aliases.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/playlist/artist_aliases.py data/artist_aliases.yaml tests/unit/test_artist_aliases.py tests/conftest.py
git commit --only -m "feat(artist-links): resolver module + empty YAML + test seam" -- src/playlist/artist_aliases.py data/artist_aliases.yaml tests/unit/test_artist_aliases.py tests/conftest.py
```

---

## Task 2: Alias merge — seed/pier + Fire row gathering (`_artist_indices_in_bundle`)

This is the single highest-leverage chokepoint: both pier gathering (`cluster_artist_tracks`) and Fire popularity (`load_artist_popularity_values`) route through it, so seeding "Alex G" gathers "(Sandy) Alex G" tracks after this one edit.

**Files:**
- Modify: `src/playlist/artist_style.py:35-42`
- Test: `tests/unit/test_artist_aliases_integration.py`

**Interfaces:**
- Consumes: `resolve_alias` (Task 1).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_artist_aliases_integration.py`:

```python
import types
import numpy as np
from src.playlist.artist_aliases import set_artist_link_map_for_testing


def _ns_bundle(artist_keys, track_artists=None):
    return types.SimpleNamespace(
        artist_keys=np.array(artist_keys, dtype=object),
        track_artists=np.array(track_artists if track_artists is not None else artist_keys, dtype=object),
    )


def test_artist_indices_gathers_alias_members():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0, 1]
    assert _artist_indices_in_bundle(b, "(Sandy) Alex G") == [0, 1]


def test_artist_indices_unlinked_unchanged():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing(None)  # empty
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_artist_indices_gathers_alias_members`
Expected: FAIL — currently returns `[0]` (aliases don't match).

- [ ] **Step 3: Edit `_artist_indices_in_bundle`**

In `src/playlist/artist_style.py`, the current body is:

```python
    if bundle.artist_keys is None:
        return []
    artist_key = normalize_artist_key(artist_name)
    raw_artists = getattr(bundle, "track_artists", None)
    indices: List[int] = []
    for i, ak in enumerate(bundle.artist_keys):
        if normalize_artist_key(str(ak)) == artist_key:
```

Change the two key computations to wrap `resolve_alias(...)`:

```python
    if bundle.artist_keys is None:
        return []
    from src.playlist.artist_aliases import resolve_alias
    artist_key = resolve_alias(normalize_artist_key(artist_name))
    raw_artists = getattr(bundle, "track_artists", None)
    indices: List[int] = []
    for i, ak in enumerate(bundle.artist_keys):
        if resolve_alias(normalize_artist_key(str(ak))) == artist_key:
```

(Leave the `include_collaborations` branch below unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py`
Expected: PASS (both tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/playlist/artist_style.py
git add src/playlist/artist_style.py tests/unit/test_artist_aliases_integration.py
git commit --only -m "feat(artist-links): alias-merge seed/pier + Fire row gathering" -- src/playlist/artist_style.py tests/unit/test_artist_aliases_integration.py
```

---

## Task 3: Alias merge — semantic identity chokepoint (`normalize_primary_artist_key`)

This covers the whole pier-bridge/beam semantic layer: beam diversity (`artist_key_by_idx`), seed-artist interior exclusion (`segment_pool_builder.py:471/480`), the production pool-collapse (`_ak` → `identity_keys_for_index`), and (artist,title) dedup (`seeds.py:188`).

**Files:**
- Modify: `src/playlist/identity_keys.py:13-46`
- Test: `tests/unit/test_artist_aliases_integration.py` (append)

**Interfaces:**
- Consumes: `resolve_alias` (Task 1).
- Produces: `normalize_primary_artist_key` now returns the merged key for alias members.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_artist_aliases_integration.py`)

```python
def test_normalize_primary_artist_key_merges_aliases():
    from src.playlist.identity_keys import normalize_primary_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert normalize_primary_artist_key("Alex G") == normalize_primary_artist_key("(Sandy) Alex G")


def test_identity_keys_for_index_merges_aliases():
    from src.playlist.identity_keys import identity_keys_for_index
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = types.SimpleNamespace(
        track_ids=np.array(["t0", "t1"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        track_titles=np.array(["S0", "S1"], dtype=object),
    )
    assert identity_keys_for_index(b, 0).artist_key == identity_keys_for_index(b, 1).artist_key
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_normalize_primary_artist_key_merges_aliases`
Expected: FAIL (keys differ).

- [ ] **Step 3: Edit `normalize_primary_artist_key`**

In `src/playlist/identity_keys.py`, add the top-level import (cycle-safe — `artist_aliases` has no top-level `src` imports) and wrap the return. The current tail of the function is:

```python
    return normalize_artist_name(
        text,
        strip_ensemble=True,
        strip_collaborations=True,
        lowercase=True,
        normalize_unicode=True,
    )
```

Add near the other imports at the top of the file:

```python
from src.playlist.artist_aliases import resolve_alias
```

And change the return to:

```python
    key = normalize_artist_name(
        text,
        strip_ensemble=True,
        strip_collaborations=True,
        lowercase=True,
        normalize_unicode=True,
    )
    return resolve_alias(key)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py`
Expected: PASS.

- [ ] **Step 5: Guard against the import cycle**

Run: `python -c "import src.playlist.identity_keys; import src.playlist.artist_aliases; from src.playlist.artist_aliases import build_artist_link_map; build_artist_link_map([{'type':'alias','members':['A','B']}])"`
Expected: no `ImportError`/`RecursionError`.

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/playlist/identity_keys.py
git add src/playlist/identity_keys.py tests/unit/test_artist_aliases_integration.py
git commit --only -m "feat(artist-links): alias-merge semantic identity chokepoint" -- src/playlist/identity_keys.py tests/unit/test_artist_aliases_integration.py
```

---

## Task 4: Alias merge — candidate-pool structural cap (`_normalize_artist_key` + backstop)

**Files:**
- Modify: `src/playlist/candidate_pool.py:105-108` and the two min-pool backstop sites (~1315-1345)
- Test: `tests/unit/test_artist_aliases_integration.py` (append)

**Interfaces:**
- Consumes: `resolve_alias` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_candidate_pool_normalize_key_merges_aliases():
    from src.playlist.candidate_pool import _normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert _normalize_artist_key("Alex G") == _normalize_artist_key("(Sandy) Alex G")
    set_artist_link_map_for_testing(None)
    assert _normalize_artist_key("Alex G") != _normalize_artist_key("(Sandy) Alex G")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_candidate_pool_normalize_key_merges_aliases`
Expected: FAIL.

- [ ] **Step 3: Edit `_normalize_artist_key`**

In `src/playlist/candidate_pool.py`, the current helper is:

```python
def _normalize_artist_key(raw: Any) -> str:
    from src.string_utils import normalize_artist_key

    return normalize_artist_key("" if raw is None else str(raw))
```

Change to:

```python
def _normalize_artist_key(raw: Any) -> str:
    from src.string_utils import normalize_artist_key
    from src.playlist.artist_aliases import resolve_alias

    return resolve_alias(normalize_artist_key("" if raw is None else str(raw)))
```

- [ ] **Step 4: Cover the min-pool backstop sites**

Read `src/playlist/candidate_pool.py:1315-1345`. Two sites in the min-pool-size backstop bypass the helper by using raw `str(artist_keys[i])` (flagged in the design):
- a `Counter(str(artist_keys[i]) for i in pool_indices)` (~line 1321),
- an `_ak = str(artist_keys[i])` (~line 1339).

Change each `str(artist_keys[i])` (and `str(artist_keys[seed_idx])` if present in that block) to `_normalize_artist_key(artist_keys[i])` so the backstop groups by the alias-merged key too. Verify with:

Run: `python -c "import re,io; s=open('src/playlist/candidate_pool.py',encoding='utf-8').read(); print(s.count('str(artist_keys['))"`
Expected: `0` remaining raw `str(artist_keys[` occurrences in that function (confirm by reading — leave any that are genuinely outside `build_candidate_pool` untouched, but there should be none of the backstop form left).

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_candidate_pool_normalize_key_merges_aliases`
Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/playlist/candidate_pool.py
git add src/playlist/candidate_pool.py tests/unit/test_artist_aliases_integration.py
git commit --only -m "feat(artist-links): alias-merge candidate-pool per-artist cap + backstop" -- src/playlist/candidate_pool.py tests/unit/test_artist_aliases_integration.py
```

---

## Task 5: Alias merge — Fire popularity multi-name merge (`load_artist_popularity_values`)

`_artist_indices_in_bundle` (Task 2) already makes the *local* track set span both alias names, but the Last.fm top-tracks cache is keyed per literal name — so a Fire seed on "Alex G" would ignore "(Sandy) Alex G"'s popularity row. Merge the top-tracks across all alias member names before resolving.

**Files:**
- Modify: `src/analyze/popularity_runner.py:249-275`
- Test: `tests/unit/test_artist_aliases_integration.py` (append)

**Interfaces:**
- Consumes: `resolve_alias`, `alias_group_member_names` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_fire_popularity_merges_alias_catalogs(tmp_path):
    from src.analyze.popularity_runner import init_top_tracks_cache, upsert_artist_top_tracks, load_artist_popularity_values
    from unittest.mock import MagicMock
    from src.string_utils import normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])

    b = types.SimpleNamespace(
        track_ids=np.array(["early", "late", "other"], dtype=object),
        track_titles=np.array(["Early Song", "Late Song", "Nope"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        durations_ms=None,
    )
    db = str(tmp_path / "pop.db")
    init_top_tracks_cache(db)
    # Each name's catalog has its own Last.fm hit, cached under its own key.
    upsert_artist_top_tracks(db, normalize_artist_key("Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Early Song", "mbid": "", "rank": 0}])
    upsert_artist_top_tracks(db, normalize_artist_key("(Sandy) Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Late Song", "mbid": "", "rank": 0}])
    client = MagicMock()  # fresh cache -> no network
    vec = load_artist_popularity_values(
        b, "Alex G", client=client, db_path=db, limit=50, max_age_days=30,
        now_iso="2026-06-24T00:00:00+00:00")
    assert vec is not None and vec.shape == (3,)
    assert vec[0] == 1.0 and vec[1] == 1.0   # BOTH catalogs' hits landed
    assert np.isnan(vec[2])
    client.get_artist_top_tracks.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_fire_popularity_merges_alias_catalogs`
Expected: FAIL — only `vec[0]` scored (only "Alex G"'s cache row consulted).

- [ ] **Step 3: Edit `load_artist_popularity_values`**

In `src/analyze/popularity_runner.py`, the current keying + fetch is:

```python
    artist_key = normalize_artist_key(artist_name)
    top = get_artist_top_tracks_cached_or_fetch(
        artist_key, artist_name, client=client, db_path=db_path,
        limit=limit, max_age_days=max_age_days, now_iso=now_iso)
    pop = resolve_top_tracks_to_popularity(top, local_tracks)
```

Replace with a loop over the alias group's member names (falls back to the single seed name when unlinked), merging the top-tracks lists:

```python
    from src.playlist.artist_aliases import resolve_alias, alias_group_member_names
    resolved = resolve_alias(normalize_artist_key(artist_name))
    names = alias_group_member_names(resolved) or [artist_name]
    top: list = []
    for nm in names:
        top.extend(get_artist_top_tracks_cached_or_fetch(
            normalize_artist_key(nm), nm, client=client, db_path=db_path,
            limit=limit, max_age_days=max_age_days, now_iso=now_iso))
    pop = resolve_top_tracks_to_popularity(top, local_tracks)
```

(The `local_tracks` set above already spans both names via `_artist_indices_in_bundle` from Task 2.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_fire_popularity_merges_alias_catalogs`
Also run the existing suite to confirm no regression: `python -m pytest -q tests/unit/test_popularity_lazy.py`
Expected: PASS (both).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/analyze/popularity_runner.py
git add src/analyze/popularity_runner.py tests/unit/test_artist_aliases_integration.py
git commit --only -m "feat(artist-links): Fire popularity merges alias member catalogs" -- src/analyze/popularity_runner.py tests/unit/test_artist_aliases_integration.py
```

---

## Task 6: Sibling repulsion in the beam

Siblings stay independent artists everywhere (own budget, own catalog) but may not be placed within `min_gap` of each other. Implement as a parallel `used_sibling_groups` set on `BeamState`, mirroring the existing whole-segment `used_artists` approximation (segments are ≤ `min_gap` long) plus the cross-segment boundary already carried in `recent_global_artists`. When there are no sibling groups, `sibling_group_of` returns `None` and every added branch is a no-op (bit-identical).

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py`
- Test: `tests/unit/test_artist_aliases_integration.py` (append)

**Interfaces:**
- Consumes: `sibling_group_of` (Task 1); `build_pier_bridge_playlist` (existing).

- [ ] **Step 1: Write the failing test** (append)

```python
def _artist_positions(result_track_ids, bundle, artist_name):
    idx = {str(t): i for i, t in enumerate(bundle.track_ids)}
    tset = {str(bundle.track_ids[i]) for i in range(len(bundle.track_ids))
            if str(bundle.track_artists[i]) == artist_name}
    return [pos for pos, tid in enumerate(result_track_ids) if str(tid) in tset]


def test_siblings_never_within_min_gap():
    import numpy as np
    from pathlib import Path
    from src.features.artifacts import ArtifactBundle
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    rng = np.random.default_rng(3)
    n = 40
    # Interior candidates are all Smog/Bill Callahan (siblings); piers are two other artists.
    artists = []
    for i in range(n):
        if i in (0, 1):
            artists.append("Pier A" if i == 0 else "Pier B")
        else:
            artists.append("Smog" if i % 2 == 0 else "Bill Callahan")
    bundle = ArtifactBundle(
        artifact_path=Path("sib_test"),
        track_ids=np.array([f"t{i}" for i in range(n)]),
        artist_keys=np.array(artists, dtype=object),
        track_artists=np.array(artists, dtype=object),
        track_titles=np.array([f"Song {i}" for i in range(n)]),
        X_sonic=rng.standard_normal((n, 16)),
        X_sonic_start=None, X_sonic_mid=None, X_sonic_end=None,
        X_genre_raw=(rng.random((n, 8)) > 0.7).astype(float),
        X_genre_smoothed=np.clip(rng.random((n, 8)), 0.0, 1.0),
        genre_vocab=np.array([f"g{i}" for i in range(8)]),
        track_id_to_index={f"t{i}": i for i in range(n)},
        durations_ms=np.full(n, 200_000, dtype=np.int64),
    )
    min_gap = 3
    set_artist_link_map_for_testing([{"type": "sibling", "members": ["Smog", "Bill Callahan"]}])
    cfg = PierBridgeConfig(bridge_floor=0.0, transition_floor=0.0, center_transitions=False,
                           variable_bridge_length=False, edge_delete_enabled=False)
    result = build_pier_bridge_playlist(
        seed_track_ids=["t0", "t1"], total_tracks=16, bundle=bundle,
        candidate_pool_indices=[i for i in range(n) if i not in (0, 1)],
        cfg=cfg, min_gap=min_gap, min_genre_similarity=None, X_genre_smoothed=None,
    )
    smog = _artist_positions(result.track_ids, bundle, "Smog")
    bill = _artist_positions(result.track_ids, bundle, "Bill Callahan")
    # No Smog position is within min_gap of any Bill Callahan position.
    for s in smog:
        for b in bill:
            assert abs(s - b) >= min_gap, f"sibling within min_gap: Smog@{s}, Bill@{b} -> {result.track_ids}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_siblings_never_within_min_gap`
Expected: FAIL — siblings currently land adjacent (no repulsion yet).

- [ ] **Step 3: Import + BeamState field**

In `src/playlist/pier_bridge/beam.py`, add the top-level import near the existing `resolve_artist_identity_keys` import (lines 19-22):

```python
from src.playlist.artist_aliases import sibling_group_of
```

Add a field to the `BeamState` dataclass (after `used_artists`):

```python
    used_sibling_groups: Set[str] = field(default_factory=set)
```

- [ ] **Step 4: Sibling-group closure + init seeding**

Just above the main beam loop, next to the existing `_cand_identity_keys` closure (~line 1016), add a memoized sibling-group resolver:

```python
    _sibling_cache: Dict[int, Optional[str]] = {}

    def _idx_sibling_group(idx_int: int) -> Optional[str]:
        if idx_int in _sibling_cache:
            return _sibling_cache[idx_int]
        s = ""
        if bundle is not None and getattr(bundle, "track_artists", None) is not None:
            try:
                s = str(bundle.track_artists[int(idx_int)] or "")
            except Exception:
                s = ""
        if not s and artist_key_by_idx is not None:
            s = str(artist_key_by_idx.get(int(idx_int), "") or "")
        sg = sibling_group_of(s) if s else None
        _sibling_cache[idx_int] = sg
        return sg
```

At the init block (after `used_artists_init` is fully built, ~line 944, before `initial_state = BeamState(...)`), seed the sibling set from the cross-segment boundary and the piers:

```python
    used_sibling_groups_init: Set[str] = set()
    if recent_global_artists:
        for ak in recent_global_artists:
            sg = sibling_group_of(str(ak)) if ak else None
            if sg:
                used_sibling_groups_init.add(sg)
    for pier_idx in (pier_a, pier_b):
        sg = _idx_sibling_group(int(pier_idx))
        if sg:
            used_sibling_groups_init.add(sg)
```

Pass it into the initial state:

```python
    initial_state = BeamState(
        path=[pier_a],
        score=0.0,
        used={pier_a, pier_b},
        used_artists=used_artists_init,
        used_sibling_groups=used_sibling_groups_init,
        last_progress=0.0,
    )
```

Note: `_idx_sibling_group` is defined below this point in the current file order; move its `def` (and `_sibling_cache`) to just above the init block, OR seed the piers with an inline `sibling_group_of` call mirroring the pier-artist-string logic. Keep the closure definition before its first use.

- [ ] **Step 5: Admission gate**

Immediately after the existing artist-diversity gate that ends at line 1237 (`if cand_artist and cand_artist in state.used_artists: continue`), and before the `bridge_floor` check at line 1239, add:

```python
                # Sibling repulsion: a sibling of an already-placed (in-window) artist
                # may not be placed adjacent. No-op when there are no sibling links.
                _cand_sg = _idx_sibling_group(int(cand))
                if _cand_sg is not None and _cand_sg in state.used_sibling_groups:
                    continue
```

- [ ] **Step 6: Carry-forward (both scoring branches)**

In the main-path state build (after `new_used_artists` is finalized, ~line 1464) AND in the tie-break-path build (~line 1609), add:

```python
                    new_used_sibling_groups = state.used_sibling_groups
                    _place_sg = _idx_sibling_group(int(cand))
                    if _place_sg is not None:
                        new_used_sibling_groups = state.used_sibling_groups | {_place_sg}
```

Then thread `used_sibling_groups=new_used_sibling_groups` into the two `BeamState(...)` constructors that append to `next_beam` (~lines 1506-1513 and 1651-1658).

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest -q tests/unit/test_artist_aliases_integration.py::test_siblings_never_within_min_gap`
Expected: PASS.

- [ ] **Step 8: Run the beam golden guard (bit-identical when empty)**

Run: `python -m pytest -q tests/unit/test_pier_bridge_smoke_golden.py`
Expected: PASS (goldens unchanged — the sibling additions are no-ops with no links, and the autouse fixture keeps the empty map).

- [ ] **Step 9: Lint + commit**

```bash
ruff check src/playlist/pier_bridge/beam.py
git add src/playlist/pier_bridge/beam.py tests/unit/test_artist_aliases_integration.py
git commit --only -m "feat(artist-links): sibling repulsion in the beam (min_gap spacing)" -- src/playlist/pier_bridge/beam.py tests/unit/test_artist_aliases_integration.py
```

---

## Task 7: No-op regression, full-suite verification, and skill update

**Files:**
- Modify: `.claude/skills/playlist-testing/SKILL.md`
- Test: `tests/unit/test_artist_aliases_integration.py` (append)

- [ ] **Step 1: Add an explicit empty-map no-op test** (append)

```python
def test_empty_map_leaves_smoke_generation_identical():
    """With no links, build_pier_bridge_playlist output is unchanged vs the code path
    that never consulted the resolver (guards the bit-identical guarantee directly)."""
    import numpy as np
    from pathlib import Path
    from src.features.artifacts import ArtifactBundle
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    def _mk():
        rng = np.random.default_rng(7)
        n, num_artists = 50, 10
        return ArtifactBundle(
            artifact_path=Path("noop_test"),
            track_ids=np.array([f"t{i}" for i in range(n)]),
            artist_keys=np.array([f"a{i % num_artists}" for i in range(n)], dtype=object),
            track_artists=np.array([f"Artist {i % num_artists}" for i in range(n)], dtype=object),
            track_titles=np.array([f"Song {i}" for i in range(n)]),
            X_sonic=rng.standard_normal((n, 16)),
            X_sonic_start=None, X_sonic_mid=None, X_sonic_end=None,
            X_genre_raw=(rng.random((n, 8)) > 0.7).astype(float),
            X_genre_smoothed=np.clip(rng.random((n, 8)), 0.0, 1.0),
            genre_vocab=np.array([f"g{i}" for i in range(8)]),
            track_id_to_index={f"t{i}": i for i in range(n)},
            durations_ms=np.full(n, 200_000, dtype=np.int64),
        )

    cfg = dict(bridge_floor=0.0, transition_floor=0.0, center_transitions=False,
               variable_bridge_length=False, edge_delete_enabled=False)
    kw = dict(seed_track_ids=["t0", "t10"], total_tracks=12,
              candidate_pool_indices=[i for i in range(50) if i not in (0, 10)],
              min_gap=3, min_genre_similarity=None, X_genre_smoothed=None)

    set_artist_link_map_for_testing(None)  # empty
    r1 = build_pier_bridge_playlist(bundle=_mk(), cfg=PierBridgeConfig(**cfg), **kw)
    # An unrelated alias/sibling link (no member present in this bundle) must not change output.
    set_artist_link_map_for_testing([{"type": "sibling", "members": ["Nobody Here", "Also Absent"]}])
    r2 = build_pier_bridge_playlist(bundle=_mk(), cfg=PierBridgeConfig(**cfg), **kw)
    assert list(r1.track_ids) == list(r2.track_ids)
```

- [ ] **Step 2: Run the alias + adjacent suites**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py tests/unit/test_artist_aliases_integration.py tests/unit/test_pier_bridge_smoke_golden.py tests/unit/test_popularity_lazy.py`
Expected: PASS (all).

- [ ] **Step 3: Run the fast suite for regressions**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS. Quote the real pass/fail counts from the output you actually saw (do not claim green on a subset).

- [ ] **Step 4: Update the playlist-testing skill**

In `.claude/skills/playlist-testing/SKILL.md`, under "Known follow-ups", the line:

```
- **Smog ≟ Bill Callahan**: identity resolution does not collapse same-person/different-project names; a true test needs that capability first.
```

Replace with:

```
- **Artist links (alias / sibling)**: `data/artist_aliases.yaml` + `src/playlist/artist_aliases.py` now provide manual identity linking. Inject in tests with `set_artist_link_map_for_testing([...])` (autouse-reset in `tests/conftest.py`). Alias = full merge; sibling = independent artists spaced ≥ `min_gap` (beam `used_sibling_groups`). Design: `docs/superpowers/specs/2026-07-09-artist-alias-linking-design.md`.
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_artist_aliases_integration.py .claude/skills/playlist-testing/SKILL.md
git commit --only -m "test(artist-links): empty-map no-op regression + update testing skill" -- tests/unit/test_artist_aliases_integration.py .claude/skills/playlist-testing/SKILL.md
```

- [ ] **Step 6: Verify end-to-end with a real generation (per the `verify` discipline)**

Hand-add a temporary sibling link for two artists that both exist in the library to `data/artist_aliases.yaml`, then run one real generation through the GUI-fidelity harness at INFO and confirm from the log + output that the two never land within `min_gap`. Revert the temporary edit afterward (leave `groups: []`). This exercises the live path, not just unit fixtures. If the live artifact is unavailable in this workspace, state that explicitly instead of claiming verification.

---

## Self-Review (completed during authoring)

- **Spec coverage:** alias full-merge → Tasks 2 (seed/pier/Fire rows), 3 (semantic: beam diversity, dedup, seed-exclusion, pool-collapse), 4 (structural cap), 5 (Fire cache merge). Sibling repulsion → Task 6. Storage/loader/validation/cache-bust seam → Task 1. No-op guarantee → Task 6 Step 8 + Task 7 Step 1. GUI is explicitly out of scope (Plan 2).
- **Cache-bust (`clear_cache`)** is provided in Task 1 for Plan 2's worker to call; no consumer in this plan needs it (fresh processes reload).
- **Type consistency:** `resolve_alias(str)->str`, `sibling_group_of(str)->Optional[str]`, `alias_group_member_names(str)->list[str]`, `build_artist_link_map(list)->ArtistLinkMap`, `set_artist_link_map_for_testing(None|ArtistLinkMap|list)` — used consistently across Tasks 2–6.
- **Known deferrals:** nested alias-within-sibling is out of scope (single-group validation in Task 1). The min-pool backstop (Task 4 Step 4) requires reading the exact lines before editing — called out explicitly, not left vague.
