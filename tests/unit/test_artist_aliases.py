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


def test_validate_artist_link_groups_flags_bad_input():
    from src.playlist.artist_aliases import validate_artist_link_groups
    errs = validate_artist_link_groups([
        {"type": "alias", "members": ["Solo"]},          # <2 members
        {"type": "bogus", "members": ["A", "B"]},         # bad type
        {"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]},
        {"type": "sibling", "members": ["Alex G", "X"]},  # Alex G reused across groups
    ])
    assert len(errs) == 3
    assert validate_artist_link_groups([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}]) == []


def test_save_and_read_round_trip(tmp_path):
    from src.playlist.artist_aliases import save_artist_link_groups, read_artist_link_groups
    p = tmp_path / "artist_aliases.yaml"
    groups = [{"type": "sibling", "members": ["Smog", "Bill Callahan"]}]
    save_artist_link_groups(groups, path=p)
    assert read_artist_link_groups(path=p) == groups
    assert read_artist_link_groups(path=tmp_path / "missing.yaml") == []


def test_save_rejects_invalid_and_backs_up(tmp_path):
    import pytest
    from src.playlist.artist_aliases import save_artist_link_groups, read_artist_link_groups
    p = tmp_path / "artist_aliases.yaml"
    save_artist_link_groups([{"type": "alias", "members": ["A", "B"]}], path=p)
    with pytest.raises(ValueError):
        save_artist_link_groups([{"type": "alias", "members": ["OnlyOne"]}], path=p)
    # original file unchanged; a .bak.* backup exists from the second-write attempt? No:
    # invalid input raises BEFORE any write, so no backup and the good file is intact.
    assert read_artist_link_groups(path=p) == [{"type": "alias", "members": ["A", "B"]}]
    # a valid overwrite creates a timestamped backup of the prior file
    save_artist_link_groups([{"type": "sibling", "members": ["X", "Y"]}], path=p)
    assert list(p.parent.glob("artist_aliases.yaml.bak.*"))


def test_nonempty_ondisk_file_loads_without_recursion(tmp_path, monkeypatch):
    """Regression: loading a NON-EMPTY data/artist_aliases.yaml through the real
    _cached_load path must resolve correctly and NOT infinitely recurse
    (build_artist_link_map -> normalize_primary_artist_key -> resolve_alias ->
    get_active_map -> build ...). Reproduces the production bug where a saved file
    silently fell back to "empty" (no aliasing) — every prior test missed it by using
    set_artist_link_map_for_testing or an empty on-disk file."""
    import yaml
    import src.playlist.artist_aliases as aa
    from src.playlist.identity_keys import normalize_primary_artist_key as npk

    p = tmp_path / "artist_aliases.yaml"
    p.write_text(
        yaml.safe_dump(
            {"version": 1, "groups": [
                {"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]},
                {"type": "sibling", "members": ["Smog", "Bill Callahan"]},
            ]},
            sort_keys=False, allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(aa, "_DEFAULT_ALIAS_PATH", p)
    aa.clear_cache()
    m = aa.get_active_map()
    assert m.alias_key and m.sibling_key, "non-empty file must load, not fall back to empty"
    # alias merges through the real consumer path (npk applies resolve_alias)
    assert aa.resolve_alias(npk("Alex G")) == aa.resolve_alias(npk("(Sandy) Alex G"))
    assert aa.resolve_alias(npk("Alex G")).startswith("alias_group:")
    # sibling: shared group id, NOT alias-merged
    sg = aa.sibling_group_of("Smog")
    assert sg is not None and sg == aa.sibling_group_of("Bill Callahan")
    assert aa.resolve_alias(normalize_artist_key("Smog")) == normalize_artist_key("Smog")
    aa.clear_cache()
