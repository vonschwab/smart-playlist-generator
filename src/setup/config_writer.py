# src/setup/config_writer.py
"""Write a user's config.yaml by patching the shipped commented template.

Comment-preserving (ruamel round-trip) so the user gets a complete, tunable,
documented config — not opaque dumped YAML. Atomic + never-clobber.
"""
from __future__ import annotations

import os
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _THIS_REPO_ROOT / "config.example.yaml"


class ConfigExistsError(Exception):
    """Raised when config.yaml exists and reconfigure was not requested."""


def _yaml():
    from ruamel.yaml import YAML  # lazy — only the [web] extra ships ruamel
    y = YAML()
    y.preserve_quotes = True
    return y


def write_config(home, draft: dict, *, reconfigure: bool = False) -> str:
    target = Path(home.config_path)
    if target.exists() and not reconfigure:
        raise ConfigExistsError(str(target))

    y = _yaml()
    data = y.load(_TEMPLATE.read_text(encoding="utf-8"))

    lib = data.setdefault("library", {})
    lib["music_directory"] = draft["music_directory"]

    if draft.get("lastfm"):
        lf = data.setdefault("lastfm", {})
        lf["api_key"] = draft["lastfm"].get("api_key", "")
        lf["username"] = draft["lastfm"].get("username", "")
    if draft.get("discogs"):
        data.setdefault("discogs", {})["token"] = draft["discogs"].get("token", "")
    if draft.get("plex"):
        plex = data.setdefault("plex", {})
        for k, v in draft["plex"].items():
            plex[k] = v
    if draft.get("ai_genre_provider"):
        data.setdefault("ai_genre", {})["provider"] = draft["ai_genre_provider"]

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        y.dump(data, fh)
    os.replace(tmp, target)
    return str(target)
