"""Flow-tagged gold corpus + pair-set construction for the pace eval."""
from __future__ import annotations

import os
import random
import re
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class AlbumSpec:
    key: str
    artist_like: str
    album_like: str
    flow_type: str          # tight_continuous | gradient_flow | flat_uniform_mix
    register: str           # high | mid | low
    path_contains: str | None = None
    usable_first: int | None = None
    usable_last: int | None = None
    expected: int | None = None


# Register-balanced, flow_type-tagged. LIKE patterns + expected counts are
# verified against the live DB in Task 5 (adjust patterns/path_contains there).
CORPUS: list[AlbumSpec] = [
    AlbumSpec("renaissance", "%Beyonc%", "%RENAISSANCE%", "flat_uniform_mix", "high", path_contains="Explicit", expected=16),
    AlbumSpec("discovery", "%Daft Punk%", "%Discovery%", "gradient_flow", "high", expected=14),
    # Avalanches "Since I Left You (Deluxe)" dropped from Pass 1: deluxe-edition
    # filenames use disc-track format "1-NN." which the leading-int parser reads as
    # the disc number, and there's no "Disc 1" path substring. Needs disc-aware
    # parsing (a per-spec `disc` field) — revisit in Pass 2 if more high-register
    # tight_continuous data is wanted.
    AlbumSpec("lcd_tih", "%LCD Soundsystem%", "%This Is Happening%", "gradient_flow", "high", usable_first=1, usable_last=9, expected=9),
    AlbumSpec("caribou_swim", "%Caribou%", "%Swim%", "gradient_flow", "high", usable_first=1, usable_last=9, expected=9),
    AlbumSpec("donuts", "%Dilla%", "%Donuts%", "tight_continuous", "mid", expected=31),
    AlbumSpec("voodoo", "%D'Angelo%", "%Voodoo%", "gradient_flow", "mid", expected=13),
    AlbumSpec("dub_roots", "%King Tubby%", "%Dub From the Roots%", "tight_continuous", "mid", expected=16),
    AlbumSpec("boc_mhtrtc", "%Boards of Canada%", "%Right To Children%", "gradient_flow", "mid", expected=18),
    AlbumSpec("beach_bloom", "%Beach House%", "%Bloom%", "gradient_flow", "mid", expected=10),
    AlbumSpec("eno_onland", "%Brian Eno%", "%On Land%", "tight_continuous", "low", expected=8),
    AlbumSpec("hiroshi_green", "%Yoshimura%", "%Green%", "tight_continuous", "low", expected=8),
    AlbumSpec("sotl_aspera", "%Stars of the Lid%", "%Per Aspera%", "tight_continuous", "low", expected=6),
]

_TRACKNO = re.compile(r"^\s*(\d+)")


@dataclass(frozen=True)
class CorpusTrack:
    track_id: str
    album_key: str
    track_no: int
    flow_type: str
    register: str


def _track_no(file_path: str | None) -> int | None:
    base = os.path.basename(file_path or "")
    match = _TRACKNO.match(base)
    return int(match.group(1)) if match else None


def resolve_corpus(conn, specs: list[AlbumSpec] = CORPUS):
    out: list[CorpusTrack] = []
    counts: dict[str, int] = {}
    for spec in specs:
        rows = conn.execute(
            "SELECT track_id, file_path FROM tracks "
            "WHERE artist LIKE ? AND album LIKE ? AND file_path IS NOT NULL",
            (spec.artist_like, spec.album_like),
        ).fetchall()
        items: list[tuple[int, str]] = []
        for tid, fp in rows:
            if spec.path_contains and spec.path_contains.lower() not in (fp or "").lower():
                continue
            tn = _track_no(fp)
            if tn is None:
                continue
            if spec.usable_first is not None and tn < spec.usable_first:
                continue
            if spec.usable_last is not None and tn > spec.usable_last:
                continue
            items.append((tn, str(tid)))
        items.sort()
        seen: set[int] = set()
        ordered: list[tuple[int, str]] = []
        for tn, tid in items:
            if tn in seen:
                continue  # dedup duplicate rips at same track number
            seen.add(tn)
            ordered.append((tn, tid))
        counts[spec.key] = len(ordered)
        if spec.expected is not None and counts[spec.key] != spec.expected:
            warnings.warn(f"{spec.key}: expected {spec.expected} tracks, resolved {counts[spec.key]}")
        for tn, tid in ordered:
            out.append(CorpusTrack(tid, spec.key, tn, spec.flow_type, spec.register))
    return out, counts


def build_pairs(tracks: list[CorpusTrack], *, seed: int = 13, n_random: int = 2000):
    by_album: dict[str, list[CorpusTrack]] = {}
    for t in tracks:
        by_album.setdefault(t.album_key, []).append(t)
    for k in by_album:
        by_album[k].sort(key=lambda t: t.track_no)

    adjacent: list[tuple[str, str]] = []
    adjacent_gradient: list[tuple[str, str]] = []
    non_adjacent: list[tuple[str, str]] = []
    for ts in by_album.values():
        ids = [t.track_id for t in ts]
        for i in range(len(ids) - 1):
            adjacent.append((ids[i], ids[i + 1]))
            if ts[0].flow_type == "gradient_flow":
                adjacent_gradient.append((ids[i], ids[i + 1]))
        if ts and ts[0].flow_type == "gradient_flow":
            for i in range(len(ids)):
                for j in range(i + 2, len(ids)):
                    non_adjacent.append((ids[i], ids[j]))

    register_of = {t.track_id: t.register for t in tracks}
    all_ids = [t.track_id for t in tracks]
    rng = random.Random(seed)
    random_cross: list[tuple[str, str]] = []
    tries = 0
    seen_pairs: set[tuple[str, str]] = set()
    while len(random_cross) < n_random and tries < n_random * 40:
        tries += 1
        a, b = rng.choice(all_ids), rng.choice(all_ids)
        if a == b or register_of[a] == register_of[b]:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        random_cross.append((a, b))
    if len(random_cross) < n_random:
        warnings.warn(f"random_cross under-sampled: {len(random_cross)}/{n_random} pairs")
    return {"adjacent": adjacent, "adjacent_gradient": adjacent_gradient,
            "non_adjacent_same_album": non_adjacent, "random_cross": random_cross}


def write_corpus_tsv(path: str, tracks: list[CorpusTrack]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("track_id\talbum_key\ttrack_no\tflow_type\tregister\n")
        for t in tracks:
            f.write(f"{t.track_id}\t{t.album_key}\t{t.track_no}\t{t.flow_type}\t{t.register}\n")
