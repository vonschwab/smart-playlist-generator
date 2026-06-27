import numpy as np
from src.title_dedupe import calculate_version_preference_score
from src.playlist.artist_style import _dedupe_artist_indices


def test_version_preference_penalizes_live_album():
    # Clean track title, but the ALBUM is a live recording -> penalized.
    studio = calculate_version_preference_score("Polly", "Nevermind")
    unplugged = calculate_version_preference_score("Polly", "MTV Unplugged in New York")
    reading = calculate_version_preference_score("Lithium", "Live at Reading")
    assert studio > unplugged
    assert studio > reading
    # Title-only call (no album) is unchanged / backwards-compatible.
    assert calculate_version_preference_score("Polly") == 100
    # A standalone "live" word IS penalized so live LPs named "Live <X>" (e.g. Unwound's
    # "Live Leaves") are caught. This also penalizes the rare studio LP named "Live <X>"
    # (e.g. "Live Through This") — an accepted tradeoff: the -30 only changes a tie-break
    # between two versions of the SAME song, which those studio LPs' tracks almost never have.
    assert calculate_version_preference_score("Doll Parts", "Live Through This") == 70
    # ...but a substring like "Alive"/"Olive" is NOT a false positive (word boundary).
    assert calculate_version_preference_score("Doll Parts", "Still Alive") == 100


def test_dedupe_album_aware_beats_duration_tiebreak():
    # Two "Polly": idx 0 = MTV Unplugged (LIVE, and LONGER as live takes often are),
    # idx 1 = Nevermind (studio, shorter). Clean titles, so score ties on title alone.
    titles = ["Polly", "Polly"]
    durations = np.array([240000, 196000], dtype=float)  # idx0 live longer, idx1 studio shorter
    albums = {0: "MTV Unplugged in New York", 1: "Nevermind"}
    # Without album info: score ties -> duration tiebreak keeps the LONGER (live) one — the bug.
    assert _dedupe_artist_indices([0, 1], titles, durations, None) == [0]
    # With album info: the studio version wins despite being shorter — the fix.
    assert _dedupe_artist_indices([0, 1], titles, durations, albums) == [1]
