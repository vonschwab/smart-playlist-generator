import tempfile
from pathlib import Path

from src.metadata_client import MetadataClient


def _client(tmp: Path) -> MetadataClient:
    return MetadataClient(str(tmp / "metadata.db"))


def test_fetch_scope_blacklists_round_trip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        with _client(tmp) as mc:
            # Seed a couple of tracks so artist/album scope has something to flag.
            mc.add_track("t1", "Pink Moon", "Nick Drake", "Pink Moon")
            mc.add_track("t2", "Road", "Nick Drake", "Pink Moon")
            mc.add_track("t3", "Yellow", "Coldplay", "Parachutes")

            mc.set_artist_blacklisted("Coldplay", True)
            mc.set_album_blacklisted("Nick Drake", "Pink Moon", True)
            mc.set_blacklisted(["t1"], True)  # also individually blacklist a track

            artists = mc.fetch_artist_blacklist()
            albums = mc.fetch_album_blacklist()
            tracks = mc.fetch_track_blacklist()

        assert any(a["artist_name"] == "Coldplay" for a in artists)
        assert any(al["album_name"] == "Pink Moon" and al["artist_name"] == "Nick Drake" for al in albums)
        assert any(t["track_id"] == "t1" for t in tracks)
