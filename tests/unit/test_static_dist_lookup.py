"""Static dir resolution: repo web/dist preferred, packaged static_dist fallback."""
from pathlib import Path


def test_resolve_static_dir_prefers_repo(tmp_path, monkeypatch):
    import src.playlist_web.app as appmod

    repo_dist = tmp_path / "web" / "dist"
    repo_dist.mkdir(parents=True)
    monkeypatch.setattr(appmod, "ROOT", tmp_path)
    assert appmod.resolve_static_dir() == repo_dist


def test_resolve_static_dir_falls_back_to_packaged(tmp_path, monkeypatch):
    import src.playlist_web.app as appmod

    monkeypatch.setattr(appmod, "ROOT", tmp_path)  # no web/dist here
    # Inject the packaged path explicitly rather than monkeypatching
    # Path.exists globally (which would affect every other Path in the
    # process during this test) -- resolve_static_dir() accepts an
    # optional `packaged` override for exactly this reason.
    packaged = tmp_path / "packaged_static_dist"
    assert appmod.resolve_static_dir(packaged=packaged) == packaged


def test_resolve_static_dir_default_packaged_is_sibling_of_module(tmp_path, monkeypatch):
    import src.playlist_web.app as appmod

    monkeypatch.setattr(appmod, "ROOT", tmp_path)  # no web/dist here
    expected = Path(appmod.__file__).resolve().parent / "static_dist"
    assert appmod.resolve_static_dir() == expected
