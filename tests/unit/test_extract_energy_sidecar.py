from scripts.extract_energy_sidecar import _parse_args, _win_to_wsl


def test_win_to_wsl_drive_letter():
    assert _win_to_wsl(r"E:\MUSIC\a.flac") == "/mnt/e/MUSIC/a.flac"


def test_win_to_wsl_already_posix():
    assert _win_to_wsl("/mnt/e/x.flac") == "/mnt/e/x.flac"


def test_parse_args_force_flag():
    assert _parse_args(["--force"]).force is True
    assert _parse_args([]).force is False


def test_parse_args_defaults():
    a = _parse_args([])
    assert a.workers == 14 and a.limit == 0 and a.merge_only is False
