from scripts.extract_energy_sidecar import _win_to_wsl


def test_win_to_wsl_drive_letter():
    assert _win_to_wsl(r"E:\MUSIC\a.flac") == "/mnt/e/MUSIC/a.flac"


def test_win_to_wsl_already_posix():
    assert _win_to_wsl("/mnt/e/x.flac") == "/mnt/e/x.flac"
