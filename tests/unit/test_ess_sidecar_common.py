import json
import numpy as np
from scripts.ess_sidecar_common import (
    win_to_wsl_path,
    read_checkpoint_ids,
    append_checkpoint,
    merge_sidecar_npz,
)


def test_win_to_wsl_path():
    assert win_to_wsl_path(r"C:\Users\Dylan\Desktop\x.py") == "/mnt/c/Users/Dylan/Desktop/x.py"


def test_read_checkpoint_ids_skips_blank_and_bad_lines(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"track_id": "a", "voice_prob": 0.9}) + "\n"
        + "\n"
        + "not-json\n"
        + json.dumps({"track_id": "b", "voice_prob": 0.1}) + "\n",
        encoding="utf-8",
    )
    assert read_checkpoint_ids(str(ckpt)) == {"a", "b"}


def test_merge_sidecar_npz_aligns_and_nan_fills(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"track_id": "a", "voice_prob": 0.9}) + "\n"
        + json.dumps({"track_id": "b", "missing": True}) + "\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "instrumental_sidecar.npz"
    merge_sidecar_npz(str(sidecar), str(ckpt), columns={"voice_prob": "voice_prob"})
    data = np.load(str(sidecar), allow_pickle=True)
    ids = list(data["track_ids"])
    vp = data["voice_prob"]
    by_id = {t: vp[i] for i, t in enumerate(ids)}
    assert abs(float(by_id["a"]) - 0.9) < 1e-6
    assert np.isnan(float(by_id["b"]))  # missing track -> NaN, still present in alignment


def test_append_checkpoint_writes_and_flushes(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    with open(ckpt, "a", encoding="utf-8") as fh:
        append_checkpoint(fh, {"track_id": "x", "voice_prob": 0.7})
        append_checkpoint(fh, {"track_id": "y", "voice_prob": 0.3})

    lines = ckpt.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [r["track_id"] for r in records] == ["x", "y"]
    assert records[0]["voice_prob"] == 0.7
    assert records[1]["voice_prob"] == 0.3
    assert read_checkpoint_ids(str(ckpt)) == {"x", "y"}


def test_merge_sidecar_npz_writes_meta(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(json.dumps({"track_id": "a", "voice_prob": 0.9}) + "\n", encoding="utf-8")
    sidecar = tmp_path / "instrumental_sidecar.npz"
    merge_sidecar_npz(str(sidecar), str(ckpt), columns={"voice_prob": "voice_prob"}, meta={"model": "foo"})
    data = np.load(str(sidecar), allow_pickle=True)
    assert str(data["model"]) == "foo"


def test_merge_sidecar_npz_backs_up_existing_sidecar(tmp_path):
    sidecar = tmp_path / "instrumental_sidecar.npz"
    np.savez_compressed(str(sidecar), track_ids=np.array(["old"], dtype=object))

    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(json.dumps({"track_id": "a", "voice_prob": 0.5}) + "\n", encoding="utf-8")

    merge_sidecar_npz(str(sidecar), str(ckpt), columns={"voice_prob": "voice_prob"})

    assert sidecar.exists()
    data = np.load(str(sidecar), allow_pickle=True)
    assert list(data["track_ids"]) == ["a"]

    backups = list(tmp_path.glob("instrumental_sidecar.npz.*.bak"))
    assert len(backups) == 1
