"""Build the energy-descriptor probe manifest (pre-registration).

Writes docs/run_audits/energy_descriptor_probe/manifest.tsv with PRE-REGISTERED
expected buckets per track, committed BEFORE the Essentia scores are seen.
The manifest is the independent (non-circular) arm of the eval per the
evaluation-methodology skill: expectations are genre/known-character priors,
not derived from Essentia. ANCHOR rows were already measured in scans 1-2 and
are excluded from agreement stats (scale calibration only).

Pre-registered bucket thresholds (committed here; calibrated to scans 1-2,
emoMusic arousal 1-9 scale + danceability P):
    arousal:  LO < 4.5   MID 4.5-5.5   HI >= 5.5
    dance:    LO < 0.45  MID 0.45-0.75 HI >= 0.75
"""
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "data", "metadata.db")
OUT_DIR = os.path.join(ROOT, "docs", "run_audits", "energy_descriptor_probe")

# (track_id, role, stratum, exp_arousal, exp_dance, trap/rationale)
ROWS = [
    # ---- TEST items (blind: scores NOT seen at pre-registration) ----
    ("5a401854442c5d31095498f7752522f7", "test", "dance/house",     "HI",  "HI",  "4-on-floor french house"),
    ("d655a1b124f63ff7ff1c9e499fc85b83", "test", "dance/synthpop",  "HI",  "HI",  "synthpop banger"),
    ("ef8ba9e72596289429b2b55a3196d5bc", "test", "dance/dancepunk", "HI",  "HI",  "dancepunk"),
    ("87fa81cd66842e048dd2d47e8d9c3f63", "test", "dance/electro",   "HI",  "HI",  "electronic dance"),
    ("45bd1bde8a0a9d3a9e38d75d2300094f", "test", "hiphop",          "HI",  "HI",  "trap banger: beat-driven+intense"),
    ("daacf0a535b4fd1a445b92efb6689de4", "test", "hiphop",          "HI",  "HI",  "frantic fast rap"),
    ("dcd610c7985a9657d209c776287904f0", "test", "loud/noise",      "HI",  "LO",  "TRAP: loud pummel, no groove -> dance LO"),
    ("d49ca10ff05db63ce7016dcffcd2e616", "test", "doom",            "HI",  "MID", "TRAP: SLOW doom -> arousal HI not LO"),
    ("dfd9b13d276a0bf6e5baa1ad9e4cc3db", "test", "punk",            "HI",  "MID", "fast simple punk"),
    ("5787e06bba6cb7dcce1fc2f61793742f", "test", "twee",            "MID", "MID", "jangly gentle"),
    ("d81a2ddc9a37f2fcc72750bd11502fe8", "test", "indie",           "MID", "MID", "mid indie rock"),
    ("70b200f4bb5a66ed87ed4d7df0692e37", "test", "indie",           "MID", "MID", "lo-fi indie rock"),
    ("8b8cfdd6e4a9b6224f4c3316c575c3af", "test", "postrock",        "MID", "LO",  "TRAP: quiet->loud build, mean masks arc"),
    ("80cdc0f5f87d53140330526aa18f0723", "test", "postrock",        "MID", "LO",  "dynamic post-rock"),
    ("ac2e4a6a177fb75360a93a80d1a781e6", "test", "shoegaze",        "MID", "MID", "dreamy mid shoegaze"),
    ("8b0c959497784f03f103b14a13cc2831", "test", "dreampop",        "LO",  "MID", "languid dreampop w/ pulse"),
    ("ad125dcfe17bad74ebd838668051ffb7", "test", "jazz-ballad",     "LO",  "LO",  "rubato ballad"),
    ("5f0feb808fe3c9de763d04f7fbad5a60", "test", "jazz-swing",      "MID", "MID", "TRAP: uptempo acoustic swing"),
    ("5347efa7fc2ad6129f60aeeeac69fdd9", "test", "folk",            "LO",  "LO",  "solo acoustic quiet"),
    ("9cc4ace8edc409d5e94a08bfdf2a3615", "test", "folk",            "LO",  "LO",  "solo acoustic quiet"),
    ("9db39361eed3b3be4911a07a3f3a9f4b", "test", "soul",            "MID", "MID", "TRAP: slow groove -> dance MID not LO"),
    ("f2e689e12e231087caeb41ed102aa077", "test", "neoclassical",    "MID", "LO",  "TRAP: energetic strings, NO beat -> dance LO"),
    ("3e08b3a1bbf6fca0d78ddf25d62eb574", "test", "piano",           "LO",  "LO",  "solo piano quiet"),
    ("0e7b7d7263d06d3d4fa2ae966e215c15", "test", "experimental",    "LO",  "LO",  "mic-feedback art piece ~ drone"),
    ("cb72f36ac9cc59e27503f9b0b6b2be5b", "test", "idm",             "MID", "MID", "downtempo hazy w/ beat"),
    ("3c12f55293fd5658fbfb1637f02452cc", "test", "drone",           "LO",  "LO",  "noise/drone"),
    # ---- ANCHOR items (already measured in scans 1-2; calibration only) ----
    ("e35c291a3c5f923d20727cb4cd2c5c83", "anchor", "dance",   "HI", "HI",  "Jessy Lanza (anchor)"),
    ("a0c25b8cf70f48d3791d5062c62f1751", "anchor", "dance",   "HI", "HI",  "Beyonce CUFF IT (anchor)"),
    ("c1ccca513a656d1c50238d05c81e5985", "anchor", "dance",   "HI", "HI",  "Whitney (anchor)"),
    ("e117d8618a7f6d1f57c14d79ebd31fe6", "anchor", "ambient", "LO", "LO",  "Stars of the Lid (anchor)"),
    ("70c9eca730c12726f1ada63afc2fd3e6", "anchor", "ambient", "LO", "LO",  "Alex Somers (anchor)"),
    ("0bef7a0ce4404c638bd42e0bfd03e08e", "anchor", "drone",   "LO", "LO",  "YLT Georgia (anchor)"),
    ("aa2f61fcd336932ebc09b3cd27dc5f48", "anchor", "rock",    "HI", "HI",  "Ovlov (anchor)"),
    ("7b9d07d22e4ff5988e51ebedf9250985", "anchor", "shoegaze","HI", "MID", "MBV Feed Me (anchor)"),
]


def win_to_wsl(p: str) -> str:
    p = p.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()
    out = os.path.join(OUT_DIR, "manifest.tsv")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("track_id\trole\tstratum\texp_arousal\texp_dance\ttrap\tartist\ttitle\twsl_path\n")
        for tid, role, stratum, ea, ed, trap in ROWS:
            r = cur.execute(
                "SELECT artist, title, file_path FROM tracks WHERE track_id=?", [tid]
            ).fetchone()
            if not r:
                print("MISSING", tid, trap)
                continue
            artist, title, path = r
            f.write(f"{tid}\t{role}\t{stratum}\t{ea}\t{ed}\t{trap}\t{artist}\t{title}\t{win_to_wsl(path)}\n")
            n += 1
    con.close()
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
