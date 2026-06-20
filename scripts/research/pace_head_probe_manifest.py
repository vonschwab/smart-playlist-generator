import os
import sqlite3
import sys

WT = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/.claude/worktrees/pace-energy-steering"
ROOT = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3"
sys.path.insert(0, WT)
from scripts.research.pace_eval_corpus import resolve_corpus  # noqa: E402

con = sqlite3.connect(f"file:{ROOT}/data/metadata.db?mode=ro", uri=True)
tracks, counts = resolve_corpus(con)
ids = [t.track_id for t in tracks]
paths = {}
cur = con.cursor()
for i in range(0, len(ids), 900):
    b = ids[i:i + 900]
    ph = ",".join("?" for _ in b)
    for tid, fp in cur.execute(f"SELECT track_id,file_path FROM tracks WHERE track_id IN ({ph})", b):
        paths[str(tid)] = fp
con.close()


def w(p):
    p = p.replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:] if len(p) > 1 and p[1] == ":" else p


out = f"{ROOT}/docs/run_audits/pace_axis_eval/head_probe_manifest.tsv"
os.makedirs(os.path.dirname(out), exist_ok=True)
n = 0
with open(out, "w", encoding="utf-8") as f:
    f.write("track_id\tflow_type\tregister\twsl_path\n")
    for t in tracks:
        fp = paths.get(t.track_id)
        if fp:
            f.write(f"{t.track_id}\t{t.flow_type}\t{t.register}\t{w(fp)}\n")
            n += 1
print("wrote", out, "rows:", n, "counts:", counts)
