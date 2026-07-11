"""Analyze the energy-descriptor probe: join pre-registration vs measured scores.

Reports per-axis bucket agreement on TEST items only (anchors excluded per
evaluation-methodology), flags disagreements for human ratification, flags
dynamic-mean tracks (wide arousal p10-p90), and runs a determinism/provenance
check on anchors against scans 1-2.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "docs", "run_audits", "energy_descriptor_probe")
MANIFEST = os.path.join(DIR, "manifest.tsv")
RESULTS = os.path.join(DIR, "results.tsv")

# committed thresholds (must match energy_probe_build.py)
def aro_bucket(a):
    return "LO" if a < 4.5 else ("MID" if a < 5.5 else "HI")

def dance_bucket(d):
    return "LO" if d < 0.45 else ("MID" if d < 0.75 else "HI")

ORDER = {"LO": 0, "MID": 1, "HI": 2}

# prior anchor values from scans 1-2 (determinism check): tid -> (arousal, dance)
PRIOR = {
    "e35c291a3c5f923d20727cb4cd2c5c83": (5.890, 0.987),
    "a0c25b8cf70f48d3791d5062c62f1751": (5.120, 0.998),
    "c1ccca513a656d1c50238d05c81e5985": (6.452, 0.979),
    "e117d8618a7f6d1f57c14d79ebd31fe6": (4.180, 0.213),
    "70c9eca730c12726f1ada63afc2fd3e6": (3.681, 0.436),
    "0bef7a0ce4404c638bd42e0bfd03e08e": (4.450, 0.562),
    "aa2f61fcd336932ebc09b3cd27dc5f48": (6.159, 0.921),
    "7b9d07d22e4ff5988e51ebedf9250985": (6.474, 0.909),
}


def load_tsv(p):
    with open(p, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        return [dict(zip(hdr, ln.rstrip("\n").split("\t"))) for ln in f]


def main():
    man = {r["track_id"]: r for r in load_tsv(MANIFEST)}
    res = {r["track_id"]: r for r in load_tsv(RESULTS)}

    rows = []
    for tid, m in man.items():
        r = res.get(tid)
        if not r or r["arousal"] == "ERR":
            rows.append((m, None))
            continue
        rows.append((m, r))

    # sort: test first by measured arousal desc, then anchors
    def sortkey(x):
        m, r = x
        a = float(r["arousal"]) if r else -1
        return (0 if m["role"] == "test" else 1, -a)
    rows.sort(key=sortkey)

    print(f"{'role':6}{'stratum':14}{'track':40}{'exp(a/d)':>10}{'meas a':>8}{'b':>4}{'meas d':>8}{'b':>4}{'dyn':>6}  flags / trap")
    print("-" * 140)
    test_a_exact = test_a_inv = test_d_exact = test_d_inv = test_n = 0
    disagreements = []
    for m, r in rows:
        name = f"{m['artist']} - {m['title']}"[:39]
        if not r:
            print(f"{m['role']:6}{m['stratum']:14}{name:40}{m['exp_arousal']+'/'+m['exp_dance']:>10}{'ERR':>8}")
            continue
        a = float(r["arousal"]); d = float(r["dance"])
        ab = aro_bucket(a); db = dance_bucket(d)
        dyn = float(r["aro_p90"]) - float(r["aro_p10"])
        ea, ed = m["exp_arousal"], m["exp_dance"]
        a_gap = ORDER[ab] - ORDER[ea]
        d_gap = ORDER[db] - ORDER[ed]
        flags = []
        if abs(a_gap) == 2: flags.append("AROUSAL-INVERSION")
        elif a_gap != 0: flags.append(f"aro{'+' if a_gap>0 else ''}{a_gap}")
        if abs(d_gap) == 2: flags.append("DANCE-INVERSION")
        elif d_gap != 0: flags.append(f"dance{'+' if d_gap>0 else ''}{d_gap}")
        if dyn > 2.0: flags.append("DYNAMIC")
        mark = "OK" if not flags else ",".join(flags)
        if m["role"] == "test":
            test_n += 1
            if a_gap == 0: test_a_exact += 1
            if abs(a_gap) == 2: test_a_inv += 1
            if d_gap == 0: test_d_exact += 1
            if abs(d_gap) == 2: test_d_inv += 1
            if a_gap != 0 or d_gap != 0:
                disagreements.append((name, ea, ab, a, ed, db, d, m["trap"], flags))
        print(f"{m['role']:6}{m['stratum']:14}{name:40}{ea+'/'+ed:>10}{a:8.2f}{ab:>4}{d:8.2f}{db:>4}{dyn:6.1f}  {mark}  | {m['trap']}")

    print("\n=== AGREEMENT (test items only, N=%d) ===" % test_n)
    print(f"arousal  exact-bucket: {test_a_exact}/{test_n}   inversions(LO<->HI): {test_a_inv}")
    print(f"dance    exact-bucket: {test_d_exact}/{test_n}   inversions(LO<->HI): {test_d_inv}")

    print("\n=== DISAGREEMENTS to ratify (exp != measured bucket) ===")
    for name, ea, ab, a, ed, db, d, trap, flags in disagreements:
        print(f"  {name:40} aro exp {ea}->meas {ab}({a:.2f}) | dance exp {ed}->meas {db}({d:.2f})  [{','.join(flags)}]  {trap}")

    print("\n=== ANCHOR determinism check (this run vs scans 1-2) ===")
    for tid, (pa, pd) in PRIOR.items():
        r = res.get(tid)
        if not r or r["arousal"] == "ERR":
            print(f"  {man[tid]['artist'][:20]:20} MISSING/ERR"); continue
        a = float(r["arousal"]); d = float(r["dance"])
        da = abs(a - pa); dd = abs(d - pd)
        ok = "OK" if (da < 0.01 and dd < 0.01) else "DRIFT!"
        print(f"  {man[tid]['artist'][:20]:20} aro {a:.3f} vs {pa:.3f} (d{da:.3f}) | dance {d:.3f} vs {pd:.3f} (d{dd:.3f})  {ok}")


if __name__ == "__main__":
    main()
