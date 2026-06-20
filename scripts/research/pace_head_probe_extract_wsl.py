import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import essentia
essentia.log.warningActive = False
import essentia.standard as es

M = "/opt/ess/models"
EMB = f"{M}/msd-musicnn-1.pb"
ROOT = "/mnt/c/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3"
MAN = f"{ROOT}/docs/run_audits/pace_axis_eval/head_probe_manifest.tsv"
OUT = f"{ROOT}/docs/run_audits/pace_axis_eval/head_probe.tsv"

emb_model = es.TensorflowPredictMusiCNN(graphFilename=EMB, output="model/dense/BiasAdd")

# head -> (pb, output_node, class_index_to_keep)
HEADS = {
    "arousal":      (f"{M}/emomusic-msd-musicnn-2.pb",            "model/Identity", 1),  # (valence,arousal)
    "danceability": (f"{M}/danceability-msd-musicnn-1.pb",        "model/Softmax",  0),  # (danceable,not)
    "aggressive":   (f"{M}/mood_aggressive-msd-musicnn-1.pb",     "model/Softmax",  0),  # (aggressive,not)
    "relaxed":      (f"{M}/mood_relaxed-msd-musicnn-1.pb",        "model/Softmax",  1),  # (non_relaxed,relaxed)
    "electronic":   (f"{M}/mood_electronic-msd-musicnn-1.pb",     "model/Softmax",  0),  # (electronic,non)
    "acoustic":     (f"{M}/mood_acoustic-msd-musicnn-1.pb",       "model/Softmax",  0),  # (acoustic,non)
    "instrumental": (f"{M}/voice_instrumental-msd-musicnn-1.pb",  "model/Softmax",  0),  # (instrumental,voice)
}
models = {k: es.TensorflowPredict2D(graphFilename=pb, output=node) for k, (pb, node, _) in HEADS.items()}

rows = [ln.rstrip("\n").split("\t") for ln in open(MAN, encoding="utf-8")][1:]
out = open(OUT, "w", encoding="utf-8")
out.write("track_id\t" + "\t".join(HEADS.keys()) + "\tframes\n")
for i, (tid, flow, reg, path) in enumerate(rows, 1):
    try:
        audio = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        if len(audio) == 0:
            out.write(f"{tid}" + "\tERR" * len(HEADS) + "\t0\n"); out.flush(); continue
        emb = emb_model(audio)  # (n,200) — computed ONCE, reused by all heads
        vals = []
        for k, (_, _, idx) in HEADS.items():
            pred = models[k](emb)            # (n,2) or (n,2)
            vals.append(round(float(np.mean(pred[:, idx])), 4))
        out.write(f"{tid}\t" + "\t".join(str(v) for v in vals) + f"\t{emb.shape[0]}\n")
        out.flush()
        print(f"[{i}/{len(rows)}] {tid} ok", file=sys.stderr, flush=True)
    except Exception as e:
        out.write(f"{tid}" + "\tERR" * len(HEADS) + "\t0\n"); out.flush()
        print(f"[{i}/{len(rows)}] {tid} ERR {e!r}", file=sys.stderr, flush=True)
out.close()
print("DONE", file=sys.stderr, flush=True)
