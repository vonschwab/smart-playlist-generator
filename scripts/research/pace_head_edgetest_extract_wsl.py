import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import essentia
essentia.log.warningActive = False
import essentia.standard as es

M = "/opt/ess/models"
ROOT = "/mnt/c/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3"
_name = sys.argv[1] if len(sys.argv) > 1 else "instr_test"
MAN = f"{ROOT}/docs/run_audits/pace_axis_eval/{_name}_manifest.tsv"
OUT = f"{ROOT}/docs/run_audits/pace_axis_eval/{_name}_results.tsv"

emb_model = es.TensorflowPredictMusiCNN(graphFilename=f"{M}/msd-musicnn-1.pb", output="model/dense/BiasAdd")
instr = es.TensorflowPredict2D(graphFilename=f"{M}/voice_instrumental-msd-musicnn-1.pb", output="model/Softmax")  # [instrumental, voice]
av = es.TensorflowPredict2D(graphFilename=f"{M}/emomusic-msd-musicnn-2.pb", output="model/Identity")  # [val, arousal]
dance = es.TensorflowPredict2D(graphFilename=f"{M}/danceability-msd-musicnn-1.pb", output="model/Softmax")

rows = [ln.rstrip("\n").split("\t") for ln in open(MAN, encoding="utf-8")][1:]
out = open(OUT, "w", encoding="utf-8")
out.write("label\tartist\ttitle\tinstrumental\tarousal\tdance\n")
for lab, tid, artist, title, path in rows:
    try:
        a = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        e = emb_model(a)
        iv = float(np.mean(instr(e)[:, 0]))   # P(instrumental)
        ar = float(np.mean(av(e)[:, 1]))
        dc = float(np.mean(dance(e)[:, 0]))
        out.write(f"{lab}\t{artist[:22]}\t{title[:30]}\t{iv:.3f}\t{ar:.2f}\t{dc:.3f}\n")
        out.flush()
        print(f"{lab} {title[:30]:30} instr={iv:.3f}", file=sys.stderr, flush=True)
    except Exception as ex:
        print(f"ERR {title}: {ex!r}", file=sys.stderr, flush=True)
out.close()
print("DONE", file=sys.stderr, flush=True)
