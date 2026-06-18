import numpy as np
from scripts.research import pace_eval_run as run


def test_score_candidates_ranks_separating_feature_high():
    # 4 corpus tracks; arousal_p50 cleanly separates adjacent (close) from random (far)
    ids = ["a1", "a2", "b1", "b2"]
    corpus_index = {t: i for i, t in enumerate(ids)}
    pairs = {
        "adjacent": [("a1", "a2"), ("b1", "b2")],
        "non_adjacent_same_album": [],
        "random_cross": [("a1", "b1"), ("a2", "b2")],
    }
    # arousal_p50: a-cluster ~0, b-cluster ~10 -> adjacent close, cross far
    zs = {k: np.zeros(4) for k in __import__("scripts.research.pace_eval_features",
                                             fromlist=["SCALAR_KEYS"]).SCALAR_KEYS}
    zs["arousal_p50"] = np.array([0.0, 0.1, 10.0, 10.1])
    zs["danceability"] = np.array([0.0, 0.1, 10.0, 10.1])
    zt = np.zeros((4, 9))
    res = run.score_candidates(corpus_index, pairs, zs, zt)
    assert res["arousal_p50"]["auc_adj_vs_random"] == 1.0
    assert res["arousal_p50"]["adjacent"]["n"] == 2
