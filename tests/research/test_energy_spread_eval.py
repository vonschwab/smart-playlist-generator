import numpy as np

from scripts.research.energy_spread_eval import pier_arousal_span


def test_pier_arousal_span_basic():
    track_ids = ["a", "b", "c", "d"]
    arousal = np.array([0.0, 1.0, -1.0, np.nan])
    # medoids a,c span [−1, 0] => 1.0
    assert pier_arousal_span(["a", "c"], track_ids, arousal) == 1.0
    # single finite => 0.0
    assert pier_arousal_span(["a"], track_ids, arousal) == 0.0
    # NaN ignored
    assert pier_arousal_span(["a", "d"], track_ids, arousal) == 0.0
