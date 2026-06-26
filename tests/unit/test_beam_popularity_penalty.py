from src.playlist.pier_bridge.beam import _popularity_factor


def test_popularity_factor_grades_and_handles_nan():
    assert _popularity_factor(1.0, 0.3) == 1.0           # banger: no demotion
    assert abs(_popularity_factor(0.0, 0.3) - 0.7) < 1e-9   # bottom of chart: full strength
    assert abs(_popularity_factor(float("nan"), 0.3) - 0.7) < 1e-9  # unknown -> max (ruthless)
    assert abs(_popularity_factor(0.5, 0.2) - 0.9) < 1e-9    # graded
    assert _popularity_factor(0.2, 0.0) == 1.0           # strength 0 -> inert


def test_popularity_factor_bounded_in_zero_one():
    for p in (0.0, 0.25, 0.5, 0.75, 1.0, float("nan")):
        for s in (0.1, 0.3, 0.9):
            f = _popularity_factor(p, s)
            assert 0.0 < f <= 1.0
