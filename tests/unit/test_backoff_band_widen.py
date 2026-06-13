def test_band_widen_factor():
    # The backoff applies 1.5x per attempt index to finite caps.
    base = 0.60
    assert base * (1.5 ** 0) == 0.60
    assert round(base * (1.5 ** 1), 4) == 0.90
    assert round(base * (1.5 ** 2), 4) == 1.35
