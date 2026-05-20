from src.playlist.title_quality import compute_title_artifact_penalty


def test_no_flags_no_penalty():
    assert compute_title_artifact_penalty(
        title="Library Pictures",
        weights={"demo": 0.10, "live": 0.05, "remix": 0.08},
    ) == 0.0


def test_single_flag_applies_weight():
    p = compute_title_artifact_penalty(
        title="Lee #2 (8 Track Demo)",
        weights={"demo": 0.10},
    )
    assert abs(p - 0.10) < 1e-9


def test_multiple_flags_sum():
    # 'Live At Venue (Take 2)' triggers both live and take
    p = compute_title_artifact_penalty(
        title="Live At The Village Vanguard (Take 2)",
        weights={"live": 0.05, "take": 0.07},
    )
    assert abs(p - 0.12) < 1e-9


def test_unmapped_flags_ignored():
    # 'remaster' flag fires, but no weight provided -> no penalty
    p = compute_title_artifact_penalty(
        title="Some Song (Remastered 2025)",
        weights={"demo": 0.10},
    )
    assert p == 0.0
