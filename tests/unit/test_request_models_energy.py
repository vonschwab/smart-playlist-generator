from src.playlist.request_models import (
    ANALYZE_LIBRARY_STAGE_ORDER,
    LibraryPipelineRequest,
)


def test_energy_in_stage_order_after_artifacts():
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "energy" in order
    assert order.index("energy") == order.index("artifacts") + 1


def test_clean_stages_keeps_energy():
    req = LibraryPipelineRequest(config_path="dummy.yaml", stages=["energy"])
    assert req.stages == ["energy"]


def test_default_run_includes_energy():
    req = LibraryPipelineRequest(config_path="dummy.yaml")
    assert "energy" in req.stages


def test_gui_default_runs_sonnet_adjudicator_not_legacy_enrich():
    # L3 regression: the GUI/web default must run the album-grain Sonnet path
    # (adjudicate+apply), not the retired tag-grain enrich stage.
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "adjudicate" in order and "apply" in order
    assert "enrich" not in order
    assert order.index("apply") == order.index("adjudicate") + 1
    assert order.index("adjudicate") > order.index("mert")


def test_gui_and_cli_default_orders_match():
    # These two diverged silently (GUI on enrich, CLI on adjudicate/apply); they
    # are now the same constant. Keep them identical.
    import scripts.analyze_library as al

    assert list(al.STAGE_ORDER_DEFAULT) == list(ANALYZE_LIBRARY_STAGE_ORDER)
