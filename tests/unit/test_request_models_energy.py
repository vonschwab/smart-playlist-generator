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
