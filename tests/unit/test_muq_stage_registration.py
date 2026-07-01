from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER, LibraryPipelineRequest


def test_muq_is_in_order_right_after_mert():
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "muq" in order
    assert order.index("muq") == order.index("mert") + 1


def test_request_accepts_muq_stage():
    req = LibraryPipelineRequest(config_path="config.yaml", stages=["muq"])
    assert "muq" in req.stages
