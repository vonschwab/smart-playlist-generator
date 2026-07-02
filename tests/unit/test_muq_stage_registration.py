from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER, LibraryPipelineRequest


def test_muq_registered_and_mert_gone():
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "muq" in order
    assert "mert" not in order


def test_request_accepts_muq_stage():
    req = LibraryPipelineRequest(config_path="config.yaml", stages=["muq"])
    assert "muq" in req.stages
