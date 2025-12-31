from src.playlist_gui.jobs.job_store import JobStore
from src.playlist_gui.jobs.job_model import Job, JobType, JobStatus


def test_job_store_redacts(tmp_path):
    store = JobStore(app_name="PlaylistGeneratorTest", max_items=5)
    store.history_path = tmp_path / "jobs_history.json"

    job = Job(job_id="1", job_type=JobType.SCAN_LIBRARY, status=JobStatus.SUCCESS, summary="token=ABC", error_message="secret=XYZ")
    store.save_history([job])

    text = store.history_path.read_text()
    assert "ABC" not in text
    assert "XYZ" not in text
