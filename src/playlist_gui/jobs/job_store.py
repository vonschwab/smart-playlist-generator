"""
Persistence for job history.
Stores compact JSON data in the user data directory so the Jobs pane can show
recent runs after restart. Only non-sensitive fields are persisted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

try:
    from platformdirs import user_data_dir
except ImportError:  # pragma: no cover - platformdirs is optional
    user_data_dir = None

from .job_model import Job
from ..utils.redaction import redact_text


class JobStore:
    """Load/save job history to disk."""

    def __init__(self, app_name: str = "PlaylistGenerator", max_items: int = 50):
        self.app_name = app_name
        self.max_items = max_items
        self.history_path = self._resolve_path()

    def _resolve_path(self) -> Path:
        if user_data_dir:
            base = Path(user_data_dir(self.app_name, self.app_name))
        else:
            base = Path.home() / f".{self.app_name}"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            base = Path.cwd()
        return base / "jobs_history.json"

    def load_history(self) -> List[Job]:
        """Load the most recent jobs."""
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return []

        jobs: List[Job] = []
        for item in payload[-self.max_items :]:
            try:
                jobs.append(Job.from_dict(item))
            except Exception:
                continue
        return jobs

    def save_history(self, jobs: List[Job]) -> None:
        """Persist up to max_items jobs."""
        try:
            data = []
            for job in jobs[-self.max_items :]:
                record = job.to_dict()
                record["summary"] = redact_text(record.get("summary"))
                record["error_message"] = redact_text(record.get("error_message"))
                record["traceback"] = redact_text(record.get("traceback"))
                data.append(record)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True, indent=2)
        except Exception:
            # Persistence failures should never crash the UI.
            return
