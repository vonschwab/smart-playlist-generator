"""
Job type definitions for library maintenance tasks.
"""
from enum import Enum


class JobType(str, Enum):
    """Supported job types."""

    SCAN_LIBRARY = "scan_library"
    UPDATE_GENRES = "update_genres"
    UPDATE_SONIC = "update_sonic"
    BUILD_ARTIFACTS = "build_artifacts"

    @classmethod
    def ordered_pipeline(cls) -> list:
        """Return the default pipeline order."""
        return [
            cls.SCAN_LIBRARY,
            cls.UPDATE_GENRES,
            cls.UPDATE_SONIC,
            cls.BUILD_ARTIFACTS,
        ]

    def label(self) -> str:
        """Human-friendly label."""
        labels = {
            JobType.SCAN_LIBRARY: "Scan Library",
            JobType.UPDATE_GENRES: "Update Genres",
            JobType.UPDATE_SONIC: "Update Sonic Features",
            JobType.BUILD_ARTIFACTS: "Build Artifacts",
        }
        return labels.get(self, self.value)
