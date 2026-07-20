# src/setup/result.py
"""One structured result type shared by health checks and connection tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: Literal["pass", "warn", "fail"]
    summary: str
    fix_hint: str | None = None
