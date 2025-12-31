"""
Bounded buffer with caps on count and total bytes.
Oldest entries are dropped when limits are exceeded.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, List


class BoundedBuffer:
    def __init__(self, max_events: int = 2000, max_bytes: int = 2 * 1024 * 1024):
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._items: deque[str] = deque()
        self._bytes = 0

    def append(self, item: str) -> None:
        item = item if isinstance(item, str) else str(item)
        item_bytes = len(item.encode("utf-8", "ignore"))
        self._items.append(item)
        self._bytes += item_bytes
        self._trim()

    def _trim(self) -> None:
        while self._items and (len(self._items) > self.max_events or self._bytes > self.max_bytes):
            removed = self._items.popleft()
            self._bytes -= len(removed.encode("utf-8", "ignore"))
            if self._bytes < 0:
                self._bytes = 0

    def items(self) -> List[str]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[str]:
        return iter(self._items)
