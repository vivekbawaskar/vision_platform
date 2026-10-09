"""Detection throttling to prevent database bloat.

At 30 FPS a single visible person would produce 30 rows per second. The
throttle decides which detections are worth persisting:

* a class is logged immediately when it *enters* the frame (it was not seen
  for ``reentry_gap`` seconds), and
* afterwards at most once every ``interval`` seconds per class.

Only the highest-confidence detection of each class is logged per decision.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence


class DetectionThrottle:
    """Per-class rate limiter for detection logging."""

    def __init__(self, interval: float = 1.0, reentry_gap: float = 2.0) -> None:
        self.interval = interval
        self.reentry_gap = reentry_gap
        self._last_logged: Dict[str, float] = {}
        self._last_seen: Dict[str, float] = {}

    def reset(self) -> None:
        """Forget all state (e.g. when the video source changes)."""
        self._last_logged.clear()
        self._last_seen.clear()

    def select(
        self,
        detections: Sequence[dict],
        now: Optional[float] = None,
    ) -> List[dict]:
        """Return the detections that should be written to the database."""
        now = time.monotonic() if now is None else now

        best: Dict[str, dict] = {}
        for det in detections:
            name = str(det["class_name"])
            if name not in best or det["confidence"] > best[name]["confidence"]:
                best[name] = det

        selected: List[dict] = []
        for name, det in best.items():
            last_seen = self._last_seen.get(name)
            last_logged = self._last_logged.get(name)
            entered = last_seen is None or (now - last_seen) > self.reentry_gap
            due = last_logged is None or (now - last_logged) >= self.interval
            if entered or due:
                selected.append(det)
                self._last_logged[name] = now
            self._last_seen[name] = now
        return selected
