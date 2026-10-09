"""Video acquisition with automatic reconnection.

Live sources (webcam / RTSP) are read by a background thread that always
keeps only the *latest* frame, so slow inference never builds up latency.
Video files are read synchronously and paced to their native frame rate.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Source = Union[int, str]


class VideoStream:
    """Unified reader for webcams, RTSP streams and local video files."""

    def __init__(
        self,
        source: Source,
        loop: bool = False,
        max_reconnects: int = 5,
        reconnect_delay: float = 1.5,
        read_timeout: float = 1.0,
    ) -> None:
        self.source = source
        self.loop = loop
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay
        self.read_timeout = read_timeout
        self.is_file = isinstance(source, str) and os.path.isfile(source)

        self.ended = False                 # a non-looping file finished
        self.error: Optional[str] = None   # fatal, unrecoverable failure

        self._cap: Optional[cv2.VideoCapture] = None
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._counter = 0
        self._last_returned = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_interval = 1.0 / 30
        self._last_read_ts = 0.0

    # ------------------------------------------------------------ properties
    @property
    def is_alive(self) -> bool:
        """True while the stream can still produce frames."""
        return not self.ended and self.error is None

    # -------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Open the source; returns False if it cannot be opened."""
        self._cap = self._open()
        if self._cap is None:
            self.error = f"Unable to open video source: {self.source!r}"
            return False

        if self.is_file:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self._frame_interval = 1.0 / fps
        else:
            self._thread = threading.Thread(
                target=self._reader, name="video-reader", daemon=True
            )
            self._thread.start()
        return True

    def release(self) -> None:
        """Stop the reader thread and free the capture device."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open(self) -> Optional[cv2.VideoCapture]:
        """Open a capture handle, or return None on failure."""
        backends = [None]
        if os.name == "nt" and isinstance(self.source, int):
            # DirectShow opens webcams much faster on Windows; fall back to
            # OpenCV's default backend if it is unavailable.
            backends = [cv2.CAP_DSHOW, None]
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, backend)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
                except cv2.error:
                    pass
                return cap
            if cap is not None:
                cap.release()
        return None

    # ------------------------------------------------------------ live reader
    def _reader(self) -> None:
        """Background loop: keep the newest frame, reconnect on failure."""
        failures = 0
        while not self._stop.is_set():
            ok, frame = False, None
            try:
                if self._cap is not None:
                    ok, frame = self._cap.read()
            except cv2.error as exc:
                logger.warning("Frame read raised: %s", exc)

            if ok and frame is not None:
                failures = 0
                with self._cond:
                    self._frame = frame
                    self._counter += 1
                    self._cond.notify_all()
                continue

            failures += 1
            if failures < 3:           # tolerate brief hiccups
                time.sleep(0.05)
                continue
            if not self._reconnect():
                with self._cond:
                    if not self._stop.is_set():
                        self.error = (
                            f"Lost video source {self.source!r} and could not "
                            f"reconnect after {self.max_reconnects} attempts."
                        )
                    self._cond.notify_all()
                return
            failures = 0

    def _reconnect(self) -> bool:
        """Release and re-open the capture device with a fixed back-off."""
        for attempt in range(1, self.max_reconnects + 1):
            if self._stop.is_set():
                return False
            logger.warning(
                "Reconnecting to %r (attempt %d/%d)",
                self.source, attempt, self.max_reconnects,
            )
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            time.sleep(self.reconnect_delay)
            self._cap = self._open()
            if self._cap is not None:
                return True
        return False

    # ------------------------------------------------------------------- read
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)``; ``ok`` is False if no new frame arrived."""
        if self.is_file:
            return self._read_file()

        with self._cond:
            self._cond.wait_for(
                lambda: self._counter != self._last_returned
                or self.error is not None
                or self._stop.is_set(),
                timeout=self.read_timeout,
            )
            if self._counter == self._last_returned or self._frame is None:
                return False, None
            self._last_returned = self._counter
            return True, self._frame

    def _read_file(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next file frame, paced to the file's FPS."""
        if self._cap is None:
            return False, None

        wait = self._frame_interval - (time.perf_counter() - self._last_read_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_read_ts = time.perf_counter()

        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok or frame is None:
            self.ended = True
            return False, None
        return True, frame
