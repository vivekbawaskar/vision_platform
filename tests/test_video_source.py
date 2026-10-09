"""VideoStream tested with a scripted flaky camera and a real tiny video."""
import os
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import video_source  # noqa: E402
from video_source import VideoStream  # noqa: E402


class FakeCapture:
    """Scripted camera. Each (re)open consumes the next 'session'.

    A session is the number of good frames before reads start failing
    (None = never fails); ``None`` in the list of sessions = open fails.
    """

    sessions = []
    opens = 0

    def __init__(self, source, *args):
        FakeCapture.opens += 1
        plan = FakeCapture.sessions.pop(0) if FakeCapture.sessions else "fail"
        self._open = plan != "fail"
        self._budget = None if plan == "fail" else plan
        self._index = FakeCapture.opens
        self._served = 0

    def isOpened(self):
        return self._open

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def release(self):
        self._open = False

    def read(self):
        time.sleep(0.002)
        if not self._open or (
            self._budget is not None and self._served >= self._budget
        ):
            return False, None
        self._served += 1
        return True, np.full((4, 4, 3), self._index * 100 + self._served, np.uint8)


@pytest.fixture()
def fake_camera(monkeypatch):
    FakeCapture.sessions, FakeCapture.opens = [], 0
    monkeypatch.setattr(video_source.cv2, "VideoCapture", FakeCapture)
    return FakeCapture


def pull(stream, seconds=2.0):
    """Read for a while; return the set of pixel values observed."""
    seen, end = set(), time.perf_counter() + seconds
    while time.perf_counter() < end and stream.is_alive:
        ok, frame = stream.read()
        if ok:
            seen.add(int(frame[0, 0, 0]))
    return seen


def test_live_stream_delivers_frames(fake_camera):
    fake_camera.sessions = [None]  # healthy forever
    stream = VideoStream(0)
    assert stream.start()
    ok = False
    for _ in range(50):
        ok, frame = stream.read()
        if ok:
            break
    stream.release()
    assert ok and frame.shape == (4, 4, 3)


def test_reconnects_after_camera_drops(fake_camera):
    fake_camera.sessions = [5, None]  # 5 good frames, drop, then healthy
    stream = VideoStream(0, reconnect_delay=0.01)
    assert stream.start()
    deadline = time.perf_counter() + 4
    seen = set()
    while time.perf_counter() < deadline and not any(v >= 200 for v in seen):
        ok, frame = stream.read()
        if ok:
            seen.add(int(frame[0, 0, 0]))
    stream.release()
    assert any(v >= 200 for v in seen), "never recovered after the drop"
    assert fake_camera.opens == 2
    assert stream.error is None


def test_gives_up_after_max_reconnects_and_reports_error(fake_camera):
    fake_camera.sessions = [3]  # then every re-open fails
    stream = VideoStream(0, max_reconnects=2, reconnect_delay=0.01)
    assert stream.start()
    pull(stream, seconds=3.0)
    stream.release()
    assert stream.error and "reconnect" in stream.error
    assert not stream.is_alive
    assert fake_camera.opens == 3  # 1 initial + 2 attempts


def test_unopenable_source_fails_cleanly(fake_camera):
    fake_camera.sessions = ["fail"]
    stream = VideoStream(0)
    assert stream.start() is False
    assert "Unable to open" in stream.error


def test_release_stops_the_reader_thread(fake_camera):
    fake_camera.sessions = [None]
    stream = VideoStream(0)
    stream.start()
    stream.release()
    assert not stream._thread.is_alive()


def _make_video(path, frames=12):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 120, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_file_plays_once_then_reports_end(tmp_path):
    path = str(tmp_path / "clip.avi")
    _make_video(path, 12)
    stream = VideoStream(path)
    assert stream.start() and stream.is_file
    count = 0
    while stream.read()[0]:
        count += 1
    stream.release()
    assert count == 12 and stream.ended and not stream.is_alive


def test_file_loops_when_requested(tmp_path):
    path = str(tmp_path / "clip.avi")
    _make_video(path, 5)
    stream = VideoStream(path, loop=True)
    stream.start()
    count = sum(1 for _ in range(20) if stream.read()[0])
    stream.release()
    assert count == 20 and not stream.ended
