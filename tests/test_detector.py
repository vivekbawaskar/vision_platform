"""Detector logic tested against a stand-in YOLO model (no weights needed)."""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("cv2")

from detector import YOLOObjectDetector  # noqa: E402


class Tensor:
    """Mimics the bits of a torch tensor the detector touches."""

    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Boxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = Tensor(xyxy), Tensor(conf), Tensor(cls)
        self.id = None if ids is None else Tensor(ids)
        self._n = len(xyxy)

    def __len__(self):
        return self._n


class Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Records the kwargs it is called with and returns scripted boxes."""

    calls = []
    boxes = None  # set per test

    def __init__(self, path):
        self.names = {0: "person", 67: "cell phone"}

    def _run(self, kind, frame, kwargs):
        FakeYOLO.calls.append((kind, kwargs))
        return iter([Result(FakeYOLO.boxes)])

    def predict(self, frame, **kwargs):
        return self._run("predict", frame, kwargs)

    def track(self, frame, **kwargs):
        return self._run("track", frame, kwargs)


@pytest.fixture()
def detector(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    FakeYOLO.boxes = None
    det = YOLOObjectDetector("fake.pt")
    FakeYOLO.calls.clear()  # forget the warm-up call
    return det


def one_person(ids=None):
    return Boxes([[10, 20, 110, 220]], [0.9], [0], ids)


def test_coordinates_are_converted_to_x_y_w_h(detector):
    FakeYOLO.boxes = one_person()
    _, dets = detector.process_frame(np.zeros((480, 640, 3), np.uint8))
    d = dets[0]
    assert (d["x"], d["y"], d["w"], d["h"]) == (10, 20, 100, 200)
    assert d["class_name"] == "person" and abs(d["confidence"] - 0.9) < 1e-6
    assert d["track_id"] is None


def test_output_is_rgb_not_bgr(detector):
    FakeYOLO.boxes = Boxes(np.zeros((0, 4)), [], [])
    blue_bgr = np.zeros((100, 100, 3), np.uint8)
    blue_bgr[:] = (255, 0, 0)  # BGR blue
    rgb, dets = detector.process_frame(blue_bgr)
    assert dets == []
    assert tuple(rgb[50, 50]) == (0, 0, 255)  # RGB blue


def test_boxes_and_labels_are_drawn_and_input_untouched(detector):
    FakeYOLO.boxes = one_person()
    frame = np.zeros((480, 640, 3), np.uint8)
    rgb, _ = detector.process_frame(frame)
    assert rgb.sum() > 0           # something was drawn
    assert frame.sum() == 0        # caller's frame not mutated


def test_inference_is_lazy_stream_with_threshold_and_class_ids(detector):
    FakeYOLO.boxes = one_person()
    detector.process_frame(
        np.zeros((100, 100, 3), np.uint8), 0.8, ["Person", "cell phone"]
    )
    kind, kw = FakeYOLO.calls[-1]
    assert kind == "predict"
    assert kw["stream"] is True
    assert kw["conf"] == 0.8
    assert sorted(kw["classes"]) == [0, 67]


def test_no_filter_means_all_classes(detector):
    FakeYOLO.boxes = one_person()
    detector.process_frame(np.zeros((100, 100, 3), np.uint8), 0.5, [])
    assert FakeYOLO.calls[-1][1]["classes"] is None


def test_unknown_class_names_skip_inference(detector):
    frame = np.zeros((100, 100, 3), np.uint8)
    rgb, dets = detector.process_frame(frame, 0.5, ["unicorn"])
    assert dets == [] and rgb.shape == frame.shape
    assert FakeYOLO.calls == []


def test_tracking_uses_track_and_reports_ids(detector):
    FakeYOLO.boxes = one_person(ids=[7])
    _, dets = detector.process_frame(
        np.zeros((100, 100, 3), np.uint8), track=True
    )
    kind, kw = FakeYOLO.calls[-1]
    assert kind == "track" and kw["persist"] is True
    assert dets[0]["track_id"] == 7


def test_box_partly_outside_frame_does_not_crash(detector):
    FakeYOLO.boxes = Boxes([[-30, -10, 500, 400]], [0.7], [67])
    rgb, dets = detector.process_frame(np.zeros((100, 120, 3), np.uint8))
    assert rgb.shape == (100, 120, 3) and dets[0]["class_name"] == "cell phone"


def test_empty_frame_raises(detector):
    with pytest.raises(ValueError):
        detector.process_frame(np.zeros((0, 0, 3), np.uint8))
