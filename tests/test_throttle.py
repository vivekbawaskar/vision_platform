"""Unit tests for the per-class detection throttle."""
from throttle import DetectionThrottle


def det(name, conf=0.9):
    return {"class_name": name, "confidence": conf}


def test_first_sighting_is_logged():
    t = DetectionThrottle(interval=1.0)
    assert len(t.select([det("person")], now=0.0)) == 1


def test_rate_limited_within_interval():
    t = DetectionThrottle(interval=1.0)
    t.select([det("person")], now=0.0)
    assert t.select([det("person")], now=0.5) == []
    assert len(t.select([det("person")], now=1.0)) == 1


def test_best_confidence_per_class_is_chosen():
    t = DetectionThrottle()
    out = t.select([det("person", 0.6), det("person", 0.95)], now=0.0)
    assert len(out) == 1 and out[0]["confidence"] == 0.95


def test_classes_are_throttled_independently():
    t = DetectionThrottle(interval=1.0)
    t.select([det("person")], now=0.0)
    out = t.select([det("person"), det("cell phone")], now=0.4)
    assert [d["class_name"] for d in out] == ["cell phone"]


def test_reentry_logs_immediately_but_flicker_does_not():
    t = DetectionThrottle(interval=10.0, reentry_gap=2.0)
    t.select([det("person")], now=0.0)
    t.select([det("person")], now=1.0)
    # Brief dropout (< gap): not a new entry.
    assert t.select([det("person")], now=2.5) == []
    # Long absence (> gap): counts as entering the frame again.
    assert len(t.select([det("person")], now=9.0)) == 1
