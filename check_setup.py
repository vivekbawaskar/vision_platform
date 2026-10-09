#!/usr/bin/env python3
"""Self-check for the Vision Platform (run this BEFORE `streamlit run`).

    python check_setup.py                      # config, database, model, video
    python check_setup.py --image photo.jpg    # also real detection + end-to-end
    python check_setup.py --camera 0           # also test the webcam

Phases (mirrors the verification workflow in the README):
    1. Environment & configuration
    2. Database (connect, insert, read back, clean up)
    3. Inference engine (YOLOv8 loads and runs, filters work)
    4. Video pipeline (file always, webcam with --camera)
    5. End to end (image -> detector -> throttle -> async writer -> database)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback

RESULTS = []
ARGS = None


def phase(title):
    """Run a check, trap every failure, and record PASS / FAIL / SKIP."""

    def decorator(fn):
        def runner(*a, **k):
            print(f"\n== {title}")
            try:
                status, detail = fn(*a, **k)
            except ImportError as exc:
                status = "FAIL"
                detail = (
                    f"missing package ({exc.name}). "
                    "Run: pip install -r requirements.txt"
                )
            except Exception as exc:  # noqa: BLE001 - report, never crash
                status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
                if ARGS.verbose:
                    traceback.print_exc()
            print(f"[{status}] {detail}")
            RESULTS.append((title, status))
            return status

        return runner

    return decorator


@phase("1. Environment & configuration")
def check_config():
    from config import CONFIG

    hidden = "*" * len(CONFIG.db_password) if CONFIG.db_password else "(empty)"
    print(f"   backend={CONFIG.db_backend}  host={CONFIG.db_host}:"
          f"{CONFIG.db_port}  user={CONFIG.db_user}  password={hidden}  "
          f"db={CONFIG.db_name}")
    if CONFIG.db_backend not in ("mysql", "sqlite", "auto"):
        return "FAIL", f"DB_BACKEND must be mysql|sqlite|auto, got {CONFIG.db_backend!r}"
    env_file = os.path.isfile(".env")
    return "PASS", (".env found" if env_file else
                    ".env not found - using built-in defaults")


@phase("2. Database")
def check_database():
    from database import create_database

    db = create_database()
    if not db.available:
        return "FAIL", f"{db.label}: {db.last_error}"
    marker = "__smoke_test__"
    if not db.log_detection(marker, 0.99, 1, 2, 3, 4):
        return "FAIL", f"insert failed: {db.last_error}"
    logs = db.fetch_recent_logs(50)
    found = logs[logs["object_class"] == marker]
    db.delete_class(marker)
    if found.empty:
        return "FAIL", "inserted row could not be read back"
    note = f" (fallback: {db.fallback_reason})" if db.fallback_reason else ""
    return "PASS", f"{db.label}: insert + read-back + cleanup OK{note}"


def _load_detector():
    from config import CONFIG
    from detector import YOLOObjectDetector

    return YOLOObjectDetector(CONFIG.model_path)


@phase("3. Inference engine")
def check_inference():
    import numpy as np

    detector = _load_detector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb, dets = detector.process_frame(blank, 0.5, None)
    if rgb.shape != blank.shape or not isinstance(dets, list):
        return "FAIL", "unexpected output shape/type from process_frame"
    if not ARGS.image:
        return "PASS", ("model loads and runs on a blank frame "
                        "(pass --image to test real detections)")

    import cv2

    image = cv2.imread(ARGS.image)
    if image is None:
        return "FAIL", f"could not read image {ARGS.image!r}"
    _, all_dets = detector.process_frame(image, 0.25, None)
    _, strict = detector.process_frame(image, 0.95, None)
    names = sorted({d["class_name"] for d in all_dets})
    if not all_dets:
        return "FAIL", "no objects found in the image at conf 0.25"
    only = names[0]
    _, filtered = detector.process_frame(image, 0.25, [only])
    if any(d["class_name"] != only for d in filtered):
        return "FAIL", "class filter let other classes through"
    if len(strict) > len(all_dets):
        return "FAIL", "higher confidence threshold returned MORE detections"
    return "PASS", (f"{len(all_dets)} detections {names}; "
                    f"conf 0.95 -> {len(strict)}; filter '{only}' OK")


@phase("4a. Video pipeline (synthetic file)")
def check_video_file():
    import cv2
    import numpy as np

    from video_source import VideoStream

    path = os.path.join(tempfile.mkdtemp(), "synthetic.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    if not writer.isOpened():
        return "FAIL", "OpenCV cannot write test video (codec problem)"
    for i in range(45):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i * 5, 80), (i * 5 + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()

    stream = VideoStream(path)
    if not stream.start():
        return "FAIL", stream.error or "could not open synthetic video"
    frames = 0
    while True:
        ok, _ = stream.read()
        if not ok:
            break
        frames += 1
    stream.release()
    if not stream.ended or frames < 40:
        return "FAIL", f"read only {frames}/45 frames"
    return "PASS", f"read {frames}/45 frames and detected end of file"


@phase("4b. Video pipeline (webcam)")
def check_webcam():
    from video_source import VideoStream

    if ARGS.camera is None:
        return "SKIP", "pass --camera 0 to test the webcam"
    stream = VideoStream(ARGS.camera)
    if not stream.start():
        return "FAIL", stream.error or "could not open camera"
    frames, start = 0, time.perf_counter()
    while frames < 30 and time.perf_counter() - start < 8:
        ok, _ = stream.read()
        frames += int(ok)
    elapsed = time.perf_counter() - start
    stream.release()
    if frames < 10:
        return "FAIL", f"only {frames} frames in {elapsed:.1f}s"
    return "PASS", f"{frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS)"


@phase("5. End to end")
def check_end_to_end():
    if not ARGS.image:
        return "SKIP", "pass --image photo.jpg (a photo containing objects)"

    import cv2

    from database import AsyncLogWriter, create_database
    from throttle import DetectionThrottle

    db = create_database()
    if not db.available:
        return "FAIL", f"database unavailable: {db.last_error}"
    detector = _load_detector()
    image = cv2.imread(ARGS.image)
    if image is None:
        return "FAIL", f"could not read image {ARGS.image!r}"

    _, dets = detector.process_frame(image, 0.25, None)
    if not dets:
        return "FAIL", "no detections to log"
    before = db.count_logs()
    writer = AsyncLogWriter(db, flush_interval=0.1)
    to_log = DetectionThrottle(1.0).select(dets)
    for d in to_log:
        writer.submit(d["class_name"], d["confidence"], d["x"], d["y"], d["w"], d["h"])
    writer.flush(timeout=5)
    writer.stop()
    added = db.count_logs() - before
    if added != len(to_log):
        return "FAIL", f"expected {len(to_log)} new rows, found {added}"
    return "PASS", (f"{len(dets)} detections -> throttled to {len(to_log)} "
                    f"-> {added} rows in {db.label}")


def main() -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="photo containing objects (person, phone...)")
    parser.add_argument("--camera", type=int, help="webcam index to test, e.g. 0")
    parser.add_argument("--verbose", action="store_true", help="show tracebacks")
    ARGS = parser.parse_args()

    check_config()
    check_database()
    check_inference()
    check_video_file()
    check_webcam()
    check_end_to_end()

    print("\n" + "=" * 52)
    for title, status in RESULTS:
        print(f"{status:5} {title}")
    failed = [t for t, s in RESULTS if s == "FAIL"]
    print("=" * 52)
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
