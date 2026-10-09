"""Entry point: ``streamlit run main.py``.

Pipeline per frame:
    VideoStream -> YOLOObjectDetector -> DetectionThrottle -> AsyncLogWriter
                         |
                         +-> ui.display_video_stream

Streamlit re-executes this script on every widget interaction. The video
stream and throttle state therefore live in ``st.session_state`` so a slider
tweak does not re-open the camera or re-log every class.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st

import ui
from config import CONFIG
from database import AsyncLogWriter, BaseDatabase, create_database
from detector import YOLOObjectDetector
from throttle import DetectionThrottle
from video_source import VideoStream

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import av
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
    WEBRTC_AVAILABLE = True
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
except ImportError:
    WEBRTC_AVAILABLE = False
    RTC_CONFIGURATION = None

KPI_REFRESH_S = 0.5
LOG_REFRESH_S = 1.0
ANALYTICS_REFRESH_S = 5.0


# --------------------------------------------------------- cached singletons
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_detector(model_path: str) -> YOLOObjectDetector:
    """Load the model once per server process."""
    return YOLOObjectDetector(model_path)


@st.cache_resource(show_spinner="Connecting to MySQL…")
def load_database() -> BaseDatabase:
    """Create the DB backend (and schema) once per server process."""
    return create_database()


@st.cache_resource
def load_writer(_db: BaseDatabase) -> AsyncLogWriter:
    """Start the background batch writer once per server process."""
    return AsyncLogWriter(_db)


# ------------------------------------------------------------ session state
def _init_state() -> None:
    state = st.session_state
    state.setdefault("running", False)
    state.setdefault("throttle", DetectionThrottle(CONFIG.log_interval))
    state.setdefault("unique_ids", set())
    state.setdefault("logged_total", 0)


def _release_stream() -> None:
    stream: Optional[VideoStream] = st.session_state.pop("stream", None)
    st.session_state.pop("stream_key", None)
    if stream is not None:
        stream.release()


def _acquire_stream(
    settings: ui.SidebarSettings, detector: YOLOObjectDetector
) -> Optional[VideoStream]:
    """Reuse the open stream if the source is unchanged, else open a new one."""
    key = (settings.source_type, settings.source)
    stream: Optional[VideoStream] = st.session_state.get("stream")
    if (
        stream is not None
        and st.session_state.get("stream_key") == key
        and stream.is_alive
    ):
        return stream

    _release_stream()
    if settings.source is None:
        st.error("Choose a valid input source first.")
        return None

    stream = VideoStream(
        settings.source, loop=(settings.source_type == "Sample Video")
    )
    if not stream.start():
        st.error(stream.error or "Could not open the video source.")
        return None

    st.session_state["stream"] = stream
    st.session_state["stream_key"] = key
    st.session_state["unique_ids"] = set()
    st.session_state["throttle"].reset()
    detector.reset_tracking()
    return stream


def _fit_width(frame_rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale for the browser; detections keep original pixel coords."""
    height, width = frame_rgb.shape[:2]
    if width <= max_width:
        return frame_rgb
    scale = max_width / width
    return cv2.resize(
        frame_rgb, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA
    )


def _refresh_data(
    db: BaseDatabase, elements: ui.DashboardElements, analytics: bool
) -> None:
    """Pull logs (and optionally analytics) from MySQL into the UI."""
    ui.render_logs(elements, db.fetch_recent_logs(limit=15))
    if analytics:
        ui.render_analytics(
            elements,
            db.fetch_class_summary(),
            db.fetch_timeline(minutes=30),
            db.count_logs(),
        )


# ------------------------------------------------------------------ pipeline
def run_pipeline(
    settings: ui.SidebarSettings,
    detector: YOLOObjectDetector,
    db: BaseDatabase,
    writer: AsyncLogWriter,
    elements: ui.DashboardElements,
) -> None:
    """Capture -> detect -> throttle -> log -> render, until stopped."""
    stream = _acquire_stream(settings, detector)
    if stream is None:
        st.session_state.running = False
        ui.show_idle(elements, "Could not start - check the input source")
        _refresh_data(db, elements, analytics=True)
        return

    throttle: DetectionThrottle = st.session_state.throttle
    throttle.interval = settings.log_interval
    allowed = settings.classes or None
    log_to_db = settings.db_logging and db.available

    fps = 0.0
    latency_ms = 0.0
    in_frame = 0
    empty_reads = 0
    status_shown = False
    last_kpi = last_logs = last_analytics = 0.0
    previous = time.perf_counter()

    _refresh_data(db, elements, analytics=True)

    while st.session_state.running:
        ok, frame = stream.read()
        if not ok or frame is None:
            if not stream.is_alive:
                break
            empty_reads += 1
            if empty_reads > 30:
                elements.status.warning("Waiting for the video source…")
                status_shown = True
            continue
        empty_reads = 0
        if status_shown:
            elements.status.empty()
            status_shown = False

        started = time.perf_counter()
        try:
            frame_rgb, detections = detector.process_frame(
                frame,
                conf_threshold=settings.confidence,
                allowed_classes=allowed,
                track=settings.enable_tracking,
            )
        except Exception as exc:  # keep the UI alive on a bad frame
            logger.exception("Inference failed: %s", exc)
            elements.status.error(f"Inference error: {exc}")
            status_shown = True
            continue
        latency_ms = (time.perf_counter() - started) * 1000.0
        in_frame = len(detections)

        # Unique-object counting (tracking mode only).
        for det in detections:
            if det["track_id"] is not None:
                st.session_state.unique_ids.add(
                    (det["class_name"], det["track_id"])
                )

        # Throttled, non-blocking persistence.
        if log_to_db:
            for det in throttle.select(detections):
                if writer.submit(
                    str(det["class_name"]),
                    float(det["confidence"]),
                    float(det["x"]),
                    float(det["y"]),
                    float(det["w"]),
                    float(det["h"]),
                ):
                    st.session_state.logged_total += 1

        ui.display_video_stream(
            elements, _fit_width(frame_rgb, CONFIG.max_display_width)
        )

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - previous, 1e-6)
        fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
        previous = now

        if now - last_kpi >= KPI_REFRESH_S:
            ui.update_kpis(
                elements,
                fps,
                latency_ms,
                in_frame,
                len(st.session_state.unique_ids)
                if settings.enable_tracking
                else None,
                st.session_state.logged_total,
            )
            last_kpi = now
        if db.available and now - last_logs >= LOG_REFRESH_S:
            analytics_due = now - last_analytics >= ANALYTICS_REFRESH_S
            _refresh_data(db, elements, analytics=analytics_due)
            last_logs = now
            if analytics_due:
                last_analytics = now

    if not stream.is_alive:
        if stream.error:
            elements.status.error(stream.error)
        else:
            elements.status.info("Video finished.")
        st.session_state.running = False
        _release_stream()
        ui.show_idle(elements, "Stream ended - press Start to run again")


# ---------------------------------------------------------------------- main
def main() -> None:
    """Build the page, then run the live pipeline if it is switched on."""
    ui.configure_page()
    _init_state()

    try:
        detector = load_detector(CONFIG.model_path)
    except Exception as exc:
        st.error(f"Could not load the YOLO model '{CONFIG.model_path}': {exc}")
        st.stop()

    db = load_database()
    writer = load_writer(db)

    settings = ui.render_sidebar()
    if ui.render_db_status(db):
        db.connect()
        st.rerun()
    if ui.render_log_tools(db):
        st.session_state.logged_total = 0
        st.toast("Logs cleared")

    elements = ui.render_dashboard()
    ui.render_filter_caption(elements, settings)
    ui.update_kpis(elements, 0.0, 0.0, 0, None, st.session_state.logged_total)

    if settings.start_clicked:
        st.session_state.running = True
    if settings.stop_clicked:
        st.session_state.running = False
        _release_stream()

    if settings.source_type == "Browser Webcam (Live)":
        if not WEBRTC_AVAILABLE:
            elements.status.error("streamlit-webrtc is not installed on this server.")
            ui.show_idle(elements, "Browser Webcam requires streamlit-webrtc")
            _refresh_data(db, elements, analytics=True)
            return

        throttle = st.session_state.throttle
        throttle.interval = settings.log_interval
        allowed = settings.classes or None
        log_to_db = settings.db_logging and db.available

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            try:
                frame_rgb, detections = detector.process_frame(
                    img,
                    conf_threshold=settings.confidence,
                    allowed_classes=allowed,
                    track=settings.enable_tracking,
                )
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                return frame

            if log_to_db:
                for det in throttle.select(detections):
                    if writer.submit(
                        str(det["class_name"]),
                        float(det["confidence"]),
                        float(det["x"]),
                        float(det["y"]),
                        float(det["w"]),
                        float(det["h"]),
                    ):
                        st.session_state.logged_total += 1

            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), format="bgr24"
            )

        with elements.video.container():
            st.info("📹 Click **START** below to allow browser camera access for live real-time YOLO detection:")
            webrtc_streamer(
                key="browser_webcam_yolo",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        _refresh_data(db, elements, analytics=True)
    elif st.session_state.running:
        run_pipeline(settings, detector, db, writer, elements)
    else:
        ui.show_idle(elements, "Press ▶ Start to begin detection")
        _refresh_data(db, elements, analytics=True)


main()
