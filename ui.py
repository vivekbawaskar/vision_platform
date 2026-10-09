"""Streamlit dashboard components.

Layout is built once per script run; the main loop then updates the returned
placeholders in place (video frame, KPI tiles, log table, analytics charts).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from PIL import Image, ImageDraw, ImageFont

from config import CONFIG, DEFAULT_CONF
from detector import COCO_CLASSES

SOURCE_OPTIONS = [
    "Browser Webcam (Live)",
    "Sample Video",
    "Upload Video",
    "Webcam (Local USB)",
    "RTSP Stream",
]

_CSS = """
<style>
div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 10px;
    padding: 10px 14px;
}
.block-container { padding-top: 1.5rem; }
</style>
"""


@dataclass
class SidebarSettings:
    """Everything the user can control from the sidebar."""

    confidence: float
    classes: List[str]
    source_type: str
    source: Optional[Union[int, str]]
    enable_tracking: bool
    log_interval: float
    db_logging: bool
    start_clicked: bool
    stop_clicked: bool


@dataclass
class DashboardElements:
    """Placeholders the pipeline updates while running."""

    status: "st.delta_generator.DeltaGenerator"
    caption: "st.delta_generator.DeltaGenerator"
    video: "st.delta_generator.DeltaGenerator"
    kpis: Dict[str, "st.delta_generator.DeltaGenerator"]
    logs: "st.delta_generator.DeltaGenerator"
    summary: "st.delta_generator.DeltaGenerator"
    bar_chart: "st.delta_generator.DeltaGenerator"
    timeline: "st.delta_generator.DeltaGenerator"


def _stretch(widget, *args, **kwargs):
    """Make a widget fill its container on old *and* new Streamlit versions.

    Recent releases replaced ``use_container_width=True`` with
    ``width="stretch"``; older ones only know the former.
    """
    try:
        return widget(*args, width="stretch", **kwargs)
    except (TypeError, ValueError, StreamlitAPIException):
        return widget(*args, use_container_width=True, **kwargs)


# ------------------------------------------------------------------ page setup
def configure_page() -> None:
    """Must be the first Streamlit call of the script."""
    st.set_page_config(
        page_title="Vision Platform | Real-Time Object Detection",
        page_icon="🎯",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- sidebar
def _save_upload(uploaded) -> str:
    """Persist an uploaded video to a temp file once and return its path."""
    folder = os.path.join(tempfile.gettempdir(), "vision_platform_uploads")
    os.makedirs(folder, exist_ok=True)
    safe_name = os.path.basename(uploaded.name)
    path = os.path.join(folder, f"{uploaded.size}_{safe_name}")
    if not os.path.exists(path):
        with open(path, "wb") as handle:
            handle.write(uploaded.getbuffer())
    return path


def _resolve_source(source_type: str) -> Optional[Union[int, str]]:
    """Render the source-specific widget and return the capture argument."""
    sb = st.sidebar
    if source_type == "Browser Webcam (Live)":
        sb.caption("Uses your device's camera directly in the browser via WebRTC.")
        return "browser_webrtc"

    if source_type in ("Webcam", "Webcam (Local USB)"):
        return int(sb.number_input("Camera index", 0, 10, 0, 1))

    if source_type == "Sample Video":
        path = CONFIG.sample_video_path
        if os.path.isfile(path):
            sb.caption(f"Using `{path}` (looping)")
            return path
        sb.warning(f"`{path}` not found. Put an .mp4 next to main.py.")
        return None

    if source_type == "Upload Video":
        uploaded = sb.file_uploader(
            "Video file", type=["mp4", "avi", "mov", "mkv"]
        )
        return _save_upload(uploaded) if uploaded is not None else None

    url = sb.text_input(
        "RTSP URL", placeholder="rtsp://user:pass@192.168.1.10:554/stream"
    ).strip()
    return url or None


def render_sidebar() -> SidebarSettings:
    """Draw the control panel and return the selected settings."""
    sb = st.sidebar
    sb.title("🎯 Vision Platform")

    col_start, col_stop = sb.columns(2)
    start = _stretch(col_start.button, "▶ Start", type="primary")
    stop = _stretch(col_stop.button, "■ Stop")

    sb.divider()
    sb.subheader("Detection")
    confidence = sb.slider(
        "Confidence threshold", 0.10, 1.00, float(DEFAULT_CONF), 0.05
    )
    classes = sb.multiselect(
        "Target classes",
        COCO_CLASSES,
        default=[],
        help="Leave empty to detect and log every class.",
    )
    tracking = sb.toggle(
        "Object tracking (unique counts)",
        value=False,
        help="Adds ByteTrack IDs to boxes and counts unique objects.",
    )

    sb.subheader("Input source")
    default_index = (
        SOURCE_OPTIONS.index(CONFIG.default_source)
        if CONFIG.default_source in SOURCE_OPTIONS
        else 0
    )
    source_type = sb.selectbox("Source", SOURCE_OPTIONS, index=default_index)
    source = _resolve_source(source_type)

    sb.subheader("Logging")
    db_logging = sb.toggle("Log detections to MySQL", value=True)
    interval = sb.slider(
        "Min seconds between logs (per class)",
        0.2, 10.0, float(max(0.2, min(CONFIG.log_interval, 10.0))), 0.2,
    )

    return SidebarSettings(
        confidence=float(confidence),
        classes=list(classes),
        source_type=source_type,
        source=source,
        enable_tracking=bool(tracking),
        log_interval=float(interval),
        db_logging=bool(db_logging),
        start_clicked=start,
        stop_clicked=stop,
    )


def render_db_status(db) -> bool:
    """Show database health; returns True if "Retry connection" was clicked."""
    sb = st.sidebar
    sb.divider()
    sb.subheader("Database")
    if db.available:
        sb.success(f"Connected · {db.label}")
        if db.fallback_reason:
            sb.caption(
                f"MySQL unavailable ({db.fallback_reason}) - "
                "using the SQLite fallback."
            )
        return False
    sb.error("Database unavailable - detection runs, but nothing is stored.")
    if db.last_error:
        sb.caption(db.last_error)
    return _stretch(sb.button, "Retry connection")


def render_log_tools(db) -> bool:
    """CSV export and confirmed log wipe; returns True if logs were cleared."""
    sb = st.sidebar
    if not db.available:
        return False
    sb.subheader("Log tools")
    export = db.fetch_recent_logs(limit=5000)
    _stretch(
        sb.download_button,
        "⬇ Export logs (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="event_logs.csv",
        mime="text/csv",
        disabled=export.empty,
    )
    confirm = sb.checkbox("Confirm clear all logs")
    if _stretch(sb.button, "🗑 Clear logs", disabled=not confirm):
        return db.clear_logs()
    return False


# ------------------------------------------------------------------- dashboard
def render_dashboard() -> DashboardElements:
    """Create the page layout and return placeholders for live updates."""
    st.title("Real-Time Object Detection & Logging")
    live_tab, analytics_tab = st.tabs(["📹 Live Monitor", "📊 Analytics"])

    with live_tab:
        status = st.empty()
        columns = st.columns(5)
        kpis = {
            key: column.empty()
            for key, column in zip(
                ("fps", "latency", "in_frame", "unique", "logged"), columns
            )
        }
        caption = st.empty()
        video = st.empty()
        st.subheader("Recent detection events")
        logs = st.empty()

    with analytics_tab:
        left, right = st.columns(2)
        summary = left.empty()
        bar_chart = right.empty()
        st.markdown("**Detections over time** (per 10 s, last 30 min)")
        timeline = st.empty()

    return DashboardElements(
        status=status,
        caption=caption,
        video=video,
        kpis=kpis,
        logs=logs,
        summary=summary,
        bar_chart=bar_chart,
        timeline=timeline,
    )


def update_kpis(
    elements: DashboardElements,
    fps: float,
    latency_ms: float,
    in_frame: int,
    unique: Optional[int],
    logged: int,
) -> None:
    """Refresh the five KPI tiles."""
    elements.kpis["fps"].metric("FPS", f"{fps:.1f}")
    elements.kpis["latency"].metric("Inference", f"{latency_ms:.0f} ms")
    elements.kpis["in_frame"].metric("Objects in frame", in_frame)
    elements.kpis["unique"].metric(
        "Unique tracked", "off" if unique is None else unique
    )
    elements.kpis["logged"].metric("Logged this session", logged)


def render_filter_caption(
    elements: DashboardElements, settings: SidebarSettings
) -> None:
    """Show the active filters under the KPI row."""
    classes = ", ".join(settings.classes) if settings.classes else "all classes"
    elements.caption.caption(
        f"Confidence ≥ {settings.confidence:.2f} · Classes: {classes} · "
        f"Tracking: {'on' if settings.enable_tracking else 'off'} · "
        f"DB logging: {'on' if settings.db_logging else 'off'}"
    )


def _idle_frame(message: str) -> Image.Image:
    """Dark placeholder image shown while no stream is running."""
    image = Image.new("RGB", (960, 540), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # older Pillow without a size argument
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), message, font=font)
    draw.text(
        ((960 - (right - left)) / 2, (540 - (bottom - top)) / 2),
        message,
        fill=(170, 175, 190),
        font=font,
    )
    return image


def show_idle(elements: DashboardElements, message: str) -> None:
    """Paint the placeholder frame into the video area."""
    _stretch(elements.video.image, _idle_frame(message))


def display_video_stream(
    elements: DashboardElements, frame_rgb: np.ndarray
) -> None:
    """Paint one annotated RGB frame into the video area."""
    _stretch(
        elements.video.image,
        frame_rgb,
        channels="RGB",
        output_format="JPEG",
    )


def render_logs(elements: DashboardElements, logs: pd.DataFrame) -> None:
    """Show the newest events in the log table."""
    if logs.empty:
        elements.logs.info("No events logged yet.")
        return
    table = logs.round(
        {"confidence": 3, "bbox_x": 1, "bbox_y": 1, "bbox_w": 1, "bbox_h": 1}
    )
    _stretch(elements.logs.dataframe, table, hide_index=True)


def render_analytics(
    elements: DashboardElements,
    summary: pd.DataFrame,
    timeline: pd.DataFrame,
    total_rows: int,
) -> None:
    """Refresh the analytics tab from database aggregates."""
    with elements.summary.container():
        st.markdown(f"**Detections by class** · {total_rows:,} rows total")
        if summary.empty:
            st.info("No data yet.")
        else:
            _stretch(st.dataframe, summary, hide_index=True)

    with elements.bar_chart.container():
        st.markdown("**Class distribution**")
        if not summary.empty:
            st.bar_chart(summary.set_index("object_class")["detections"])

    with elements.timeline.container():
        if timeline.empty:
            st.info("No events in the last 30 minutes.")
        else:
            st.line_chart(timeline)
