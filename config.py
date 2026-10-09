"""Application configuration.

Loads settings from environment variables (optionally populated from a local
``.env`` file via python-dotenv) and exposes them as an immutable
:class:`AppConfig`. Secrets such as the database password are never
hard-coded; the fallback defaults below are for local development only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Union

from dotenv import load_dotenv

# Read the .env file (if present) into the process environment.
load_dotenv()

# --- Model configuration constants -----------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.5


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    """Typed, read-only view of every runtime setting."""

    # Database
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = "password"
    db_name: str = "vision_platform"
    db_port: int = 3306
    db_pool_size: int = 5

    # Model / pipeline
    model_path: str = DEFAULT_MODEL
    default_conf: float = DEFAULT_CONF
    log_interval: float = 1.0          # min seconds between logs per class
    sample_video_path: str = "sample.mp4"
    max_display_width: int = 960       # frames are downscaled for the browser
    writer_queue_size: int = 2000      # async DB writer buffer
    default_source: str = "Webcam"     # preselected input in the sidebar

    # Backend selection: "mysql", "sqlite", or "auto" (default: MySQL with
    # automatic SQLite fallback - handy for hosted demos).
    db_backend: str = "auto"
    sqlite_path: str = "vision_platform.db"
    db_ssl_ca: str = ""                # CA file for managed MySQL (TLS)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build the configuration from the current environment."""
        return cls(
            db_host=os.getenv("DB_HOST", "localhost"),
            db_user=os.getenv("DB_USER", "root"),
            db_password=os.getenv("DB_PASSWORD", "password"),
            db_name=os.getenv("DB_NAME", "vision_platform"),
            db_port=_get_int("DB_PORT", 3306),
            db_pool_size=max(1, min(_get_int("DB_POOL_SIZE", 5), 32)),
            model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL),
            default_conf=_get_float("DEFAULT_CONF", DEFAULT_CONF),
            log_interval=_get_float("LOG_INTERVAL_SECONDS", 1.0),
            sample_video_path=os.getenv("SAMPLE_VIDEO_PATH", "sample.mp4"),
            max_display_width=_get_int("MAX_DISPLAY_WIDTH", 960),
            writer_queue_size=_get_int("WRITER_QUEUE_SIZE", 2000),
            default_source=os.getenv("DEFAULT_SOURCE", "Webcam"),
            db_backend=os.getenv("DB_BACKEND", "auto").strip().lower(),
            sqlite_path=os.getenv("SQLITE_PATH", "vision_platform.db"),
            db_ssl_ca=os.getenv("DB_SSL_CA", ""),
        )


CONFIG = AppConfig.from_env()


def get_db_credentials() -> Dict[str, Union[str, int]]:
    """Return connection keyword arguments for ``mysql.connector``."""
    return {
        "host": CONFIG.db_host,
        "user": CONFIG.db_user,
        "password": CONFIG.db_password,
        "port": CONFIG.db_port,
        "database": CONFIG.db_name,
    }
