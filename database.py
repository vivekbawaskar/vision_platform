"""Persistence layer.

Backends
--------
* :class:`DatabaseManager` - MySQL with connection pooling (the default).
* :class:`SQLiteDatabaseManager` - zero-setup file database, used for hosted
  demos or as an automatic fallback when MySQL is unreachable.

Both expose the same interface, so the rest of the app never cares which one
is active. Use :func:`create_database` to get the right one for the current
``DB_BACKEND`` setting (``mysql`` | ``sqlite`` | ``auto``).

:class:`AsyncLogWriter` batches inserts on a background thread so the video
loop never blocks on SQL commits. Every public method catches database
errors and degrades gracefully instead of crashing the pipeline.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from config import CONFIG, get_db_credentials

try:  # MySQL is optional so the SQLite backend works without the driver.
    import mysql.connector
    from mysql.connector import Error as MySQLError
    from mysql.connector import pooling
except ImportError:  # pragma: no cover - exercised only without the driver
    mysql = None  # type: ignore[assignment]
    pooling = None  # type: ignore[assignment]

    class MySQLError(Exception):  # type: ignore[no-redef]
        """Placeholder so ``except MySQLError`` stays valid."""


logger = logging.getLogger(__name__)

# Database identifiers are interpolated into DDL, so validate them.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")

LOG_COLUMNS = [
    "log_id", "timestamp", "object_class", "confidence",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]

MYSQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS event_logs (
    log_id       INT AUTO_INCREMENT PRIMARY KEY,
    timestamp    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence   FLOAT NOT NULL,
    bbox_x       FLOAT NOT NULL,
    bbox_y       FLOAT NOT NULL,
    bbox_w       FLOAT NOT NULL,
    bbox_h       FLOAT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_logs (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    object_class TEXT NOT NULL,
    confidence   REAL NOT NULL,
    bbox_x       REAL NOT NULL,
    bbox_y       REAL NOT NULL,
    bbox_w       REAL NOT NULL,
    bbox_h       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON event_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_object_class ON event_logs (object_class);
"""

# (timestamp, class, confidence, x, y, w, h)
LogRow = Tuple[datetime, str, float, float, float, float, float]


def _utcnow() -> datetime:
    """Naive UTC timestamp (the schema stores UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BaseDatabase:
    """Behaviour shared by both backends (SQL differs only in placeholders)."""

    placeholder = "%s"
    clear_sql = "TRUNCATE TABLE event_logs"

    def __init__(self) -> None:
        self.available = False
        self.last_error: Optional[str] = None
        self.label = "database"
        self.fallback_reason: Optional[str] = None

    # ---- hooks implemented by the concrete backends
    def connect(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _execute(self, sql: str, params=(), many: bool = False) -> bool:
        raise NotImplementedError  # pragma: no cover

    def _fetch_all(self, query: str, params: Tuple = ()) -> List[dict]:
        raise NotImplementedError  # pragma: no cover

    def _ts(self, value: datetime):
        """Convert a datetime into the backend's parameter format."""
        return value

    # ---- writes
    def log_detection(
        self,
        object_class: str,
        confidence: float,
        bbox_x: float,
        bbox_y: float,
        bbox_w: float,
        bbox_h: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Insert one detection using a parameterised query."""
        return self.log_detections_batch(
            [(
                timestamp or _utcnow(), str(object_class)[:50],
                float(confidence), float(bbox_x), float(bbox_y),
                float(bbox_w), float(bbox_h),
            )]
        )

    def log_detections_batch(self, rows: Sequence[LogRow]) -> bool:
        """Insert many detections in one transaction."""
        if not rows:
            return False
        ph = self.placeholder
        sql = (
            "INSERT INTO event_logs (timestamp, object_class, confidence, "
            "bbox_x, bbox_y, bbox_w, bbox_h) "
            f"VALUES ({', '.join([ph] * 7)})"
        )
        params = [(self._ts(r[0]),) + tuple(r[1:]) for r in rows]
        return self._execute(sql, params, many=True)

    def clear_logs(self) -> bool:
        """Remove every row from ``event_logs``."""
        return self._execute(self.clear_sql)

    def delete_class(self, object_class: str) -> bool:
        """Delete all rows of one class (used by the self-check script)."""
        return self._execute(
            f"DELETE FROM event_logs WHERE object_class = {self.placeholder}",
            (object_class,),
        )

    # ---- reads
    def fetch_recent_logs(self, limit: int = 15) -> pd.DataFrame:
        """Newest detections first, as a DataFrame."""
        rows = self._fetch_all(
            "SELECT log_id, timestamp, object_class, confidence, "
            "bbox_x, bbox_y, bbox_w, bbox_h FROM event_logs "
            f"ORDER BY log_id DESC LIMIT {self.placeholder}",
            (int(limit),),
        )
        frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def count_logs(self) -> int:
        """Total number of stored events."""
        rows = self._fetch_all("SELECT COUNT(*) AS n FROM event_logs")
        return int(rows[0]["n"]) if rows else 0

    def fetch_class_summary(self) -> pd.DataFrame:
        """Detections and average confidence per class."""
        rows = self._fetch_all(
            "SELECT object_class, COUNT(*) AS detections, "
            "ROUND(AVG(confidence), 3) AS avg_confidence "
            "FROM event_logs GROUP BY object_class ORDER BY detections DESC"
        )
        return pd.DataFrame(
            rows, columns=["object_class", "detections", "avg_confidence"]
        )

    def fetch_timeline(self, minutes: int = 30) -> pd.DataFrame:
        """Detections per 10-second bucket, one column per class."""
        cutoff = _utcnow() - timedelta(minutes=int(minutes))
        rows = self._fetch_all(
            "SELECT timestamp, object_class FROM event_logs "
            f"WHERE timestamp >= {self.placeholder} "
            "ORDER BY timestamp DESC LIMIT 20000",
            (self._ts(cutoff),),
        )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows, columns=["timestamp", "object_class"])
        frame["bucket"] = pd.to_datetime(frame["timestamp"]).dt.floor("10s")
        return (
            frame.groupby(["bucket", "object_class"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )


# ============================================================ MySQL backend
class DatabaseManager(BaseDatabase):
    """Connection-pooled MySQL backend."""

    def __init__(
        self,
        credentials: Optional[Dict[str, object]] = None,
        pool_size: Optional[int] = None,
        auto_connect: bool = True,
    ) -> None:
        super().__init__()
        self._credentials = dict(credentials or get_db_credentials())
        self._pool_size = pool_size or CONFIG.db_pool_size
        self._pool = None
        self.label = f"MySQL · {self._credentials['database']}"
        if auto_connect:
            self.connect()

    def _connect_args(self) -> Dict[str, object]:
        args: Dict[str, object] = {
            "host": self._credentials["host"],
            "user": self._credentials["user"],
            "password": self._credentials["password"],
            "port": self._credentials["port"],
            "connection_timeout": 5,
        }
        if CONFIG.db_ssl_ca:  # managed MySQL (Aiven, TiDB, ...) needs TLS
            args["ssl_ca"] = CONFIG.db_ssl_ca
        return args

    def connect(self) -> bool:
        """Create the database/table if needed and build the pool.

        Safe to call repeatedly (used by the "Retry connection" button).
        """
        if mysql is None:
            self.last_error = "mysql-connector-python is not installed"
            self.available = False
            return False

        db_name = str(self._credentials["database"])
        if not _IDENTIFIER.match(db_name):
            self.last_error = f"Invalid database name: {db_name!r}"
            self.available = False
            return False

        bootstrap = None
        try:
            bootstrap = mysql.connector.connect(**self._connect_args())
            cursor = bootstrap.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4"
            )
            cursor.execute(f"USE `{db_name}`")
            cursor.execute(MYSQL_CREATE_TABLE)
            bootstrap.commit()
            cursor.close()

            self._pool = pooling.MySQLConnectionPool(
                pool_name="vision_pool",
                pool_size=self._pool_size,
                pool_reset_session=True,
                database=db_name,
                **self._connect_args(),
            )
            self.available = True
            self.last_error = None
            logger.info("MySQL ready (database=%s)", db_name)
        except MySQLError as exc:
            self.available = False
            self._pool = None
            self.last_error = str(exc)
            logger.error("MySQL connection failed: %s", exc)
        finally:
            if bootstrap is not None and bootstrap.is_connected():
                bootstrap.close()
        return self.available

    @contextmanager
    def _connection(self) -> Iterator["mysql.connector.MySQLConnection"]:
        """Borrow a pooled connection and always hand it back."""
        if self._pool is None:
            raise MySQLError("Database pool is not initialised")
        conn = self._pool.get_connection()
        try:
            yield conn
        finally:
            conn.close()  # returns it to the pool

    def _execute(self, sql: str, params=(), many: bool = False) -> bool:
        if not self.available:
            return False
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                try:
                    if many:
                        cursor.executemany(sql, list(params))
                    else:
                        cursor.execute(sql, params)
                    conn.commit()
                    return True
                except MySQLError:
                    conn.rollback()
                    raise
                finally:
                    cursor.close()
        except MySQLError as exc:
            self.last_error = str(exc)
            logger.error("MySQL write failed: %s", exc)
            return False

    def _fetch_all(self, query: str, params: Tuple = ()) -> List[dict]:
        if not self.available:
            return []
        try:
            with self._connection() as conn:
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(query, params)
                    return cursor.fetchall()
                finally:
                    cursor.close()
        except MySQLError as exc:
            self.last_error = str(exc)
            logger.error("MySQL query failed: %s", exc)
            return []


# =========================================================== SQLite backend
class SQLiteDatabaseManager(BaseDatabase):
    """File-based backend: no server, no credentials."""

    placeholder = "?"
    clear_sql = "DELETE FROM event_logs"

    def __init__(
        self,
        path: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = path or CONFIG.sqlite_path
        self.fallback_reason = fallback_reason
        self.label = f"SQLite · {os.path.basename(self.path)}"
        self.connect()

    def _ts(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def connect(self) -> bool:
        """Create the file and schema if needed."""
        try:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            with self._connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SQLITE_SCHEMA)
                conn.commit()
            self.available = True
            self.last_error = None
        except (sqlite3.Error, OSError, ValueError) as exc:
            self.available = False
            self.last_error = str(exc)
            logger.error("SQLite setup failed: %s", exc)
        return self.available

    def _execute(self, sql: str, params=(), many: bool = False) -> bool:
        if not self.available:
            return False
        try:
            with self._connection() as conn:
                if many:
                    conn.executemany(sql, list(params))
                else:
                    conn.execute(sql, params)
                conn.commit()
            return True
        except sqlite3.Error as exc:
            self.last_error = str(exc)
            logger.error("SQLite write failed: %s", exc)
            return False

    def _fetch_all(self, query: str, params: Tuple = ()) -> List[dict]:
        if not self.available:
            return []
        try:
            with self._connection() as conn:
                return [dict(row) for row in conn.execute(query, params)]
        except sqlite3.Error as exc:
            self.last_error = str(exc)
            logger.error("SQLite query failed: %s", exc)
            return []


def create_database() -> BaseDatabase:
    """Pick a backend from ``DB_BACKEND``: mysql (default), sqlite or auto."""
    backend = CONFIG.db_backend
    if backend == "sqlite":
        return SQLiteDatabaseManager()

    mysql_db = DatabaseManager()
    if backend == "auto" and not mysql_db.available:
        logger.warning("MySQL unavailable, falling back to SQLite")
        return SQLiteDatabaseManager(fallback_reason=mysql_db.last_error)
    return mysql_db


class AsyncLogWriter:
    """Background batch writer that keeps SQL I/O off the video thread."""

    def __init__(
        self,
        db: BaseDatabase,
        max_queue: Optional[int] = None,
        batch_size: int = 50,
        flush_interval: float = 0.5,
    ) -> None:
        self._db = db
        self._queue: "queue.Queue[LogRow]" = queue.Queue(
            maxsize=max_queue or CONFIG.writer_queue_size
        )
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._stop = threading.Event()
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="db-log-writer", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        object_class: str,
        confidence: float,
        bbox_x: float,
        bbox_y: float,
        bbox_w: float,
        bbox_h: float,
    ) -> bool:
        """Queue one detection; never blocks. Returns False if dropped."""
        if not self._db.available:
            self.dropped += 1
            return False
        row: LogRow = (
            _utcnow(), str(object_class)[:50], float(confidence),
            float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h),
        )
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def _run(self) -> None:
        """Drain the queue in batches until stopped."""
        while not self._stop.is_set() or not self._queue.empty():
            batch: List[LogRow] = []
            deadline = time.monotonic() + self._flush_interval
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            if batch:
                if self._db.log_detections_batch(batch):
                    self.written += len(batch)
                else:
                    self.dropped += len(batch)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue is empty and written (tests / shutdown)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._queue.empty() and (
                self.written + self.dropped >= self.submitted
            ):
                return True
            time.sleep(0.05)
        return False

    def stop(self, timeout: float = 3.0) -> None:
        """Flush remaining rows and stop the worker thread."""
        self._stop.set()
        self._thread.join(timeout=timeout)
