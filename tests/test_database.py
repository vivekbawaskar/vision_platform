"""Tests for the SQLite backend, the async writer, and backend selection."""
import time
from datetime import datetime, timedelta

import pytest

import database
from database import AsyncLogWriter, SQLiteDatabaseManager, create_database


@pytest.fixture()
def db(tmp_path):
    return SQLiteDatabaseManager(path=str(tmp_path / "test.db"))


def test_schema_created_and_available(db):
    assert db.available
    assert db.count_logs() == 0
    assert list(db.fetch_recent_logs().columns) == database.LOG_COLUMNS


def test_log_and_fetch_roundtrip(db):
    assert db.log_detection("person", 0.91, 10, 20, 30, 40)
    row = db.fetch_recent_logs(5).iloc[0]
    assert row["object_class"] == "person"
    assert row["confidence"] == pytest.approx(0.91, abs=1e-6)
    assert (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]) == (
        10, 20, 30, 40,
    )


def test_newest_first_and_limit(db):
    for i in range(5):
        db.log_detection(f"c{i}", 0.5, 0, 0, 1, 1)
    frame = db.fetch_recent_logs(limit=3)
    assert list(frame["object_class"]) == ["c4", "c3", "c2"]


def test_sql_injection_is_inert(db):
    evil = "x'); DROP TABLE event_logs; --"
    assert db.log_detection(evil, 0.5, 0, 0, 1, 1)
    assert db.count_logs() == 1
    assert db.fetch_recent_logs().iloc[0]["object_class"] == evil[:50]


def test_batch_summary_and_timeline(db):
    now = datetime.utcnow()
    rows = [(now, "person", 0.9, 0, 0, 1, 1)] * 3 + [
        (now, "cell phone", 0.7, 0, 0, 1, 1)
    ]
    assert db.log_detections_batch(rows)
    summary = db.fetch_class_summary()
    assert dict(zip(summary["object_class"], summary["detections"])) == {
        "person": 3, "cell phone": 1,
    }
    timeline = db.fetch_timeline(minutes=5)
    assert int(timeline["person"].sum()) == 3


def test_timeline_ignores_old_rows(db):
    old = datetime.utcnow() - timedelta(hours=2)
    db.log_detections_batch([(old, "person", 0.9, 0, 0, 1, 1)])
    assert db.fetch_timeline(minutes=30).empty


def test_delete_class_and_clear(db):
    db.log_detection("a", 0.5, 0, 0, 1, 1)
    db.log_detection("b", 0.5, 0, 0, 1, 1)
    assert db.delete_class("a") and db.count_logs() == 1
    assert db.clear_logs() and db.count_logs() == 0


def test_async_writer_batches_everything(db):
    writer = AsyncLogWriter(db, flush_interval=0.05)
    for i in range(120):
        assert writer.submit("person", 0.8, i, i, 5, 5)
    assert writer.flush(timeout=5)
    assert db.count_logs() == 120
    assert writer.written == 120 and writer.dropped == 0
    writer.stop()


def test_writer_never_blocks_when_queue_full(db):
    writer = AsyncLogWriter(db, max_queue=1, batch_size=1, flush_interval=5)
    start = time.perf_counter()
    results = [writer.submit("p", 0.5, 0, 0, 1, 1) for _ in range(200)]
    assert time.perf_counter() - start < 1.0   # never blocks the caller
    assert not all(results) and writer.dropped > 0


def test_unavailable_db_degrades_gracefully(tmp_path):
    bad = SQLiteDatabaseManager(path=str(tmp_path / "missing" / "x" / "\0bad"))
    assert not bad.available
    assert bad.log_detection("p", 0.5, 0, 0, 1, 1) is False
    assert bad.fetch_recent_logs().empty
    writer = AsyncLogWriter(bad)
    assert writer.submit("p", 0.5, 0, 0, 1, 1) is False


def test_auto_backend_falls_back_to_sqlite(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(
        database, "CONFIG",
        config.AppConfig(db_backend="auto", sqlite_path=str(tmp_path / "f.db")),
    )

    class DownMySQL:
        available = False
        last_error = "connection refused"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(database, "DatabaseManager", DownMySQL)
    chosen = create_database()
    assert isinstance(chosen, SQLiteDatabaseManager)
    assert chosen.fallback_reason == "connection refused"


def test_sqlite_backend_selected_explicitly(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(
        database, "CONFIG",
        config.AppConfig(db_backend="sqlite", sqlite_path=str(tmp_path / "s.db")),
    )
    assert isinstance(create_database(), SQLiteDatabaseManager)
