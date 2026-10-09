# 🎯 Real-Time Object Detection & Logging Platform

Live object detection with **YOLOv8**, event persistence in **MySQL**, and an
interactive **Streamlit** dashboard — built as small, decoupled modules.

## Features

**Core**
- YOLOv8 Nano inference (lazy `stream=True` generator, no memory build-up)
- Webcam, RTSP stream, bundled sample video, or uploaded video file
- Confidence slider (0.10 – 1.00) and multi-select class filter
- Parameterised MySQL inserts; database + table auto-created on first run
- Per-class logging throttle (immediate on entry, then at most once per interval)
- Live KPIs (FPS, inference latency, objects in frame) and a live event table
- Credentials via `.env` only; `.env` and weights are git-ignored

**Enhancements beyond the base spec**
- **Non-blocking DB writes** – a background thread batches inserts, so SQL never stalls the video loop
- **Connection pooling** and a graceful "MySQL unavailable" mode (detection keeps running; one-click retry)
- **Threaded capture** that always serves the *latest* frame (no latency build-up) with **auto-reconnect** for webcam/RTSP drops
- **Start / Stop controls**; the stream and throttle persist across slider changes (no camera re-open, no duplicate log bursts)
- **Object tracking toggle** (ByteTrack) with track IDs on boxes and a *unique objects* counter
- **MySQL *or* SQLite backend** (`DB_BACKEND=mysql|sqlite|auto`) - hosted demos need no database server
- **Analytics tab** – detections by class, class distribution chart, timeline (10 s buckets)
- **CSV export** and confirmed **clear logs** from the sidebar
- Model warm-up, colour-coded boxes with label chips, display downscaling for smoother browser streaming
- `check_setup.py` self-check for every verification phase, a unit-test suite, GitHub Actions CI, and a `Dockerfile` for hosting

## Architecture

```
VideoStream ──► YOLOObjectDetector ──► DetectionThrottle ──► AsyncLogWriter ──► MySQL
 (video_source)     (detector)            (throttle)             (database)
        │                 └──► annotated RGB frame ──► ui.py (Streamlit) ◄── DatabaseManager.fetch_*
```

| File | Responsibility |
|---|---|
| `main.py` | Streamlit entry point, pipeline loop, session state |
| `detector.py` | `YOLOObjectDetector`: inference, filtering, drawing, tracking |
| `video_source.py` | `VideoStream`: threaded capture, reconnect, file pacing |
| `throttle.py` | `DetectionThrottle`: per-class rate limiting |
| `database.py` | `DatabaseManager` (MySQL), `SQLiteDatabaseManager`, `create_database()`, `AsyncLogWriter` |
| `check_setup.py` | Self-check: config, DB, model, video, end-to-end |
| `ui.py` | Sidebar, dashboard layout, log table, analytics |
| `config.py` | `.env` loading and defaults |
| `database_schema.sql` | Manual schema script (optional – auto-created otherwise) |

## Setup

### 1. Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

### 2. MySQL – choose one

**XAMPP**
1. Open the XAMPP Control Panel and **Start** the *MySQL* module.
2. The default user is `root` with an **empty** password, so set `DB_PASSWORD=` (empty) in `.env`.
3. Optionally check it in phpMyAdmin (`http://localhost/phpmyadmin`).

**MySQL Workbench / MySQL Server**
1. Create a connection to `localhost:3306` with your root credentials.
2. Put those credentials in `.env`.
3. After the first run, open Workbench and run
   `SELECT * FROM vision_platform.event_logs ORDER BY log_id DESC;`
   to watch rows arrive live.

**Docker (optional)**
```bash
docker compose up -d
```

The database and `event_logs` table are created automatically. You can also
run `database_schema.sql` yourself.

### 3. Environment file
```bash
cp .env.example .env     # Windows: copy .env.example .env
```
Edit `.env` with your real credentials.

### 4. (Optional) sample video
Place any short `.mp4` next to `main.py` and name it `sample.mp4`.

## Run
```bash
streamlit run main.py
```
Pick a source in the sidebar and press **▶ Start**.

## Self-check (run this first)
```bash
python check_setup.py                    # config, database, model, video file
python check_setup.py --image photo.jpg  # + real detections and end-to-end logging
python check_setup.py --camera 0         # + webcam test
```
Each phase prints `PASS`, `FAIL` (with the reason) or `SKIP`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
The suite needs no PyTorch, camera or MySQL server (the detector is tested
against a stand-in model, the camera against a scripted flaky one).

## Deploying
See [DEPLOYMENT.md](DEPLOYMENT.md) for GitHub, Vercel (landing page) and where
the app itself can be hosted. Short version: **Vercel cannot run Streamlit.**

## Verification checklist
1. **Config** – `.env` loads; MySQL shows *Connected* in the sidebar.
2. **Inference** – boxes and labels appear; the class filter and slider change results.
3. **Video** – webcam runs without growing lag (unplug/replug to see auto-reconnect).
4. **Persistence** – new rows appear in `vision_platform.event_logs` with sensible timestamps, classes, confidences and coordinates.
5. **UI** – dashboard stays responsive while the log table refreshes.

## Notes
- Timestamps are stored in **UTC**.
- `bbox_x`/`bbox_y` are the box's top-left corner and `bbox_w`/`bbox_h` its size, in **pixels of the original frame**.
- Streamlit re-runs the script when you touch a widget; the stream stays open and the loop simply restarts with the new settings.
- For browser/cloud deployment, the webcam option reads the *server's* camera; use RTSP, uploads, or `streamlit-webrtc` instead.

## Troubleshooting
| Problem | Fix |
|---|---|
| "Database unavailable" | Start MySQL, check `.env`, press **Retry connection** - or set `DB_BACKEND=auto` / `sqlite` |
| Webcam won't open | Try camera index 1, close other apps using the camera |
| Slow FPS on CPU | Raise the confidence threshold, filter classes, or lower `MAX_DISPLAY_WIDTH` |
| Sample Video disabled | Add `sample.mp4` next to `main.py` |
