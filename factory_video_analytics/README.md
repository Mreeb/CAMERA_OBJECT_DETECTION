# Factory Vision AI - Real-Time Multi-Camera Monitoring System

A modular, real-time computer vision monitoring and analytics system for factory and warehouse yards using open-vocabulary detection (Ultralytics YOLOE / YOLO-World), persistent multi-object tracking, smoothed state estimation, interactive polygon zones, and a live web dashboard.

---

## 🎯 Key Features & Algorithmic Corrections

- **Real-Time Web Dashboard**: Live MJPEG video stream, KPI metrics cards, active worker list, active article list, real-time event timeline with snapshot evidence modal.
- **Decoupled Architecture**: Persistent background capture and inference workers decoupled from the web server. Opening, closing, or refreshing the dashboard never interrupts tracking or resets IDs.
- **Stable Session Display IDs & Grace Period**: Assigns stable display IDs (`P-01`, `P-02`, ..., `B-01`, `B-02`, ...) with a 3.0s `temporarily_lost` grace period and spatial re-association to eliminate track ID explosions.
- **Conflict & Duplicate Suppression**: Removes `hand cart` / `trolley` misclassifications; enforces person-over-article priority to prevent people from being detected as boxes.
- **Smoothed Bottom-Center Movement**: Tracks bottom-center $(x, y_2)$ anchor for people with Exponential Moving Average (EMA) smoothing; applies hysteresis thresholds (20 px/s enter, 10 px/s exit) over a 2.0s rolling window with initial warmup dampening.
- **Interactive Zone Calibration**: Web-based HTML5 canvas editor allowing operators to draw, label, and save functional zones (`Work Area`, `Storage Area`, `Loading Area`, `Exclusion Area`, `Pickup Area`, `Destination Area`) and directional counting lines.
- **Multi-Camera Ready**: Data structures and configuration prepared for 4 cameras with `primary` and `verification` roles and location identifiers.
- **SQLite Persistent Event Database**: Automatically logs timestamped state changes, zone transitions, line crossings, and camera health metrics to `data/events.db` with image snapshots in `data/snapshots/`.

---

## 📁 Directory Structure

```text
factory_video_analytics/
├── server.py                   # FastAPI real-time web server & streaming backend
├── app.py                      # Offline batch video analytics CLI
├── config.yaml                 # Central YAML configuration (cameras, zones, thresholds)
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── .env.example                # RTSP credentials environment template
├── data/
│   ├── events.db               # SQLite persistent event database
│   └── snapshots/              # Captured event snapshot images
├── src/
│   ├── camera_source.py        # Camera capture (MP4 pacing, USB webcam, RTSP, reconnect)
│   ├── detector.py             # Open-vocabulary model wrapper & conflict suppression
│   ├── tracker.py              # SessionTracker with stable IDs & 3s lost grace period
│   ├── state_estimator.py      # EMA bottom-center smoothing, velocity window, hysteresis
│   ├── zone_engine.py          # Polygon zones, point-in-poly, exclusion masks, counting lines
│   ├── database.py             # SQLite database manager & queries
│   ├── pipeline_worker.py      # Core decoupled camera processing thread
│   └── visualizer.py           # Real-time HUD, bounding boxes, motion badges, zone overlays
└── static/
    ├── index.html              # Main real-time monitoring dashboard
    ├── calibration.html        # Interactive polygon zone and counting line editor
    ├── css/
    │   └── style.css           # Modern dark-mode styling with glassmorphism
    └── js/
        ├── app.js              # Dashboard data polling, live tables, event feed
        └── calibration.js      # Canvas polygon drawing, dragging, and zone saving
```

---

## 🚀 Quick Start & Usage

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Launch the Real-Time Monitoring Server
```bash
python server.py --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
- **Main Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Zone Calibration & Editor**: [http://127.0.0.1:8000/calibration](http://127.0.0.1:8000/calibration)

---

## 📹 Configuring Camera Sources (`config.yaml`)

### A. Simulated Live Stream (Local MP4)
Reads the video at its native frame rate with looping:
```yaml
cameras:
  - id: "camera_02"
    name: "Loading Yard (Camera 02)"
    enabled: true
    source_type: "file"
    source: "Videos/VID-20260820-WA0003.mp4"
    loop: true
```

### B. USB Webcam
```yaml
cameras:
  - id: "camera_usb_01"
    name: "USB Camera 01"
    enabled: true
    source_type: "usb"
    source: "0"  # Camera index
```

### C. Network RTSP Stream
Use environment variables for credentials:
```yaml
cameras:
  - id: "camera_rtsp_01"
    name: "Factory Yard RTSP"
    enabled: true
    source_type: "rtsp"
    source: "${CAMERA_02_RTSP_URL}"
```
Configure your `.env` file:
```bash
CAMERA_02_RTSP_URL=rtsp://admin:password@192.168.1.102:554/live
```

---

## ⚙️ Core Configuration Settings

| Setting | Default | Description |
| :--- | :--- | :--- |
| `pipeline.inference_fps` | `8` | Target model inference cycles per second |
| `pipeline.display_fps` | `12` | Live stream publishing frame rate |
| `detection.prompts` | `[person, cardboard box]` | Active open-vocabulary prompts |
| `detection.person_confidence` | `0.35` | Minimum confidence for people |
| `detection.article_confidence` | `0.40` | Minimum confidence for articles |
| `tracking.lost_grace_period_sec` | `3.0` | Seconds to hold lost track before ID reset |
| `movement.warmup_seconds` | `1.5` | Initial period where velocity spikes are ignored |
| `movement.enter_moving_threshold_px_per_sec` | `20.0` | Velocity to enter `Moving` state |
| `worker_activity.low_motion_after_seconds` | `30.0` | Sustained low motion threshold |
| `worker_activity.extended_low_motion_after_seconds` | `120.0` | Sustained extended low motion threshold |
