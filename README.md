# Factory Vision AI - Multi-Camera Activity Recognition & Intelligence

A real-time, multi-camera industrial intelligence platform designed for automated factory worker activity recognition, motion state estimation, and live surveillance analytics using **YOLO26** and custom fine-tuned models.

---

## ⚡ Key Capabilities

- **Real-Time Multi-Camera Monitoring**: Concurrent processing across simulated camera streams (with instant RTSP / IP camera support).
- **YOLO26 Model Architecture**:
  - **★ Specialized Model**: Custom fine-tuned weights (`factory_person_yolo26n_best.pt`) optimized specifically for factory workers and occlusion robustness.
  - **Standard YOLO26 Scales**: Real-time switching between `Nano` (`yolo26n.pt`), `Small` (`yolo26s.pt`), `Medium` (`yolo26m.pt`), `Large` (`yolo26l.pt`), and `X-Large` (`yolo26x.pt`).
- **Targeted Inference (`classes=[0]`)**: Inference focuses exclusively on persons/workers, skipping all COCO background clutter for maximum FPS and sub-second latency.
- **Robust Worker Tracking & Motion State Engine**:
  - IoU + Normalized Center Distance spatial matching with track-level deduplication (zero ghost boxes).
  - Trajectory smoothing with motion state classification (`Moving`, `Stationary`, `Low Motion`, `Extended Low Motion`).
  - Clean human-facing worker ID recycling (`W-01`, `W-02`, etc.).
- **Live Dark Industrial Dashboard**:
  - Pitch-black minimalist UI with real-time continuous wave timeline charts and single-camera filtering pills.
  - Runtime Detection Confidence Slider (10% to 90%).
  - Live measured GPU inference latency (ms) and throughput (FPS) indicators.
  - Interactive live stream viewer with toggleable **Clean Stream** and **AI Diagnostics** overlays.
- **Automated Training Dataset Generator**: Includes `extract_training_frames.py` for extracting standardized horizontal training frames at 30 frames/min from video feeds.

---

## 📂 Repository Structure

```text
CAMERA_OBJECT_DETECTION/
├── TRAINING_FRAMES/             # Extracted horizontal training frames (265 images)
├── Videos/                      # Factory camera video test sources
├── extract_training_frames.py   # Automated training frame extraction tool
└── factory_video_analytics/     # Core intelligence platform
    ├── factory_person_yolo26n_best.pt  # Fine-tuned specialized model
    ├── yolo26n.pt / yolo26s.pt ...     # YOLO26 standard model scales
    ├── config.yaml              # Multi-camera & functional zone config
    ├── server.py                # FastAPI web server & MJPEG streaming engine
    ├── app.py                   # CLI analytics pipeline
    ├── src/                     # Detector, tracker, activity engine, visualizer
    └── static/                  # Dark industrial dashboard frontend
```

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
cd factory_video_analytics
pip install -r requirements.txt
```

### 2. Launching the Live Dashboard
```bash
python server.py --host 127.0.0.1 --port 8000
```
Open your browser and navigate to: **`http://127.0.0.1:8000/`**

### 3. Extracting Training Frames
```bash
python extract_training_frames.py
```
