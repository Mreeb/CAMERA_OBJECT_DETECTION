"""FastAPI Server for Factory Activity Recognition, 4-Camera Live Feeds, and Time-Window Analytics."""

import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import yaml

from src.activity_engine import ActivityEngine
from src.detector import YOLOEDetector
from src.multi_camera_manager import MultiCameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("server")

app = FastAPI(title="Factory Activity Recognition Intelligence", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path("config.yaml")
raw_config: Dict[str, Any] = {}

activity_engine: Optional[ActivityEngine] = None
detector: Optional[YOLOEDetector] = None
multi_cam_mgr: Optional[MultiCameraManager] = None


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        if (Path("..") / path).exists():
            path = Path("..") / path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(path: Path, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


@app.on_event("startup")
def startup_event():
    global activity_engine, detector, multi_cam_mgr, raw_config

    logger.info("Initializing Factory Activity Recognition System...")
    raw_config = load_config(CONFIG_PATH)

    activity_engine = ActivityEngine(history_maxlen=1800)

    det_cfg = raw_config.get("detection", {})
    pipeline_cfg = raw_config.get("pipeline", {})
    detector = YOLOEDetector(
        checkpoint=det_cfg.get("checkpoint", "yolov8s-worldv2.pt"),
        fallback_checkpoint=det_cfg.get("fallback_checkpoint", "yolov8s.pt"),
        prompts=det_cfg.get("prompts", ["person", "cardboard box"]),
        person_confidence=float(det_cfg.get("person_confidence", 0.35)),
        article_confidence=float(det_cfg.get("article_confidence", 0.40)),
        iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
        person_article_overlap_iou=float(det_cfg.get("person_article_overlap_iou", 0.30)),
        device=pipeline_cfg.get("device", "auto"),
    )

    cameras_cfg = raw_config.get("cameras", {})
    multi_cam_mgr = MultiCameraManager(
        cameras_config=cameras_cfg,
        detector=detector,
        activity_engine=activity_engine,
        movement_config=raw_config.get("movement", {}),
        worker_activity_config=raw_config.get("movement", {}),
        zones_config=raw_config.get("zones", []),
        counting_line_config=raw_config.get("counting_line", {}),
        inference_fps_per_cam=float(pipeline_cfg.get("inference_fps_per_cam", 5.0)),
    )
    multi_cam_mgr.start()

    logger.info("Factory Activity Recognition System fully started with 4 cameras.")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down multi-camera manager...")
    if multi_cam_mgr:
        multi_cam_mgr.stop()
    logger.info("System stopped cleanly.")


# Activity Recognition REST Endpoints

@app.get("/api/activity/summary")
def get_activity_summary():
    """Returns facility-wide executive activity metrics."""
    if not activity_engine:
        raise HTTPException(status_code=503, detail="Activity engine not initialized")
    return activity_engine.get_overall_summary()


@app.get("/api/activity/cameras")
def get_camera_cards():
    """Returns live metrics and sparkline history for all 4 camera cards."""
    if not multi_cam_mgr:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")
    return multi_cam_mgr.get_all_camera_cards()


@app.get("/api/activity/timeline")
def get_activity_timeline(camera_id: Optional[str] = None, max_points: int = 60):
    """Returns Chart.js compatible time-series datasets."""
    if not activity_engine:
        raise HTTPException(status_code=503, detail="Activity engine not initialized")
    return activity_engine.get_timeline_data(camera_id=camera_id, max_points=max_points)


@app.get("/api/activity/time-windows")
def get_time_windows(camera_id: Optional[str] = None, limit: int = 30):
    """Returns 'From When To When' peak activity intervals."""
    if not activity_engine:
        raise HTTPException(status_code=503, detail="Activity engine not initialized")
    return activity_engine.get_time_windows(camera_id=camera_id, limit=limit)


# On-Demand Live Video Stream (Clean vs AI)
@app.get("/stream/{camera_id}")
def stream_video(camera_id: str, overlay: str = "clean"):
    """Streams live MJPEG feed. Use overlay=clean for raw video or overlay=ai for bounding boxes."""
    if not multi_cam_mgr or camera_id not in multi_cam_mgr.streams:
        raise HTTPException(status_code=404, detail="Camera not found")

    def frame_generator():
        while True:
            jpeg = multi_cam_mgr.get_jpeg_frame(camera_id, overlay=overlay)
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.08)  # ~12 FPS

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Model Management Endpoints

class ModelSwitchRequest(BaseModel):
    model: str


@app.get("/api/models")
def get_available_models():
    """Returns supported YOLO26 model scales and currently active detector scale."""
    if not multi_cam_mgr:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    info = multi_cam_mgr.get_model_info()
    return info


@app.post("/api/models/switch")
def switch_detector_model(req: ModelSwitchRequest):
    """Switches active detector model at runtime."""
    if not multi_cam_mgr:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    model_name = req.model.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name or checkpoint cannot be empty")

    try:
        multi_cam_mgr.switch_model(model_name)
        return {
            "success": True,
            "active_model": model_name,
            "message": f"Successfully switched active detector model to '{model_name}'",
        }
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model to '{model_name}': {str(e)}")


class ConfidenceRequest(BaseModel):
    confidence: float


@app.post("/api/models/confidence")
def set_detector_confidence(req: ConfidenceRequest):
    """Updates detector confidence threshold at runtime."""
    if not multi_cam_mgr:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    new_conf = multi_cam_mgr.set_confidence(req.confidence)
    return {
        "success": True,
        "person_confidence": new_conf,
        "message": f"Updated person detection confidence to {new_conf:.0%}",
    }


# Static Files & Mounts
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    f = static_dir / "index.html"
    return FileResponse(f) if f.exists() else HTMLResponse("<h1>Loading Dashboard...</h1>")


def main():
    parser = argparse.ArgumentParser(description="Factory Activity Recognition Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML")

    args = parser.parse_args()
    global CONFIG_PATH
    CONFIG_PATH = Path(args.config)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
