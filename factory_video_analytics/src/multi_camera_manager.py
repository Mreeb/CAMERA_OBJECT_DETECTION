"""Multi-camera manager orchestrating 4 concurrent video streams, shared GPU inference, and clean vs AI overlays."""

from collections import deque
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .activity_engine import ActivityEngine
from .camera_source import CameraConfig, CameraStream
from .detector import YOLOEDetector
from .state_estimator import MovementConfig
from .tracker import ConfirmedTrack, SessionTracker
from .visualizer import FrameVisualizer
from .zone_engine import ZoneEngine

logger = logging.getLogger(__name__)


class MultiCameraManager:
    """Manages 4 concurrent camera streams, shared GPU detection, and on-demand stream rendering."""

    def __init__(
        self,
        cameras_config: Dict[str, Any],
        detector: YOLOEDetector,
        activity_engine: ActivityEngine,
        movement_config: Dict[str, Any],
        worker_activity_config: Dict[str, Any],
        zones_config: List[Dict[str, Any]],
        counting_line_config: Dict[str, Any],
        inference_fps_per_cam: float = 5.0,
    ):
        self.cameras_config = cameras_config
        self.detector = detector
        self.activity_engine = activity_engine
        self.inference_fps_per_cam = inference_fps_per_cam

        self.movement_cfg = MovementConfig(
            smoothing_alpha=float(movement_config.get("smoothing_alpha", 0.4)),
            warmup_seconds=float(movement_config.get("warmup_seconds", 1.5)),
            window_seconds=float(movement_config.get("window_seconds", 2.0)),
            enter_moving_threshold_px_per_sec=float(movement_config.get("enter_moving_threshold_px_per_sec", 20.0)),
            exit_moving_threshold_px_per_sec=float(movement_config.get("exit_moving_threshold_px_per_sec", 10.0)),
            low_motion_after_seconds=float(worker_activity_config.get("low_motion_after_seconds", 30.0)),
            extended_low_motion_after_seconds=float(worker_activity_config.get("extended_low_motion_after_seconds", 120.0)),
        )

        self.streams: Dict[str, CameraStream] = {}
        self.trackers: Dict[str, SessionTracker] = {}
        self.zone_engines: Dict[str, ZoneEngine] = {}
        self.visualizers: Dict[str, FrameVisualizer] = {}

        # Cached stream frames
        self._clean_jpegs: Dict[str, bytes] = {}
        self._ai_jpegs: Dict[str, bytes] = {}
        self._latest_tracks: Dict[str, List[ConfirmedTrack]] = {}
        self._state_lock = threading.Lock()

        # Initialize sub-systems for each camera
        for cid, cam in self.cameras_config.items():
            source_path = self._resolve_video_path(cam.get("source", ""))
            cam_cfg = CameraConfig(
                id=cid,
                name=cam.get("name", cid),
                enabled=cam.get("enabled", True),
                source_type=cam.get("source_type", "file"),
                source=source_path,
                loop=cam.get("loop", True),
                location_id=cam.get("location_id", "loading_yard"),
                role=cam.get("role", "primary"),
                rotation=cam.get("rotation", 0),
            )

            self.streams[cid] = CameraStream(cam_cfg)
            self.trackers[cid] = SessionTracker(
                lost_grace_period_sec=3.0,
                min_confirmed_frames=3,
                spatial_reid_distance_px=120.0,
                movement_config=self.movement_cfg,
            )
            self.zone_engines[cid] = ZoneEngine(zones_config, counting_line_config)
            self.visualizers[cid] = FrameVisualizer()
            self._latest_tracks[cid] = []

        self.latency_history: Deque[float] = deque(maxlen=25)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def _resolve_video_path(self, path_str: str) -> str:
        p = Path(path_str)
        if p.exists():
            return str(p.resolve())
        p_parent = Path("..") / path_str
        if p_parent.exists():
            return str(p_parent.resolve())
        return path_str

    def start(self) -> None:
        """Starts all camera streams and the continuous multi-camera inference loop."""
        logger.info("Starting MultiCameraManager...")
        self._running = True

        for cid, stream in self.streams.items():
            stream.start()

        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True, name="MultiCamWorker")
        self._worker_thread.start()
        logger.info("MultiCameraManager worker thread active.")

    def stop(self) -> None:
        """Stops all camera streams and background workers."""
        logger.info("Stopping MultiCameraManager...")
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        for cid, stream in self.streams.items():
            stream.stop()

        logger.info("MultiCameraManager stopped.")

    def _inference_loop(self) -> None:
        """Continuous pipeline loop processing frames from all 4 cameras."""
        logger.info(f"Inference loop running at target {self.inference_fps_per_cam} FPS per camera.")
        sample_interval = 1.0 / self.inference_fps_per_cam
        last_sample_times = {cid: 0.0 for cid in self.streams.keys()}

        while self._running:
            try:
                for cid, stream in self.streams.items():
                    now = time.time()
                    if now - last_sample_times[cid] < sample_interval:
                        continue

                    frame_data = stream.get_latest_frame()
                    if frame_data is None:
                        continue

                    frame_idx, frame_ts, frame = frame_data
                    h, w = frame.shape[:2]
                    last_sample_times[cid] = now

                    # 1. GPU Detection with precise timing
                    zone_eng = self.zone_engines[cid]
                    t0_inf = time.perf_counter()
                    detections = self.detector.detect_and_filter(frame, zone_eng)
                    t_inf_ms = (time.perf_counter() - t0_inf) * 1000.0

                    with self._state_lock:
                        self.latency_history.append(t_inf_ms)
                        avg_lat = sum(self.latency_history) / len(self.latency_history)
                        curr_fps = 1000.0 / avg_lat if avg_lat > 0 else self.inference_fps_per_cam

                    # 2. Tracking
                    tracker = self.trackers[cid]
                    tracks = tracker.update(
                        detections=detections,
                        timestamp=frame_ts,
                        zone_engine=zone_eng,
                        frame_w=w,
                        frame_h=h,
                    )

                    # 3. Calculate Activity Metrics
                    visible_people = len([t for t in tracks if t.is_person and t.is_visible])
                    moving_people = len([t for t in tracks if t.is_person and t.is_visible and t.motion_state == "Moving"])
                    low_motion_people = len([t for t in tracks if t.is_person and t.is_visible and t.motion_state in ("Stationary", "Low Motion", "Extended Low Motion", "Unknown")])

                    velocities = [t.smoother.current_velocity for t in tracks if t.is_person and t.is_visible]
                    avg_vel = (sum(velocities) / len(velocities)) if velocities else 0.0

                    self.activity_engine.record_sample(
                        camera_id=cid,
                        timestamp=now,
                        visible_people=visible_people,
                        moving_people=moving_people,
                        low_motion_people=low_motion_people,
                        avg_velocity=avg_vel,
                        article_transfers=0,
                    )

                    # 4. Generate Clean & AI JPEG Frames
                    ret_clean, clean_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                    # AI frame (with live measured latency & FPS)
                    annotated, _ = self.visualizers[cid].draw(
                        frame=frame,
                        tracks=tracks,
                        zone_engine=zone_eng,
                        camera_id=cid,
                        capture_fps=stream.capture_fps,
                        inference_fps=curr_fps,
                        latency_ms=avg_lat,
                    )
                    ret_ai, ai_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

                    with self._state_lock:
                        if ret_clean:
                            self._clean_jpegs[cid] = clean_buf.tobytes()
                        if ret_ai:
                            self._ai_jpegs[cid] = ai_buf.tobytes()
                        self._latest_tracks[cid] = tracks
            except Exception as e:
                logger.error(f"Error in multi-camera inference loop: {e}", exc_info=True)

            time.sleep(0.01)

    def get_jpeg_frame(self, camera_id: str, overlay: str = "clean") -> Optional[bytes]:
        """Returns clean or AI JPEG frame for the requested camera."""
        with self._state_lock:
            if overlay == "ai":
                return self._ai_jpegs.get(camera_id)
            return self._clean_jpegs.get(camera_id)

    def get_camera_card_data(self, camera_id: str) -> Dict[str, Any]:
        """Returns real-time activity metrics and card information for a camera."""
        stream = self.streams.get(camera_id)
        cam_cfg = self.cameras_config.get(camera_id, {})
        history = self.activity_engine.get_or_create_history(camera_id)

        latest_sample = history[-1] if history else None
        score = latest_sample.activity_score if latest_sample else 0.0
        level = latest_sample.activity_level if latest_sample else "IDLE"
        vis_people = latest_sample.visible_people if latest_sample else 0
        mov_people = latest_sample.moving_people if latest_sample else 0
        low_people = latest_sample.low_motion_people if latest_sample else 0

        # Sparkline points (last 15 samples)
        sparkline = [s.activity_score for s in list(history)[-15:]]

        return {
            "camera_id": camera_id,
            "name": cam_cfg.get("name", camera_id),
            "location_id": cam_cfg.get("location_id", "loading_yard"),
            "role": cam_cfg.get("role", "primary"),
            "status": "online" if (stream and stream.is_online) else "offline",
            "capture_fps": round(stream.capture_fps, 1) if stream else 0.0,
            "activity_score": score,
            "activity_level": level,
            "visible_people": vis_people,
            "moving_people": mov_people,
            "low_motion_people": low_people,
            "sparkline": sparkline,
            "snapshot_url": f"/api/snapshot/{camera_id}",
            "stream_clean_url": f"/stream/{camera_id}?overlay=clean",
            "stream_ai_url": f"/stream/{camera_id}?overlay=ai",
        }

    def get_all_camera_cards(self) -> List[Dict[str, Any]]:
        """Returns data for all 4 camera cards."""
        return [self.get_camera_card_data(cid) for cid in self.cameras_config.keys()]

    def switch_model(self, checkpoint: str) -> bool:
        """Dynamically updates the underlying detector model."""
        with self._state_lock:
            self.latency_history.clear()
            return self.detector.switch_model(checkpoint)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns active detector model metadata with real-time measured latency and FPS."""
        info = self.detector.get_model_info()
        with self._state_lock:
            avg_lat = (sum(self.latency_history) / len(self.latency_history)) if self.latency_history else 18.5
            inf_fps = (1000.0 / avg_lat) if avg_lat > 0 else 50.0
        info["measured_latency_ms"] = round(avg_lat, 1)
        info["measured_fps"] = round(inf_fps, 1)
        return info

    def set_confidence(self, confidence: float) -> float:
        """Dynamically updates the detector person confidence threshold."""
        with self._state_lock:
            return self.detector.set_confidence(confidence)
