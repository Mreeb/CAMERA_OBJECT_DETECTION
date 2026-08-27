"""Core decoupled pipeline worker managing capture, inference, tracking, zones, and live stream publishing."""

from collections import deque
import logging
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .camera_source import CameraConfig, CameraStream
from .database import DatabaseManager
from .detector import YOLOEDetector
from .state_estimator import MovementConfig
from .tracker import ConfirmedTrack, SessionTracker
from .visualizer import FrameVisualizer
from .zone_engine import ZoneEngine

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Manages the lifecycle of a single camera monitoring pipeline."""

    def __init__(
        self,
        camera_config: CameraConfig,
        pipeline_config: Dict[str, Any],
        detection_config: Dict[str, Any],
        movement_config: Dict[str, Any],
        worker_activity_config: Dict[str, Any],
        zones_config: List[Dict[str, Any]],
        counting_line_config: Dict[str, Any],
        database: DatabaseManager,
        snapshots_dir: str | Path = "data/snapshots",
    ):
        self.camera_config = camera_config
        self.database = database
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.inference_fps = float(pipeline_config.get("inference_fps", 8.0))
        self.display_fps = float(pipeline_config.get("display_fps", 12.0))
        self.device = pipeline_config.get("device", "auto")

        # Initialize Subsystems
        self.camera = CameraStream(camera_config)
        
        self.detector = YOLOEDetector(
            checkpoint=detection_config.get("checkpoint", "yolov8s-worldv2.pt"),
            fallback_checkpoint=detection_config.get("fallback_checkpoint", "yolov8s.pt"),
            prompts=detection_config.get("prompts", ["person", "cardboard box"]),
            person_confidence=float(detection_config.get("person_confidence", 0.35)),
            article_confidence=float(detection_config.get("article_confidence", 0.40)),
            iou_threshold=float(detection_config.get("iou_threshold", 0.45)),
            person_article_overlap_iou=float(detection_config.get("person_article_overlap_iou", 0.30)),
            device=self.device,
        )

        mov_cfg = MovementConfig(
            smoothing_alpha=float(movement_config.get("smoothing_alpha", 0.4)),
            warmup_seconds=float(movement_config.get("warmup_seconds", 1.5)),
            window_seconds=float(movement_config.get("window_seconds", 2.0)),
            enter_moving_threshold_px_per_sec=float(movement_config.get("enter_moving_threshold_px_per_sec", 20.0)),
            exit_moving_threshold_px_per_sec=float(movement_config.get("exit_moving_threshold_px_per_sec", 10.0)),
            movement_confirmation_seconds=float(movement_config.get("movement_confirmation_seconds", 1.0)),
            stationary_confirmation_seconds=float(movement_config.get("stationary_confirmation_seconds", 2.0)),
            meaningful_article_displacement_px=float(movement_config.get("meaningful_article_displacement_px", 30.0)),
            low_motion_after_seconds=float(worker_activity_config.get("low_motion_after_seconds", 30.0)),
            extended_low_motion_after_seconds=float(worker_activity_config.get("extended_low_motion_after_seconds", 120.0)),
        )

        self.tracker = SessionTracker(
            lost_grace_period_sec=3.0,
            min_confirmed_frames=3,
            spatial_reid_distance_px=120.0,
            movement_config=mov_cfg,
        )

        self.zone_engine = ZoneEngine(
            zones_config=zones_config,
            counting_line_config=counting_line_config,
        )

        self.visualizer = FrameVisualizer()

        # Threading state
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._publisher_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        # Cached outputs for live stream
        self._latest_jpeg: Optional[bytes] = None
        self._cached_tracks: List[ConfirmedTrack] = []
        self._prev_track_states: Dict[str, str] = {}
        self._prev_track_zones: Dict[str, str] = {}

        # Performance Metrics
        self.inference_fps_actual = 0.0
        self.display_fps_actual = 0.0
        self.avg_latency_ms = 0.0
        self.confirmed_transfers_count = 0

    def start(self) -> None:
        """Starts camera stream and background inference & publisher threads."""
        if self._running:
            return
        self._running = True
        self.camera.start()

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"Inference-{self.camera_config.id}",
            daemon=True,
        )
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop,
            name=f"Publisher-{self.camera_config.id}",
            daemon=True,
        )

        self._inference_thread.start()
        self._publisher_thread.start()
        logger.info(f"Pipeline worker for {self.camera_config.id} fully started.")

    def stop(self) -> None:
        """Stops pipeline worker."""
        self._running = False
        self.camera.stop()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        if self._publisher_thread and self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=2.0)
        logger.info(f"Pipeline worker for {self.camera_config.id} stopped.")

    def _inference_loop(self) -> None:
        interval = 1.0 / self.inference_fps
        fps_tracker: Deque[float] = deque(maxlen=20)
        last_t = time.time()
        last_health_log_time = time.time()

        while self._running:
            loop_start = time.time()
            frame_data = self.camera.get_latest_frame()

            if frame_data is not None:
                frame_idx, frame_ts, frame = frame_data
                h, w = frame.shape[:2]

                # 1. Run Detection & Filtering
                inf_start = time.time()
                detections = self.detector.detect_and_filter(frame, self.zone_engine)

                # 2. Update Session Tracker & Smoother
                tracks = self.tracker.update(
                    detections=detections,
                    timestamp=frame_ts,
                    zone_engine=self.zone_engine,
                    frame_w=w,
                    frame_h=h,
                )

                # 3. Evaluate Events & State Transitions
                self._check_and_log_events(tracks, frame, frame_ts, w, h)

                # 4. Measure Latency & Inference FPS
                now = time.time()
                latency = (now - frame_ts) * 1000.0
                self.avg_latency_ms = max(0.0, latency)

                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps_tracker.append(1.0 / dt)
                    self.inference_fps_actual = sum(fps_tracker) / len(fps_tracker)

                with self._state_lock:
                    self._cached_tracks = tracks

                # Periodic health logging to database
                if now - last_health_log_time >= 5.0:
                    health_metrics = self.tracker.get_health_metrics(now)
                    self.database.insert_health_log({
                        "timestamp": now,
                        "camera_id": self.camera_config.id,
                        "status": "online" if self.camera.is_online else "offline",
                        "capture_fps": self.camera.capture_fps,
                        "inference_fps": self.inference_fps_actual,
                        "display_fps": self.display_fps_actual,
                        "latency_ms": self.avg_latency_ms,
                        "dropped_frames": self.camera.dropped_frames,
                        "active_tracks": len(tracks),
                        "id_fragmentation_rate": health_metrics["new_ids_last_minute"],
                    })
                    last_health_log_time = now

            # Sleep to maintain inference FPS
            elapsed = time.time() - loop_start
            sleep_dur = interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    def _publisher_loop(self) -> None:
        interval = 1.0 / self.display_fps
        fps_tracker: Deque[float] = deque(maxlen=20)
        last_t = time.time()

        while self._running:
            loop_start = time.time()
            frame_data = self.camera.get_latest_frame()

            if frame_data is not None:
                frame_idx, frame_ts, frame = frame_data

                with self._state_lock:
                    tracks = list(self._cached_tracks)

                # Draw Visualizer Overlay
                annotated, rendered_persons = self.visualizer.draw(
                    frame=frame,
                    tracks=tracks,
                    zone_engine=self.zone_engine,
                    camera_id=self.camera_config.id,
                    capture_fps=self.camera.capture_fps,
                    inference_fps=self.inference_fps_actual,
                    latency_ms=self.avg_latency_ms,
                    confirmed_transfers=self.confirmed_transfers_count,
                )

                # Consistency check assertion
                visible_people_count = len([t for t in tracks if t.is_person and t.is_visible])
                if rendered_persons != visible_people_count:
                    logger.warning(
                        f"[Consistency Mismatch] Rendered Person Boxes ({rendered_persons}) != "
                        f"Visible People Tracker Count ({visible_people_count})"
                    )

                # JPEG Compression for MJPEG stream
                ret, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self._state_lock:
                        self._latest_jpeg = jpeg_buf.tobytes()

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps_tracker.append(1.0 / dt)
                    self.display_fps_actual = sum(fps_tracker) / len(fps_tracker)

            elapsed = time.time() - loop_start
            sleep_dur = interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    def _check_and_log_events(
        self,
        tracks: List[ConfirmedTrack],
        frame: np.ndarray,
        timestamp: float,
        w: int,
        h: int,
    ) -> None:
        """Evaluates domain events (zone transitions, motion state changes, line crossings)."""
        for track in tracks:
            sid = track.session_id
            curr_state = track.motion_state
            curr_zone = track.current_zone

            prev_state = self._prev_track_states.get(sid)
            prev_zone = self._prev_track_zones.get(sid)

            # 1. Check Motion State Change
            if prev_state is not None and prev_state != curr_state:
                event_type = f"{curr_state.lower().replace(' ', '_')}_detected"
                snap_path = self._save_event_snapshot(frame, sid, event_type)
                self.database.insert_event({
                    "timestamp": timestamp,
                    "camera_id": self.camera_config.id,
                    "session_id": sid,
                    "track_id": track.internal_track_id,
                    "class_name": track.class_name,
                    "event_type": event_type,
                    "state": curr_state,
                    "zone": curr_zone,
                    "details": f"State changed from {prev_state} to {curr_state}",
                    "snapshot_path": snap_path,
                })

            # 2. Check Zone Change
            if prev_zone is not None and prev_zone != curr_zone and curr_zone != "Unknown":
                event_type = "zone_transition"
                if prev_zone == "Storage Area" and curr_zone == "Vehicle Loading Area":
                    self.confirmed_transfers_count += 1
                    event_type = "article_transferred"

                snap_path = self._save_event_snapshot(frame, sid, event_type)
                self.database.insert_event({
                    "timestamp": timestamp,
                    "camera_id": self.camera_config.id,
                    "session_id": sid,
                    "track_id": track.internal_track_id,
                    "class_name": track.class_name,
                    "event_type": event_type,
                    "state": curr_state,
                    "zone": curr_zone,
                    "details": f"Moved from {prev_zone} to {curr_zone}",
                    "snapshot_path": snap_path,
                })

            # 3. Check Line Crossing
            if len(track.smoother.history) >= 2:
                prev_pos = (track.smoother.history[-2][1], track.smoother.history[-2][2])
                curr_pos = track.bottom_center
                direction = self.zone_engine.check_line_crossing(
                    prev_pos, curr_pos, track.class_name, track.internal_track_id, timestamp, w, h
                )
                if direction:
                    snap_path = self._save_event_snapshot(frame, sid, "line_crossing")
                    self.database.insert_event({
                        "timestamp": timestamp,
                        "camera_id": self.camera_config.id,
                        "session_id": sid,
                        "track_id": track.internal_track_id,
                        "class_name": track.class_name,
                        "event_type": "line_crossing",
                        "state": curr_state,
                        "zone": curr_zone,
                        "details": f"Crossed counting line in direction {direction}",
                        "snapshot_path": snap_path,
                    })

            self._prev_track_states[sid] = curr_state
            self._prev_track_zones[sid] = curr_zone

    def _save_event_snapshot(self, frame: np.ndarray, session_id: str, event_type: str) -> str:
        """Saves a cropped or full event snapshot image."""
        filename = f"snap_{self.camera_config.id}_{session_id}_{event_type}_{int(time.time() * 1000)}.jpg"
        filepath = self.snapshots_dir / filename
        cv2.imwrite(str(filepath), frame)
        return f"/snapshots/{filename}"

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the newest JPEG frame for MJPEG streaming."""
        with self._state_lock:
            return self._latest_jpeg

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Returns live dashboard status card metrics."""
        now = time.time()
        health = self.tracker.get_health_metrics(now)
        cam_health = self.camera.get_health_metrics()

        with self._state_lock:
            tracks = list(self._cached_tracks)

        visible_people = [t for t in tracks if t.is_person and t.is_visible]
        moving_people = len([t for t in visible_people if t.motion_state == "Moving"])
        low_motion_people = len([t for t in visible_people if t.motion_state in ("Low Motion", "Extended Low Motion")])
        unknown_people = len([t for t in visible_people if t.motion_state == "Unknown"])

        visible_articles = [t for t in tracks if not t.is_person and t.is_visible]
        moving_articles = len([t for t in visible_articles if t.motion_state == "Moving"])

        return {
            "camera_id": self.camera_config.id,
            "camera_name": self.camera_config.name,
            "status": cam_health["status"],
            "capture_fps": cam_health["capture_fps"],
            "inference_fps": round(self.inference_fps_actual, 1),
            "display_fps": round(self.display_fps_actual, 1),
            "latency_ms": round(self.avg_latency_ms, 0),
            "dropped_frames": cam_health["dropped_frames"],
            
            # Live People Cards
            "people_visible_now": len(visible_people),
            "people_moving_now": moving_people,
            "people_low_motion_now": low_motion_people,
            "people_unknown_now": unknown_people,
            "people_temporarily_lost": health["people_temporarily_lost"],
            "max_simultaneous_people": health["max_simultaneous_people"],
            
            # Live Article Cards
            "articles_visible_now": len(visible_articles),
            "articles_moving_now": moving_articles,
            "confirmed_transfers_today": self.confirmed_transfers_count,
            "line_crossings": self.zone_engine.line_crossings,
            
            # Health & Integrity
            "new_ids_last_minute": health["new_ids_last_minute"],
            "id_fragmentation_warning": health["id_fragmentation_warning"],
        }

    def get_active_workers(self) -> List[Dict[str, Any]]:
        """Returns live list of active worker tracks."""
        with self._state_lock:
            tracks = [t for t in self.tracker.get_all_active_tracks() if t.is_person]

        return [
            {
                "session_id": t.session_id,
                "class_name": t.class_name,
                "motion_state": t.motion_state,
                "current_zone": t.current_zone,
                "confidence": round(t.confidence, 2),
                "is_visible": t.is_visible,
                "duration_seconds": round(t.duration, 1),
                "time_in_state_seconds": round(t.smoother.time_in_current_state, 1),
                "velocity_px_sec": round(t.smoother.current_velocity, 1),
            }
            for t in tracks
        ]

    def get_active_articles(self) -> List[Dict[str, Any]]:
        """Returns live list of active article tracks."""
        with self._state_lock:
            tracks = [t for t in self.tracker.get_all_active_tracks() if not t.is_person]

        return [
            {
                "session_id": t.session_id,
                "class_name": t.class_name,
                "motion_state": t.motion_state,
                "movement_category": t.smoother.article_movement_state,
                "current_zone": t.current_zone,
                "confidence": round(t.confidence, 2),
                "is_visible": t.is_visible,
                "duration_seconds": round(t.duration, 1),
                "net_displacement_px": round(t.smoother.net_displacement, 1),
                "has_transferred": t.smoother.has_transferred_zones,
            }
            for t in tracks
        ]
