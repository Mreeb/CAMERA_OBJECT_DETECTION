"""Camera source abstraction supporting local MP4 simulation, USB webcams, and RTSP streams."""

from collections import deque
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def expand_env_vars(val: str) -> str:
    """Expands environment variables like ${CAMERA_02_RTSP_URL}."""
    if not isinstance(val, str):
        return str(val)
    pattern = re.compile(r"\$\{([^}]+)\}")
    matches = pattern.findall(val)
    for m in matches:
        env_val = os.environ.get(m, "")
        val = val.replace(f"${{{m}}}", env_val)
    return val


@dataclass
class CameraConfig:
    id: str
    name: str
    enabled: bool = True
    source_type: str = "file"  # "file", "usb", "rtsp"
    source: str = ""
    loop: bool = True
    location_id: str = "loading_yard"
    role: str = "primary"  # "primary" or "verification"
    rotation: Any = 0  # 0, "ccw" (270), "cw" (90), 180
    buffer_size: int = 1
    reconnect_timeout_sec: float = 5.0
    max_reconnect_attempts: int = 10


class CameraStream:
    """Threaded camera stream capture with FPS regulation, reconnect logic, and rolling buffer."""

    def __init__(self, config: CameraConfig, rolling_buffer_seconds: float = 15.0):
        self.config = config
        self.rolling_buffer_seconds = rolling_buffer_seconds

        # Resolve source string/env vars
        self.raw_source = config.source
        self.resolved_source = self._resolve_source(config.source, config.source_type)

        # Threading state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame buffers
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = 0
        self._latest_timestamp: float = 0.0
        self._rolling_buffer: Deque[Tuple[float, np.ndarray]] = deque()

        # Health & Performance metrics
        self.is_online = False
        self.capture_fps = 0.0
        self.dropped_frames = 0
        self.total_frames_captured = 0
        self.reconnect_count = 0
        self.width = 1280
        self.height = 720
        self.native_fps = 25.0
        self.last_capture_time = 0.0

    def _resolve_source(self, source_val: str, source_type: str) -> Any:
        expanded = expand_env_vars(str(source_val))
        if source_type == "usb":
            try:
                return int(expanded)
            except ValueError:
                return 0
        elif source_type == "file":
            p = Path(expanded)
            if p.exists():
                return str(p.resolve())
            parent_p = Path("..") / expanded
            if parent_p.exists():
                return str(parent_p.resolve())
            return expanded
        return expanded

    def start(self) -> None:
        """Starts background frame capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"CapThread-{self.config.id}", daemon=True)
        self._thread.start()
        logger.info(f"Started camera stream worker for {self.config.id} ({self.config.source_type})")

    def stop(self) -> None:
        """Stops background capture."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.is_online = False
        logger.info(f"Stopped camera stream worker for {self.config.id}")

    def _capture_loop(self) -> None:
        reconnect_delay = 1.0

        while self._running:
            logger.info(f"Attempting to open camera {self.config.id} at: {self.resolved_source}")
            cap = cv2.VideoCapture(self.resolved_source)

            if not cap.isOpened():
                self.is_online = False
                self.reconnect_count += 1
                logger.warning(
                    f"Failed to open camera {self.config.id}. Retrying in {reconnect_delay:.1f}s "
                    f"(Attempt {self.reconnect_count})"
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(30.0, reconnect_delay * 1.5)
                continue

            # Connection succeeded
            self.is_online = True
            reconnect_delay = 1.0
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0 and fps == fps:
                self.native_fps = float(fps)
            else:
                self.native_fps = 25.0

            frame_interval = 1.0 / self.native_fps if self.native_fps > 0 else 0.04
            is_file = (self.config.source_type == "file")

            fps_calc_window = deque(maxlen=30)
            last_frame_wall_time = time.time()

            try:
                while self._running:
                    target_time = time.time() + frame_interval
                    ret, frame = cap.read()

                    if not ret or frame is None or frame.size == 0:
                        if is_file and self.config.loop:
                            logger.info(f"File stream {self.config.id} reached EOF. Looping back to frame 0.")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            logger.warning(f"Stream {self.config.id} ended or lost connection.")
                            break

                    # Apply rotation if configured
                    rot = self.config.rotation
                    if rot in ("ccw", 270, "270"):
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rot in ("cw", 90, "90"):
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rot in (180, "180"):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    now = time.time()
                    dt = now - last_frame_wall_time
                    last_frame_wall_time = now
                    if dt > 0:
                        fps_calc_window.append(1.0 / dt)
                        self.capture_fps = sum(fps_calc_window) / len(fps_calc_window)

                    with self._lock:
                        self._latest_frame = frame
                        self._latest_frame_idx += 1
                        self._latest_timestamp = now
                        self.last_capture_time = now
                        self.total_frames_captured += 1

                        # Maintain rolling buffer
                        self._rolling_buffer.append((now, frame))
                        cutoff = now - self.rolling_buffer_seconds
                        while self._rolling_buffer and self._rolling_buffer[0][0] < cutoff:
                            self._rolling_buffer.popleft()

                    # Regulate playback speed for simulated file stream
                    if is_file:
                        sleep_dur = target_time - time.time()
                        if sleep_dur > 0:
                            time.sleep(sleep_dur)

            except Exception as e:
                logger.error(f"Unexpected error in capture loop for {self.config.id}: {e}")
            finally:
                cap.release()
                self.is_online = False
                logger.info(f"Camera stream {self.config.id} connection closed.")
                if self._running:
                    time.sleep(1.0)

    def get_latest_frame(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """Returns the newest frame (frame_idx, timestamp, frame) in a thread-safe manner."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame_idx, self._latest_timestamp, self._latest_frame.copy()

    def get_snapshot(self) -> Optional[np.ndarray]:
        """Returns current frame for snapshot saving or canvas calibration."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_health_metrics(self) -> Dict[str, Any]:
        """Returns real-time camera health dictionary."""
        return {
            "camera_id": self.config.id,
            "name": self.config.name,
            "status": "online" if self.is_online else "offline",
            "source_type": self.config.source_type,
            "capture_fps": round(self.capture_fps, 1),
            "native_fps": round(self.native_fps, 1),
            "resolution": f"{self.width}x{self.height}",
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames_captured,
            "reconnect_count": self.reconnect_count,
            "last_capture_time": self.last_capture_time,
        }
