"""Folder watcher service monitoring incoming camera directories, validating recordings, and enqueuing jobs."""

from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2

from .database import DatabaseManager

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


class FolderWatcherService:
    """Scans incoming camera folders, verifies recording completion, and enqueues jobs."""

    def __init__(
        self,
        incoming_base_dir: str | Path,
        database: DatabaseManager,
        cameras_config: Dict[str, Any],
        stability_check_interval_sec: float = 1.5,
        scan_interval_sec: float = 2.0,
    ):
        self.incoming_base_dir = Path(incoming_base_dir)
        self.database = database
        self.cameras_config = cameras_config
        self.stability_check_interval = stability_check_interval_sec
        self.scan_interval = scan_interval_sec

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_size_cache: Dict[str, Tuple[int, float]] = {}  # path -> (size, timestamp)

        # Create incoming camera subdirectories
        for cam_id in self.cameras_config.keys():
            cam_dir = self.incoming_base_dir / cam_id
            cam_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="FolderWatcher", daemon=True)
        self._thread.start()
        logger.info(f"FolderWatcherService started for: {self.incoming_base_dir}")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("FolderWatcherService stopped.")

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.scan_and_enqueue()
            except Exception as e:
                logger.error(f"Error during folder watcher scan: {e}", exc_info=True)
            time.sleep(self.scan_interval)

    def scan_and_enqueue(self) -> List[int]:
        """Scans all incoming camera folders and enqueues ready clips."""
        enqueued_ids = []

        for cam_id, cam_cfg in self.cameras_config.items():
            if not cam_cfg.get("enabled", True):
                continue

            cam_dir = self.incoming_base_dir / cam_id
            if not cam_dir.exists():
                cam_dir.mkdir(parents=True, exist_ok=True)

            video_files = list(cam_dir.glob("*.mp4")) + list(cam_dir.glob("*.avi")) + list(cam_dir.glob("*.mkv"))
            # Sort by name / modification time
            video_files.sort(key=lambda p: (p.stat().st_mtime, p.name))

            for video_path in video_files:
                # 1. Verify file completion (size stability check)
                if not self._is_file_stable(video_path):
                    logger.debug(f"File {video_path.name} is still being written. Skipping for now.")
                    continue

                # 2. Process and enqueue
                clip_id = self._ingest_clip(video_path, cam_id, cam_cfg)
                if clip_id:
                    enqueued_ids.append(clip_id)

        return enqueued_ids

    def _is_file_stable(self, filepath: Path) -> bool:
        """Verifies if the file size has stopped changing (NVR write finished)."""
        try:
            current_size = filepath.stat().st_size
            now = time.time()

            if str(filepath) not in self._file_size_cache:
                self._file_size_cache[str(filepath)] = (current_size, now)
                return False

            prev_size, prev_time = self._file_size_cache[str(filepath)]
            if now - prev_time < self.stability_check_interval:
                return False

            if current_size == prev_size and current_size > 0:
                # Size has remained identical for at least stability_check_interval
                del self._file_size_cache[str(filepath)]
                return True
            else:
                self._file_size_cache[str(filepath)] = (current_size, now)
                return False
        except Exception:
            return False

    def _ingest_clip(self, video_path: Path, camera_id: str, cam_cfg: Dict[str, Any]) -> Optional[int]:
        """Validates video, checks duplicates via hash, and enqueues in SQLite."""
        try:
            # 1. Compute SHA-256
            file_hash = compute_file_sha256(video_path)

            # 2. Check duplicate in DB
            if self.database.clip_exists_by_hash(file_hash):
                logger.warning(f"Clip {video_path.name} (hash: {file_hash[:8]}...) already exists in DB. Skipping duplicate.")
                # If duplicate, we can delete or keep
                return None

            # 3. Validate video integrity using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video file {video_path.name}. Corrupted recording.")
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, first_frame = cap.read()
            cap.release()

            if not ret or frame_count <= 0 or fps <= 0:
                logger.error(f"Invalid video metadata for {video_path.name}: frames={frame_count}, fps={fps}")
                return None

            duration_sec = frame_count / fps

            # 4. Parse timestamp from filename or metadata
            start_ts, is_estimated = self._parse_timestamp(video_path, cam_cfg)
            end_ts = start_ts + duration_sec

            dt_start = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
            dt_end = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")

            clip_data = {
                "camera_id": camera_id,
                "location_id": cam_cfg.get("location_id", "default"),
                "filename": video_path.name,
                "file_hash": file_hash,
                "file_path": str(video_path.resolve()),
                "start_time": start_ts,
                "end_time": end_ts,
                "datetime_start": dt_start,
                "datetime_end": dt_end,
                "duration_sec": duration_sec,
                "frame_count": frame_count,
                "fps": fps,
                "is_estimated_timestamp": is_estimated,
                "model_version": "yolov8s-worldv2",
                "config_version": "v2.0",
            }

            priority = 20 if cam_cfg.get("role") == "primary" else 10
            clip_id = self.database.insert_clip_and_job(clip_data, priority=priority)
            logger.info(f"Successfully enqueued clip ID {clip_id}: {video_path.name} ({duration_sec:.1f}s, start: {dt_start})")
            return clip_id

        except Exception as e:
            logger.error(f"Failed to ingest clip {video_path.name}: {e}", exc_info=True)
            return None

    def _parse_timestamp(self, filepath: Path, cam_cfg: Dict[str, Any]) -> Tuple[float, bool]:
        """Parses start timestamp from filename with clock offset adjustment."""
        clock_offset = float(cam_cfg.get("clock_offset_sec", 0.0))
        filename = filepath.stem

        # Pattern 1: camera_02_2026-08-20_09-00-00
        m1 = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", filename)
        if m1:
            dt_str = f"{m1.group(1)}-{m1.group(2)}-{m1.group(3)} {m1.group(4)}:{m1.group(5)}:{m1.group(6)}"
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp() + clock_offset, False
            except ValueError:
                pass

        # Pattern 2: 20260820_090000
        m2 = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", filename)
        if m2:
            dt_str = f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)} {m2.group(4)}:{m2.group(5)}:{m2.group(6)}"
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp() + clock_offset, False
            except ValueError:
                pass

        # Fallback: File modification time
        mtime = filepath.stat().st_mtime
        logger.info(f"Filename {filepath.name} lacks timestamp pattern. Using file mtime ({datetime.fromtimestamp(mtime)}) [ESTIMATED].")
        return mtime + clock_offset, True
