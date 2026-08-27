"""Evidence Manager for saving keyframe snapshots and 15-second event video clips."""

from datetime import datetime
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

logger = logging.getLogger(__name__)


class EvidenceManager:
    """Extracts, formats, and stores event evidence snapshots and 15-second MP4 clips."""

    def __init__(self, evidence_base_dir: str | Path = "data/evidence"):
        self.base_dir = Path(evidence_base_dir)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.clips_dir = self.base_dir / "clips"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        frame: np.ndarray,
        camera_id: str,
        session_id: str,
        event_type: str,
        timestamp: float,
        clip_id: Optional[int] = None,
    ) -> str:
        """Saves a high-resolution keyframe snapshot."""
        ts_int = int(timestamp * 1000)
        filename = f"snap_{camera_id}_{session_id}_{event_type}_{ts_int}.jpg"
        filepath = self.snapshots_dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        logger.debug(f"Saved evidence snapshot: {filename}")
        return f"/evidence/snapshots/{filename}"

    def extract_evidence_clip(
        self,
        source_video_path: str | Path,
        event_time_in_video_sec: float,
        camera_id: str,
        session_id: str,
        event_type: str,
        pre_seconds: float = 5.0,
        post_seconds: float = 10.0,
    ) -> Optional[str]:
        """Extracts a 15-second slice around the event moment using imageio-ffmpeg."""
        try:
            source_path = Path(source_video_path)
            if not source_path.exists():
                logger.warning(f"Source video {source_video_path} not found for evidence clip extraction.")
                return None

            start_t = max(0.0, event_time_in_video_sec - pre_seconds)
            duration_t = pre_seconds + post_seconds

            ts_int = int(time.time() * 1000)
            filename = f"clip_{camera_id}_{session_id}_{event_type}_{ts_int}.mp4"
            out_filepath = self.clips_dir / filename

            cmd = [
                FFMPEG_EXE,
                "-y",
                "-ss", f"{start_t:.2f}",
                "-i", str(source_path.resolve()),
                "-t", f"{duration_t:.2f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(out_filepath.resolve()),
            ]

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and out_filepath.exists() and out_filepath.stat().st_size > 0:
                logger.info(f"Successfully created 15s evidence clip: {filename}")
                return f"/evidence/clips/{filename}"
            else:
                logger.warning(f"FFmpeg failed to create evidence clip: {res.stderr[:200]}")
                return None

        except Exception as e:
            logger.error(f"Error extracting evidence clip: {e}", exc_info=True)
            return None
