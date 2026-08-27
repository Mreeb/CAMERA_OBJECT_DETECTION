"""Video information probing and streaming utilities."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    video_path: str
    filename: str
    width: int
    height: int
    resolution: str
    fps: float
    total_frames: int
    duration_seconds: float
    duration_formatted: str
    fourcc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Probes video file and extracts resolution, fps, frame count, and duration.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        VideoMetadata object with extracted information.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # check for 0 or NaN
        fps = 25.0
        logger.warning(f"Invalid FPS reported by OpenCV. Defaulting to {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    minutes = int(duration_sec // 60)
    seconds = duration_sec % 60
    duration_formatted = f"{minutes}m {seconds:.2f}s"

    meta = VideoMetadata(
        video_path=str(path),
        filename=path.name,
        width=width,
        height=height,
        resolution=f"{width}x{height}",
        fps=float(fps),
        total_frames=total_frames,
        duration_seconds=float(duration_sec),
        duration_formatted=duration_formatted,
        fourcc=fourcc_str,
    )
    return meta


def stream_video_frames(
    video_path: str | Path,
    max_seconds: Optional[float] = None,
    frame_skip: int = 1,
) -> Generator[Tuple[int, float, Any], None, None]:
    """Streams video frames one by one without loading entire video into memory.
    
    Args:
        video_path: Path to video file.
        max_seconds: Optional max seconds to stream (e.g. 60s fast test).
        frame_skip: Skip factor (1 = every frame, 2 = every 2nd frame).
        
    Yields:
        Tuple of (frame_index, timestamp_seconds, frame_bgr_image)
    """
    path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0

    frame_idx = 0
    max_frame_idx = int(max_seconds * fps) if max_seconds is not None else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frame_idx is not None and frame_idx >= max_frame_idx:
                break

            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / fps
                if frame is None or frame.size == 0:
                    logger.warning(f"Frame {frame_idx} is empty or malformed. Skipping.")
                else:
                    yield frame_idx, timestamp, frame

            frame_idx += 1
    finally:
        cap.release()


def extract_sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 5,
) -> List[Path]:
    """Extracts evenly spaced sample frames from the video.
    
    Args:
        video_path: Path to video.
        output_dir: Directory to save sample frames.
        num_samples: Number of sample frames.
        
    Returns:
        List of paths to saved sample frames.
    """
    path = Path(video_path).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    saved_paths: List[Path] = []

    if meta.total_frames <= 0:
        cap.release()
        return saved_paths

    indices = [
        int(i * (meta.total_frames - 1) / max(1, num_samples - 1))
        for i in range(num_samples)
    ]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_file = out_dir / f"sample_frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved_paths.append(out_file)
            logger.info(f"Extracted sample frame {idx} to {out_file}")

    cap.release()
    return saved_paths
