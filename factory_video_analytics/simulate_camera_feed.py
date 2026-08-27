"""Camera feed simulator splitting source video into consecutive 30s clips and dropping them into incoming folder."""

import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import time
import cv2

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulator")


def split_video_into_segments(
    source_video: Path,
    output_dir: Path,
    segment_duration_sec: float = 30.0,
    camera_id: str = "camera_02",
) -> list[Path]:
    """Splits source video into consecutive fixed-duration clips using imageio-ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {source_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    logger.info(f"Source video duration: {duration_sec:.2f}s ({total_frames} frames @ {fps:.2f} FPS)")
    segment_files = []

    current_start = 0.0
    idx = 0

    while current_start < duration_sec:
        # Generate timestamp string: 09:00:00, 09:00:30, etc.
        total_seconds = int(current_start)
        hh = 9 + (total_seconds // 3600)
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60

        dt_str = f"2026-08-20_{hh:02d}-{mm:02d}-{ss:02d}"
        filename = f"{camera_id}_{dt_str}.mp4"
        out_path = output_dir / filename

        dur = min(segment_duration_sec, duration_sec - current_start)
        if dur < 2.0:  # Skip tiny residual fragment
            break

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-ss", f"{current_start:.2f}",
            "-i", str(source_video.resolve()),
            "-t", f"{dur:.2f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_path.resolve()),
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info(f"Generated segment #{idx+1}: {filename} (start: {current_start:.1f}s, dur: {dur:.1f}s)")
        segment_files.append(out_path)

        current_start += segment_duration_sec
        idx += 1

    return segment_files


def run_simulation(
    source_video: str = "../Videos/VID-20260820-WA0003.mp4",
    incoming_dir: str = "data/incoming",
    staging_dir: str = "data/staging",
    camera_id: str = "camera_02",
    delay_between_clips_sec: float = 3.0,
):
    source_path = Path(source_video)
    if not source_path.exists():
        source_path = Path("Videos/VID-20260820-WA0003.mp4")
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found at: {source_video}")

    staging_path = Path(staging_dir)
    incoming_cam_dir = Path(incoming_dir) / camera_id
    incoming_cam_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Splitting video into 30s development segments ===")
    segments = split_video_into_segments(source_path, staging_path, segment_duration_sec=30.0, camera_id=camera_id)

    logger.info(f"=== STEP 2: Feeding {len(segments)} clips into {incoming_cam_dir} ===")
    for i, seg in enumerate(segments):
        target_file = incoming_cam_dir / seg.name
        logger.info(f"Dropping clip #{i+1}/{len(segments)} into incoming folder: {seg.name}")
        shutil.copy2(str(seg), str(target_file))
        time.sleep(delay_between_clips_sec)

    logger.info("=== STEP 3: Testing duplicate rejection ===")
    if segments:
        dup_file = incoming_cam_dir / segments[0].name
        logger.info(f"Dropping duplicate clip: {segments[0].name}")
        shutil.copy2(str(segments[0]), str(dup_file))

    logger.info("Simulation clip ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory Camera Feed Simulator")
    parser.add_argument("--source", type=str, default="../Videos/VID-20260820-WA0003.mp4")
    parser.add_argument("--incoming", type=str, default="data/incoming")
    parser.add_argument("--camera", type=str, default="camera_02")
    parser.add_argument("--delay", type=float, default=2.0)

    args = parser.parse_args()
    run_simulation(
        source_video=args.source,
        incoming_dir=args.incoming,
        camera_id=args.camera,
        delay_between_clips_sec=args.delay,
    )
