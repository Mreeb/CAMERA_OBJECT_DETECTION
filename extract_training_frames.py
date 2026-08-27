"""Extracts horizontal training frames at 30 frames per minute from all Videos in Videos/ folder."""

import glob
import logging
import os
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frames_from_videos(
    input_dir: str = "Videos",
    output_dir: str = "TRAINING_FRAMES",
    frames_per_minute: float = 30.0,
    jpeg_quality: int = 95,
):
    """
    Extracts frames at a rate of 30 frames/minute (1 frame every 2.0 seconds).
    Automatically detects vertical videos (H > W) and rotates them to horizontal.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find video files
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in extensions:
        video_files.extend(list(in_path.glob(ext)))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        logger.error(f"No video files found in '{input_dir}'")
        return

    logger.info(f"Found {len(video_files)} video(s) in '{input_dir}'")
    logger.info(f"Target extraction rate: {frames_per_minute} frames/min (1 frame every {60.0/frames_per_minute:.1f}s)")
    logger.info(f"Output directory: '{out_path.resolve()}'")

    total_extracted = 0
    extraction_summary = []

    for vid_file in video_files:
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {vid_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Step size in frames (every 2 seconds for 30 frames/min)
        interval_sec = 60.0 / frames_per_minute
        frame_step = max(1, int(round(fps * interval_sec)))

        stem = vid_file.stem
        is_vertical = orig_h > orig_w
        logger.info(
            f"Processing '{vid_file.name}': {orig_w}x{orig_h} | {duration_sec:.1f}s ({duration_sec/60.0:.2f} min) @ {fps:.1f} fps | "
            f"Orientation: {'VERTICAL -> ROTATING HORIZONTAL' if is_vertical else 'HORIZONTAL'}"
        )

        vid_extracted = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                curr_sec = frame_idx / fps
                h, w = frame.shape[:2]

                # Convert vertical frames to horizontal
                if h > w:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    h, w = frame.shape[:2]

                out_filename = f"{stem}_f{frame_idx:05d}_{curr_sec:06.1f}s.jpg"
                out_filepath = out_path / out_filename

                cv2.imwrite(
                    str(out_filepath),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                vid_extracted += 1

            frame_idx += 1

        cap.release()
        total_extracted += vid_extracted
        extraction_summary.append({
            "video": vid_file.name,
            "duration_sec": duration_sec,
            "frames_extracted": vid_extracted,
            "original_resolution": f"{orig_w}x{orig_h}",
            "is_vertical": is_vertical,
        })
        logger.info(f"Extracted {vid_extracted} frames from '{vid_file.name}'")

    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY:")
    print("=" * 65)
    for s in extraction_summary:
        print(
            f" - {s['video']:<26}: {s['frames_extracted']:>4} frames ({s['duration_sec']:.1f}s, "
            f"{'Rotated to Horizontal' if s['is_vertical'] else 'Horizontal'})"
        )
    print("=" * 65)
    print(f"TOTAL FRAMES SAVED TO '{output_dir}': {total_extracted} frames\n")


if __name__ == "__main__":
    extract_frames_from_videos()
