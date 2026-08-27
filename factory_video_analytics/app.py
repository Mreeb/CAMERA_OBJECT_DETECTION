"""Main application entry point for Factory Video Analytics Proof of Concept."""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import cv2
import yaml
from tqdm import tqdm

from src.analytics import AnalyticsEngine
from src.detector import YOLOEDetector, class_agnostic_nms
from src.event_writer import OutputWriter
from src.tracker import TrackerManager
from src.video_info import extract_sample_frames, probe_video, stream_video_frames
from src.visualizer import FrameVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("factory_analytics")


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {path}. Using default settings.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_h264_compatibility(video_path: Path) -> None:
    """Converts video to standard H.264 (yuv420p, faststart) using imageio-ffmpeg for universal player compatibility."""
    try:
        import shutil
        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not video_path.exists() or not ffmpeg_exe:
            return

        temp_out = video_path.parent / f"{video_path.stem}_temp_h264.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and temp_out.exists():
            shutil.move(str(temp_out), str(video_path))
            logger.info(f"Successfully converted video to standard H.264: {video_path}")
        else:
            logger.warning(f"H.264 post-processing failed: {res.stderr}")
    except Exception as e:
        logger.warning(f"Could not convert video to H.264: {e}")


def run_pipeline(
    config_path: str = "config.yaml",
    video_path_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    max_seconds: Optional[float] = None,
    device_override: Optional[str] = None,
    preview: bool = False,
    no_video: bool = False,
) -> Dict[str, Any]:
    """Runs the end-to-end video analytics pipeline."""
    # 1. Load configuration
    cfg = load_yaml_config(config_path)

    # Resolve paths
    video_path = Path(video_path_override or cfg.get("video_path", "Videos/VID-20260820-WA0003.mp4"))
    if not video_path.exists():
        # Check relative to parent directory if running from factory_video_analytics/
        if (Path("..") / video_path).exists():
            video_path = Path("..") / video_path
        elif (Path("Videos") / video_path.name).exists():
            video_path = Path("Videos") / video_path.name
        else:
            raise FileNotFoundError(f"Input video not found at: {video_path}")

    base_output_dir = Path(output_dir_override or cfg.get("output_dir", "outputs"))
    video_stem = video_path.stem
    if max_seconds is not None:
        video_out_dir = base_output_dir / f"{video_stem}_{int(max_seconds)}s"
    else:
        video_out_dir = base_output_dir / video_stem

    writer = OutputWriter(video_out_dir)

    # 2. Probe video information
    meta = probe_video(video_path)
    logger.info(
        f"Probed video '{meta.filename}': {meta.resolution} @ {meta.fps:.2f} fps, "
        f"{meta.total_frames} frames ({meta.duration_formatted})"
    )
    writer.write_video_info(meta)

    # Extract initial sample inspection frames
    extract_sample_frames(video_path, video_out_dir / "sample_frames", num_samples=3)

    # 3. Model & Detector Configuration
    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", "yoloe-26s.pt")
    fallback_checkpoint = model_cfg.get("fallback_checkpoint", "yolov8s-worldv2.pt")
    prompts = model_cfg.get("prompts", ["person", "cardboard box", "package", "pallet"])
    conf_thresh = float(model_cfg.get("confidence_threshold", 0.20))
    iou_thresh = float(model_cfg.get("iou_threshold", 0.45))
    class_agnostic_nms_iou = float(model_cfg.get("class_agnostic_nms_iou", 0.60))
    device = device_override or cfg.get("tracker", {}).get("device", "auto")

    logger.info(f"Initializing Open-Vocabulary Detector with prompts: {prompts}")
    detector = YOLOEDetector(
        checkpoint=checkpoint,
        fallback_checkpoint=fallback_checkpoint,
        prompts=prompts,
        confidence_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        class_agnostic_nms_iou=class_agnostic_nms_iou,
        device=device,
    )

    # 4. Tracker & Analytics Configuration
    tracker_cfg = cfg.get("tracker", {})
    tracker_type = tracker_cfg.get("type", "botsort.yaml")
    frame_skip = int(tracker_cfg.get("frame_skip", 1))

    analytics_cfg = cfg.get("analytics", {})
    min_track_dur = float(analytics_cfg.get("min_track_duration_sec", 1.0))
    min_dets = int(analytics_cfg.get("min_detections", 5))
    motion_thresh = float(analytics_cfg.get("motion_threshold_pixels_per_sec", 15.0))
    low_motion_dur = float(analytics_cfg.get("low_motion_duration_sec", 5.0))
    counting_line_cfg = analytics_cfg.get("counting_line", {})

    tracker_mgr = TrackerManager(
        min_track_duration_sec=min_track_dur,
        min_detections=min_dets,
        motion_threshold_pixels_per_sec=motion_thresh,
        low_motion_duration_sec=low_motion_dur,
    )

    analytics_engine = AnalyticsEngine(
        motion_threshold_pixels_per_sec=motion_thresh,
        low_motion_duration_sec=low_motion_dur,
        counting_line_config=counting_line_cfg,
    )

    # 5. Visualizer Configuration
    vis_cfg = cfg.get("visualization", {})
    save_video = not no_video and vis_cfg.get("save_annotated_video", True)
    display_preview = preview or vis_cfg.get("display_preview", False)
    save_sample_frames_flag = vis_cfg.get("save_sample_frames", True)
    sample_frame_interval_sec = float(vis_cfg.get("sample_frame_interval_sec", 10.0))

    visualizer = FrameVisualizer(
        counting_line_config=counting_line_cfg,
        trail_length=int(vis_cfg.get("trail_length", 30)),
        draw_boxes=vis_cfg.get("draw_boxes", True),
        draw_trails=vis_cfg.get("draw_trails", True),
        draw_hud=vis_cfg.get("draw_hud", True),
        draw_counting_line=vis_cfg.get("draw_counting_line", True),
    )

    # 6. Video Writer Setup
    video_writer = None
    annotated_video_path = video_out_dir / "annotated_video.mp4"
    if save_video:
        # Calculate effective output FPS based on frame_skip
        effective_fps = meta.fps / frame_skip
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(annotated_video_path),
            fourcc,
            effective_fps,
            (meta.width, meta.height),
        )
        logger.info(f"Video writer initialized: {annotated_video_path} ({meta.width}x{meta.height} @ {effective_fps:.2f} fps)")

    # 7. Processing Loop
    logger.info("Starting video analytics stream processing...")
    start_time = time.time()
    processed_frames = 0
    last_sample_save_time = -999.0

    # Total frames to process
    total_frames_to_process = meta.total_frames
    if max_seconds is not None:
        total_frames_to_process = min(meta.total_frames, int(max_seconds * meta.fps))
    total_frames_to_process = total_frames_to_process // frame_skip

    pbar = tqdm(
        total=total_frames_to_process,
        desc="Processing Video",
        unit="frames",
        dynamic_ncols=True,
    )

    try:
        frame_stream = stream_video_frames(
            video_path,
            max_seconds=max_seconds,
            frame_skip=frame_skip,
        )

        for frame_idx, timestamp, frame in frame_stream:
            loop_start = time.time()

            # Run tracking with Ultralytics model
            try:
                track_results = detector.model.track(
                    source=frame,
                    persist=True,
                    tracker=tracker_type,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    device=detector.device,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Tracking error at frame {frame_idx}: {e}")
                track_results = []

            # Update TrackerManager
            visible_tracks = tracker_mgr.update_from_ultralytics(
                track_results,
                frame_idx,
                timestamp,
            )

            # Record detection diagnostics stats from raw detections
            if track_results and len(track_results) > 0 and track_results[0].boxes is not None:
                boxes = track_results[0].boxes
                names = track_results[0].names or {}
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for c_id, conf in zip(cls_ids, confs):
                    c_name = str(names.get(c_id, f"class_{c_id}"))
                    detector.total_raw_detections += 1
                    if c_name not in detector.raw_detections_per_class:
                        detector.raw_detections_per_class[c_name] = 0
                        detector.accepted_detections_per_class[c_name] = 0
                        detector.confidence_sum_per_class[c_name] = 0.0
                    detector.raw_detections_per_class[c_name] += 1
                    detector.accepted_detections_per_class[c_name] += 1
                    detector.confidence_sum_per_class[c_name] += float(conf)
                    detector.total_accepted_detections += 1

            # Update Analytics Engine
            analytics_engine.process_frame(visible_tracks, frame_idx, timestamp)

            # Calculate instantaneous processing FPS
            loop_elapsed = time.time() - loop_start
            inst_fps = 1.0 / loop_elapsed if loop_elapsed > 0 else 0.0

            # Render annotations
            annotated_frame = visualizer.draw_frame(
                frame,
                visible_tracks,
                analytics_engine,
                frame_idx,
                timestamp,
                fps_display=inst_fps,
            )

            # Write to video
            if video_writer is not None:
                video_writer.write(annotated_frame)

            # Periodic sample frames
            if save_sample_frames_flag and (timestamp - last_sample_save_time >= sample_frame_interval_sec):
                writer.save_sample_frame(annotated_frame, frame_idx, timestamp, tag="annotated")
                last_sample_save_time = timestamp

            # Live preview
            if display_preview:
                cv2.imshow("Factory Video Analytics - Live Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested termination via preview window.")
                    break

            processed_frames += 1
            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user.")
    finally:
        pbar.close()
        if video_writer is not None:
            video_writer.release()
            logger.info("Video writer released.")
            ensure_h264_compatibility(annotated_video_path)
        if display_preview:
            cv2.destroyAllWindows()

    total_processing_time = time.time() - start_time
    analyzed_duration = (
        max_seconds if max_seconds is not None else meta.duration_seconds
    )

    # 8. Finalize track lifecycles and event log
    analytics_engine.finalize_track_events(tracker_mgr)

    # 9. Write outputs
    logger.info("Generating CSV reports and JSON summaries...")
    writer.write_tracks_csv(tracker_mgr, analytics_engine)
    writer.write_events_csv(analytics_engine.events)
    writer.write_summary(analytics_engine, tracker_mgr, analyzed_duration)

    diagnostics = detector.get_diagnostics()
    writer.write_detection_diagnostics(
        diagnostics,
        tracker_mgr,
        total_processing_time,
        processed_frames,
    )

    # Print Summary Report to Console
    print("\n" + "=" * 60)
    print("FACTORY VIDEO ANALYTICS - RUN SUMMARY")
    print("=" * 60)
    print(f"Output Directory: {video_out_dir.resolve()}")
    print(f"Processed Frames: {processed_frames} in {total_processing_time:.2f}s ({processed_frames / total_processing_time:.2f} FPS)")
    print(f"Total Valid Tracks: {len(tracker_mgr.get_valid_tracks())}")
    print(f"  - People Tracks: {len(tracker_mgr.get_valid_person_tracks())}")
    print(f"  - Article Tracks: {len(tracker_mgr.get_valid_article_tracks())}")
    print(f"Max Simultaneous People: {analytics_engine.max_simultaneous_people}")
    print(f"Line Crossings: A->B: {analytics_engine.line_crossing_counts['A_to_B']}, B->A: {analytics_engine.line_crossing_counts['B_to_A']}")
    print(f"Total Events Generated: {len(analytics_engine.events)}")
    print("=" * 60 + "\n")

    return {
        "output_dir": str(video_out_dir),
        "processed_frames": processed_frames,
        "total_processing_time_sec": total_processing_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Factory Video Analytics Proof of Concept")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--input", type=str, default=None, help="Path to input video file (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for outputs")
    parser.add_argument("--max-seconds", type=float, default=None, help="Process first N seconds of video (fast test)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on: auto, cuda, cpu")
    parser.add_argument("--preview", action="store_true", help="Display live OpenCV preview window")
    parser.add_argument("--no-video", action="store_true", help="Do not save output annotated video")

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        video_path_override=args.input,
        output_dir_override=args.output_dir,
        max_seconds=args.max_seconds,
        device_override=args.device,
        preview=args.preview,
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
