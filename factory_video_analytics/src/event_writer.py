"""Output writers for CSV tables, JSON summaries, diagnostics, and sample frames."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import pandas as pd
from .analytics import AnalyticsEngine, AnalyticsEvent
from .tracker import TrackerManager, TrackState
from .video_info import VideoMetadata

logger = logging.getLogger(__name__)


class OutputWriter:
    """Handles saving all structured outputs and artifacts."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_frames_dir = self.output_dir / "sample_frames"
        self.sample_frames_dir.mkdir(parents=True, exist_ok=True)

    def write_video_info(self, meta: VideoMetadata) -> Path:
        """Writes video_info.json."""
        out_path = self.output_dir / "video_info.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)
        logger.info(f"Saved video info to: {out_path}")
        return out_path

    def write_detection_diagnostics(
        self,
        diagnostics: Dict[str, Any],
        tracker: TrackerManager,
        total_processing_time_sec: float,
        total_frames_processed: int,
    ) -> Path:
        """Writes detection_diagnostics.json."""
        out_path = self.output_dir / "detection_diagnostics.json"

        # Calculate valid tracks per class
        valid_tracks = tracker.get_valid_tracks()
        valid_tracks_per_class: Dict[str, int] = {}
        for t in valid_tracks:
            valid_tracks_per_class[t.class_name] = valid_tracks_per_class.get(t.class_name, 0) + 1

        overall_fps = (
            total_frames_processed / total_processing_time_sec
            if total_processing_time_sec > 0 else 0.0
        )

        full_diagnostics = {
            **diagnostics,
            "valid_tracks_per_class": valid_tracks_per_class,
            "total_valid_tracks": len(valid_tracks),
            "total_frames_processed": total_frames_processed,
            "total_processing_time_sec": round(total_processing_time_sec, 2),
            "overall_pipeline_fps": round(overall_fps, 2),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_diagnostics, f, indent=2)
        logger.info(f"Saved detection diagnostics to: {out_path}")
        return out_path

    def write_summary(
        self,
        analytics: AnalyticsEngine,
        tracker: TrackerManager,
        video_duration: float,
    ) -> Path:
        """Writes summary.json."""
        out_path = self.output_dir / "summary.json"
        summary_data = analytics.generate_summary(tracker, video_duration)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved summary to: {out_path}")
        return out_path

    def write_tracks_csv(
        self,
        tracker: TrackerManager,
        analytics: AnalyticsEngine,
    ) -> Path:
        """Writes tracks.csv (one summarized row per track)."""
        out_path = self.output_dir / "tracks.csv"
        valid_tracks = tracker.get_valid_tracks()

        rows = []
        for t in valid_tracks:
            is_person = t.class_name.lower() in ("person", "worker")
            category = "Person" if is_person else "Article"
            
            motion_stats = analytics.track_motion_stats.get(t.track_id, {})
            moving_time = motion_stats.get("moving_time", 0.0)
            low_motion_time = motion_stats.get("low_motion_time", 0.0)

            meaningful_movement = t.cumulative_distance >= analytics.meaningful_movement_pixels

            rows.append({
                "track_id": t.track_id,
                "class_name": t.class_name,
                "category": category,
                "first_seen_frame": t.first_seen_frame,
                "first_seen_time_sec": round(t.first_seen_time, 2),
                "last_seen_frame": t.last_seen_frame,
                "last_seen_time_sec": round(t.last_seen_time, 2),
                "visible_duration_sec": round(t.duration, 2),
                "total_detections": t.total_detections,
                "cumulative_distance_pixels": round(t.cumulative_distance, 2),
                "net_displacement_pixels": round(t.net_displacement, 2),
                "average_confidence": round(t.average_confidence, 4),
                "moving_time_sec": round(moving_time, 2),
                "low_motion_time_sec": round(low_motion_time, 2),
                "meaningful_movement": meaningful_movement,
                "final_motion_state": t.motion_state,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            # Create empty dataframe with columns
            df = pd.DataFrame(columns=[
                "track_id", "class_name", "category", "first_seen_frame", "first_seen_time_sec",
                "last_seen_frame", "last_seen_time_sec", "visible_duration_sec", "total_detections",
                "cumulative_distance_pixels", "net_displacement_pixels", "average_confidence",
                "moving_time_sec", "low_motion_time_sec", "meaningful_movement", "final_motion_state"
            ])

        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(rows)} summarized tracks to: {out_path}")
        return out_path

    def write_events_csv(self, events: List[AnalyticsEvent]) -> Path:
        """Writes events.csv (timestamped event log)."""
        out_path = self.output_dir / "events.csv"
        rows = [ev.to_dict() for ev in events]

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                "timestamp", "frame_idx", "event_type", "track_id", "class_name", "details"
            ])

        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(rows)} events to: {out_path}")
        return out_path

    def save_sample_frame(
        self,
        frame: Any,
        frame_idx: int,
        timestamp: float,
        tag: str = "frame",
    ) -> Path:
        """Saves an annotated sample frame image."""
        filename = f"{tag}_{frame_idx:05d}_{int(timestamp)}s.jpg"
        out_path = self.sample_frames_dir / filename
        cv2.imwrite(str(out_path), frame)
        return out_path
