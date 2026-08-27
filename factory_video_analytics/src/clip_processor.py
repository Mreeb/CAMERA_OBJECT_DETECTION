"""Core clip processor for scheduled batch analytics, cross-clip tracking, and evidence saving."""

from datetime import datetime
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .cross_clip_stitcher import CrossClipStitcher
from .database import DatabaseManager
from .detector import YOLOEDetector
from .evidence_manager import EvidenceManager
from .state_estimator import MovementConfig
from .tracker import ConfirmedTrack, SessionTracker
from .zone_engine import ZoneEngine

logger = logging.getLogger(__name__)


class ClipProcessor:
    """Processes individual video clips, maintaining cross-clip state and extracting evidence."""

    def __init__(
        self,
        detector: YOLOEDetector,
        database: DatabaseManager,
        cross_clip_stitcher: CrossClipStitcher,
        evidence_manager: EvidenceManager,
        zones_config: List[Dict[str, Any]],
        counting_line_config: Dict[str, Any],
        movement_config: Dict[str, Any],
        worker_activity_config: Dict[str, Any],
        completed_base_dir: str | Path = "data/completed",
        failed_base_dir: str | Path = "data/failed",
        inference_fps: float = 8.0,
        archive_completed: bool = True,
    ):
        self.detector = detector
        self.database = database
        self.stitcher = cross_clip_stitcher
        self.evidence_mgr = evidence_manager
        self.inference_fps = inference_fps
        self.archive_completed = archive_completed

        self.completed_base_dir = Path(completed_base_dir)
        self.failed_base_dir = Path(failed_base_dir)
        self.completed_base_dir.mkdir(parents=True, exist_ok=True)
        self.failed_base_dir.mkdir(parents=True, exist_ok=True)

        self.movement_cfg = MovementConfig(
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

        self.zone_engine = ZoneEngine(zones_config, counting_line_config)

    def process_clip(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single clip job end-to-end."""
        job_id = job_data["job_id"]
        clip_id = job_data["clip_id"]
        camera_id = job_data["camera_id"]
        location_id = job_data.get("location_id", "default")
        file_path = Path(job_data["file_path"])
        clip_start_time = float(job_data["start_time"])
        clip_duration = float(job_data["duration_sec"])

        logger.info(f"Starting batch analysis for Clip #{clip_id} ({file_path.name}) on {camera_id}")
        start_processing_wall = time.time()

        if not file_path.exists():
            raise FileNotFoundError(f"Clip file not found: {file_path}")

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open clip: {file_path}")

        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(video_fps / self.inference_fps)))

        # Initialize SessionTracker for this clip
        tracker = SessionTracker(
            lost_grace_period_sec=3.0,
            min_confirmed_frames=3,
            spatial_reid_distance_px=120.0,
            movement_config=self.movement_cfg,
        )

        # Sync counter from stitcher
        cam_state = self.stitcher.get_or_create_camera_state(camera_id)
        tracker.person_counter = cam_state.person_counter
        tracker.article_counter = cam_state.article_counter

        # Metrics accumulators
        sampled_people_counts: List[int] = []
        person_time_moving = 0.0
        person_time_low_motion = 0.0
        person_time_ext_low = 0.0
        person_time_unknown = 0.0
        person_time_total = 0.0

        prev_track_states: Dict[str, str] = {}
        prev_track_zones: Dict[str, str] = {}
        logged_events: List[Dict[str, Any]] = []

        confirmed_transfers_count = 0
        frame_idx = 0
        is_first_inference_batch = True

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                h, w = frame.shape[:2]
                current_clip_ts = frame_idx / video_fps
                global_ts = clip_start_time + current_clip_ts
                dt_sample = frame_interval / video_fps

                # 1. Open-Vocabulary Detection & Filtering
                detections = self.detector.detect_and_filter(frame, self.zone_engine)

                # 2. First inference batch cross-clip stitch
                if is_first_inference_batch and detections:
                    initial_det_payloads = [
                        {
                            "anchor": ((d.box[0] + d.box[2]) / 2.0, d.box[3] if d.is_person else (d.box[1] + d.box[3]) / 2.0),
                            "is_person": d.is_person,
                            "class_name": d.class_name,
                        }
                        for d in detections
                    ]
                    # Attempt stitching
                    stitched_results = self.stitcher.match_initial_tracks(
                        camera_id=camera_id,
                        clip_start_time=clip_start_time,
                        initial_detections=initial_det_payloads,
                    )
                    is_first_inference_batch = False

                # 3. Update Tracker
                active_tracks = tracker.update(
                    detections=detections,
                    timestamp=global_ts,
                    zone_engine=self.zone_engine,
                    frame_w=w,
                    frame_h=h,
                )

                # 4. Sample people counts & person-time
                visible_people = [t for t in active_tracks if t.is_person and t.is_visible]
                sampled_people_counts.append(len(visible_people))

                for p in visible_people:
                    person_time_total += dt_sample
                    state = p.motion_state
                    if state == "Moving":
                        person_time_moving += dt_sample
                    elif state == "Low Motion":
                        person_time_low_motion += dt_sample
                    elif state == "Extended Low Motion":
                        person_time_ext_low += dt_sample
                    else:
                        person_time_unknown += dt_sample

                # 5. Check Events & State Transitions
                for track in active_tracks:
                    sid = track.session_id
                    curr_state = track.motion_state
                    curr_zone = track.current_zone
                    prev_state = prev_track_states.get(sid)
                    prev_zone = prev_track_zones.get(sid)

                    # A. Motion state change
                    if prev_state is not None and prev_state != curr_state:
                        event_type = f"{curr_state.lower().replace(' ', '_')}_detected"
                        snap_path = self.evidence_mgr.save_snapshot(frame, camera_id, sid, event_type, global_ts, clip_id)
                        
                        clip_path = None
                        if curr_state == "Extended Low Motion":
                            clip_path = self.evidence_mgr.extract_evidence_clip(
                                file_path, current_clip_ts, camera_id, sid, event_type
                            )

                        event_payload = {
                            "clip_id": clip_id,
                            "camera_id": camera_id,
                            "session_id": sid,
                            "track_id": track.internal_track_id,
                            "class_name": track.class_name,
                            "timestamp": global_ts,
                            "event_type": event_type,
                            "state": curr_state,
                            "zone": curr_zone,
                            "details": f"Activity changed from {prev_state} to {curr_state}",
                            "snapshot_path": snap_path,
                            "evidence_clip_path": clip_path,
                        }
                        self.database.insert_event(event_payload)
                        logged_events.append(event_payload)

                    # B. Zone transition / Article Transfer
                    if prev_zone is not None and prev_zone != curr_zone and curr_zone != "Unknown":
                        event_type = "zone_transition"
                        is_transfer = (prev_zone == "Storage Area" and curr_zone == "Vehicle Loading Area")
                        if is_transfer:
                            confirmed_transfers_count += 1
                            event_type = "article_transferred"

                        snap_path = self.evidence_mgr.save_snapshot(frame, camera_id, sid, event_type, global_ts, clip_id)
                        clip_path = self.evidence_mgr.extract_evidence_clip(
                            file_path, current_clip_ts, camera_id, sid, event_type
                        ) if is_transfer else None

                        event_payload = {
                            "clip_id": clip_id,
                            "camera_id": camera_id,
                            "session_id": sid,
                            "track_id": track.internal_track_id,
                            "class_name": track.class_name,
                            "timestamp": global_ts,
                            "event_type": event_type,
                            "state": curr_state,
                            "zone": curr_zone,
                            "details": f"Transitioned from {prev_zone} to {curr_zone}",
                            "snapshot_path": snap_path,
                            "evidence_clip_path": clip_path,
                        }
                        self.database.insert_event(event_payload)
                        logged_events.append(event_payload)

                    # C. Line crossing
                    if len(track.smoother.history) >= 2:
                        prev_pos = (track.smoother.history[-2][1], track.smoother.history[-2][2])
                        curr_pos = track.bottom_center
                        direction = self.zone_engine.check_line_crossing(
                            prev_pos, curr_pos, track.class_name, track.internal_track_id, global_ts, w, h
                        )
                        if direction:
                            snap_path = self.evidence_mgr.save_snapshot(frame, camera_id, sid, "line_crossing", global_ts, clip_id)
                            clip_path = self.evidence_mgr.extract_evidence_clip(
                                file_path, current_clip_ts, camera_id, sid, "line_crossing"
                            )
                            event_payload = {
                                "clip_id": clip_id,
                                "camera_id": camera_id,
                                "session_id": sid,
                                "track_id": track.internal_track_id,
                                "class_name": track.class_name,
                                "timestamp": global_ts,
                                "event_type": "line_crossing",
                                "state": curr_state,
                                "zone": curr_zone,
                                "details": f"Crossed counting line in direction {direction}",
                                "snapshot_path": snap_path,
                                "evidence_clip_path": clip_path,
                            }
                            self.database.insert_event(event_payload)
                            logged_events.append(event_payload)

                    prev_track_states[sid] = curr_state
                    prev_track_zones[sid] = curr_zone

            frame_idx += 1

        cap.release()

        # 6. Save final boundary state for cross-clip continuity
        all_tracks = tracker.get_all_active_tracks()
        final_boundary_tracks = [
            {
                "session_id": t.session_id,
                "class_name": t.class_name,
                "is_person": t.is_person,
                "box": t.box,
                "anchor": t.bottom_center,
                "initial_anchor": t.smoother.initial_anchor,
                "motion_state": t.motion_state,
                "current_zone": t.current_zone,
                "initial_zone": t.smoother.initial_zone,
                "last_seen_time": t.last_seen_time,
                "duration_seconds": t.duration,
                "cumulative_distance_px": t.smoother.cumulative_distance,
                "net_displacement_px": t.smoother.net_displacement,
            }
            for t in all_tracks
            if t.is_visible
        ]

        self.stitcher.update_boundary_state(
            camera_id=camera_id,
            clip_end_time=clip_start_time + clip_duration,
            final_active_tracks=final_boundary_tracks,
            person_counter=tracker.person_counter,
            article_counter=tracker.article_counter,
        )

        # 7. Store Track Sessions in SQLite
        session_records = [
            {
                "session_id": t.session_id,
                "internal_id": t.internal_track_id,
                "class_name": t.class_name,
                "is_person": t.is_person,
                "first_seen": t.first_seen_time,
                "last_seen": t.last_seen_time,
                "duration_sec": t.duration,
                "max_motion_state": t.motion_state,
                "initial_zone": t.smoother.initial_zone,
                "final_zone": t.current_zone,
                "net_displacement_px": t.smoother.net_displacement,
                "total_distance_px": t.smoother.cumulative_distance,
                "is_continued_from_prev": False,
                "possible_continuation_id": None,
            }
            for t in all_tracks
        ]
        self.database.insert_track_sessions(clip_id, camera_id, session_records)

        # 8. Compute and insert Clip Summary
        max_people = max(sampled_people_counts) if sampled_people_counts else 0
        avg_people = (sum(sampled_people_counts) / len(sampled_people_counts)) if sampled_people_counts else 0.0

        all_articles = [t for t in all_tracks if not t.is_person]
        stationary_articles = len([a for a in all_articles if a.smoother.article_movement_state == "visible_stationary"])
        moved_articles = len([a for a in all_articles if a.smoother.article_movement_state in ("moving_confirmed", "stopped_after_movement")])

        health = tracker.get_health_metrics(clip_start_time + clip_duration)
        frag_rate = health["new_ids_last_minute"]

        clip_summary = {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "location_id": location_id,
            "start_time": clip_start_time,
            "end_time": clip_start_time + clip_duration,
            "duration_sec": clip_duration,
            "max_simultaneous_people": max_people,
            "avg_simultaneous_people": round(avg_people, 2),
            "total_person_seconds": round(person_time_total, 1),
            "moving_person_seconds": round(person_time_moving, 1),
            "low_motion_person_seconds": round(person_time_low_motion, 1),
            "ext_low_motion_person_seconds": round(person_time_ext_low, 1),
            "unknown_person_seconds": round(person_time_unknown, 1),
            "total_articles_seen": len(all_articles),
            "stationary_articles": stationary_articles,
            "moved_articles": moved_articles,
            "transferred_articles": confirmed_transfers_count,
            "line_crossings_a_to_b": self.zone_engine.line_crossings["A_to_B"],
            "line_crossings_b_to_a": self.zone_engine.line_crossings["B_to_A"],
            "new_tracker_sessions": len(all_tracks),
            "fragmentation_rate": frag_rate,
        }
        self.database.insert_clip_summary(clip_summary)

        # 9. Measure Speed & Processing Ratio
        processing_wall_time = time.time() - start_processing_wall
        speed_ratio = clip_duration / max(0.01, processing_wall_time)
        self.database.mark_job_completed(job_id, clip_id, processing_wall_time, speed_ratio)

        # 10. Archive processed clip to completed directory if enabled
        if self.archive_completed:
            target_dir = self.completed_base_dir / camera_id
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / file_path.name
            if file_path.exists() and file_path != dest_path:
                shutil.move(str(file_path), str(dest_path))

        logger.info(
            f"Completed Clip #{clip_id} in {processing_wall_time:.2f}s "
            f"(Speed ratio: {speed_ratio:.1f}x real-time). Max people: {max_people}, Transfers: {confirmed_transfers_count}"
        )

        return clip_summary
