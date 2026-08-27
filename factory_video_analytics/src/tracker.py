"""High-stability Worker Session Tracker with IoU matching, duplicate track suppression, and clean ID recycling."""

from collections import deque
from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import numpy as np

from .detector import Detection
from .state_estimator import MovementConfig, TrackSmoother
from .zone_engine import ZoneEngine

logger = logging.getLogger(__name__)


def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0.0 else 0.0


@dataclass
class ConfirmedTrack:
    """Represents a confirmed worker track with a stable human-facing session ID."""
    session_id: str             # "W-01", "W-02", etc.
    internal_track_id: int
    class_name: str
    is_person: bool
    confidence: float
    box: Tuple[float, float, float, float]
    center: Tuple[float, float]
    
    first_seen_time: float
    last_seen_time: float
    
    # State Smoother
    smoother: TrackSmoother
    
    # Track status
    is_visible: bool = True
    is_confirmed: bool = False
    confirmation_count: int = 1
    
    # Missing tracking
    missing_duration: float = 0.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return self.smoother.smooth_anchor

    @property
    def motion_state(self) -> str:
        return self.smoother.motion_state

    @property
    def current_zone(self) -> str:
        return self.smoother.current_zone

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen_time - self.first_seen_time)


class SessionTracker:
    """Manages worker tracking continuity with IoU matching, duplicate suppression, and stable IDs."""

    def __init__(
        self,
        lost_grace_period_sec: float = 2.0,
        min_confirmed_frames: int = 2,
        spatial_reid_distance_px: float = 140.0,
        movement_config: Optional[MovementConfig] = None,
    ):
        self.lost_grace_period_sec = lost_grace_period_sec
        self.min_confirmed_frames = min_confirmed_frames
        self.spatial_reid_distance_px = spatial_reid_distance_px
        self.movement_config = movement_config or MovementConfig()

        # Track Storage
        self.active_tracks: Dict[str, ConfirmedTrack] = {}
        self.next_internal_id = 1

        # Available ID Pool for clean Human-Facing numbers (W-01, W-02, ...)
        self.used_id_numbers: Set[int] = set()

        # Health & Diagnostics
        self.max_simultaneous_people = 0
        self.total_people_sessions = 0

    def _allocate_session_id(self) -> str:
        """Allocates the smallest available human-facing worker ID (e.g. W-01, W-02)."""
        num = 1
        while num in self.used_id_numbers:
            num += 1
        self.used_id_numbers.add(num)
        self.total_people_sessions += 1
        return f"W-{num:02d}"

    def _release_session_id(self, sid: str) -> None:
        """Releases an ID back to the pool when a track permanently expires."""
        try:
            if sid.startswith("W-"):
                num = int(sid.split("-")[1])
                self.used_id_numbers.discard(num)
        except Exception:
            pass

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        zone_engine: Optional[ZoneEngine] = None,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> List[ConfirmedTrack]:
        """Matches person detections with active tracks using IoU & Center distance, suppressing duplicates."""
        # 0. Raw Detection NMS: Suppress redundant bounding boxes within the frame
        filtered_detections: List[Detection] = []
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for det in sorted_dets:
            overlap = False
            for kept in filtered_detections:
                if compute_box_iou(det.box, kept.box) > 0.45:
                    overlap = True
                    break
            if not overlap:
                filtered_detections.append(det)

        matched_session_ids: Set[str] = set()
        unmatched_detections: List[Detection] = []

        # 1. Matching: Build match matrix with IoU and normalized center distance
        # Prioritize active visible tracks, then recently lost tracks
        candidate_tracks = list(self.active_tracks.values())

        for det in filtered_detections:
            best_sid = None
            best_score = -1.0  # Higher is better

            for track in candidate_tracks:
                if track.session_id in matched_session_ids:
                    continue

                # Compute IoU
                iou = compute_box_iou(det.box, track.box)

                # Compute Center-to-Center Distance (NOT center to bottom_center!)
                center_dist = math.hypot(det.center[0] - track.center[0], det.center[1] - track.center[1])
                diag = max(30.0, math.hypot(det.width, det.height))
                norm_dist = center_dist / diag

                # Match criteria
                is_match = False
                score = 0.0

                if iou > 0.20:
                    is_match = True
                    score = iou + 1.0  # High priority for IoU
                elif center_dist < min(self.spatial_reid_distance_px, diag * 0.90) and norm_dist < 0.75:
                    is_match = True
                    score = 1.0 - norm_dist

                if is_match and score > best_score:
                    best_score = score
                    best_sid = track.session_id

            if best_sid is not None:
                matched_session_ids.add(best_sid)
                track = self.active_tracks[best_sid]

                # Determine current zone
                curr_zone = "Unknown"
                if zone_engine:
                    z = zone_engine.get_zone_for_point(det.center, frame_w, frame_h)
                    if z:
                        curr_zone = z.name

                track.box = det.box
                track.center = det.center
                track.confidence = det.confidence
                track.last_seen_time = timestamp
                track.is_visible = True
                track.missing_duration = 0.0
                track.confirmation_count += 1
                if track.confirmation_count >= self.min_confirmed_frames:
                    track.is_confirmed = True

                track.smoother.update(det.box, timestamp, current_zone=curr_zone)
            else:
                unmatched_detections.append(det)

        # 2. Spawn new tracks for truly unmatched detections
        for det in unmatched_detections:
            # Check if there is an existing confirmed track already overlapping this area
            duplicate_area = False
            for sid in matched_session_ids:
                if compute_box_iou(det.box, self.active_tracks[sid].box) > 0.35:
                    duplicate_area = True
                    break
            if duplicate_area:
                continue

            sid = self._allocate_session_id()
            matched_session_ids.add(sid)

            curr_zone = "Unknown"
            if zone_engine:
                z = zone_engine.get_zone_for_point(det.center, frame_w, frame_h)
                if z:
                    curr_zone = z.name

            smoother = TrackSmoother(
                track_id=self.next_internal_id,
                class_name="person",
                initial_box=det.box,
                start_time=timestamp,
                config=self.movement_config,
            )
            smoother.current_zone = curr_zone

            track = ConfirmedTrack(
                session_id=sid,
                internal_track_id=self.next_internal_id,
                class_name="person",
                is_person=True,
                confidence=det.confidence,
                box=det.box,
                center=det.center,
                first_seen_time=timestamp,
                last_seen_time=timestamp,
                smoother=smoother,
                is_visible=True,
                is_confirmed=(self.min_confirmed_frames <= 1),
                confirmation_count=1,
            )
            self.next_internal_id += 1
            self.active_tracks[sid] = track

        # 3. Update visibility and age out lost tracks
        dead_tracks = []
        for sid, track in list(self.active_tracks.items()):
            if sid not in matched_session_ids:
                track.is_visible = False
                track.missing_duration = timestamp - track.last_seen_time

                if track.missing_duration > self.lost_grace_period_sec:
                    dead_tracks.append(sid)

        for sid in dead_tracks:
            self._release_session_id(sid)
            del self.active_tracks[sid]

        # 4. Duplicate Track Suppression (Post-NMS on Active Visible Tracks)
        visible_tracks = [t for t in self.active_tracks.values() if t.is_visible]
        # Sort by confirmation count & confidence descending
        visible_tracks.sort(key=lambda t: (t.confirmation_count, t.confidence), reverse=True)

        final_tracks: List[ConfirmedTrack] = []
        for t in visible_tracks:
            has_overlap = False
            for kept in final_tracks:
                if compute_box_iou(t.box, kept.box) > 0.35:
                    has_overlap = True
                    break
            if not has_overlap:
                final_tracks.append(t)
            else:
                # Mark as duplicate invisible
                t.is_visible = False

        confirmed_visible = [t for t in final_tracks if t.is_confirmed]
        self.max_simultaneous_people = max(self.max_simultaneous_people, len(confirmed_visible))

        return confirmed_visible
