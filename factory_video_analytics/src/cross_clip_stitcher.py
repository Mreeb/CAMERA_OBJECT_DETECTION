"""Cross-clip boundary state stitcher for maintaining track continuity across adjacent video clips."""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BoundaryTrack:
    session_id: str
    class_name: str
    is_person: bool
    box: Tuple[float, float, float, float]
    anchor: Tuple[float, float]
    initial_anchor: Tuple[float, float]
    motion_state: str
    current_zone: str
    initial_zone: str
    last_seen_time: float
    duration_so_far: float
    cumulative_distance: float
    net_displacement: float


@dataclass
class CameraBoundaryState:
    camera_id: str
    last_clip_end_time: float = 0.0
    active_boundary_tracks: Dict[str, BoundaryTrack] = field(default_factory=dict)
    person_counter: int = 0
    article_counter: int = 0


class CrossClipStitcher:
    """Maintains cross-clip tracking continuity and conservative session stitching."""

    def __init__(
        self,
        max_gap_seconds: float = 5.0,
        spatial_reid_distance_px: float = 120.0,
        ambiguous_distance_px: float = 200.0,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.spatial_reid_distance_px = spatial_reid_distance_px
        self.ambiguous_distance_px = ambiguous_distance_px
        self.camera_states: Dict[str, CameraBoundaryState] = {}

    def get_or_create_camera_state(self, camera_id: str) -> CameraBoundaryState:
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraBoundaryState(camera_id=camera_id)
        return self.camera_states[camera_id]

    def match_initial_tracks(
        self,
        camera_id: str,
        clip_start_time: float,
        initial_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Matches detections in the first few frames of a new clip against trailing boundary tracks."""
        state = self.get_or_create_camera_state(camera_id)
        time_gap = clip_start_time - state.last_clip_end_time

        matches = []
        used_boundary_sessions = set()

        is_adjacent = (0.0 <= time_gap <= self.max_gap_seconds) and (state.last_clip_end_time > 0)
        if not is_adjacent and state.last_clip_end_time > 0:
            logger.info(f"Time gap between clips for {camera_id} is {time_gap:.1f}s (> {self.max_gap_seconds}s). Resetting active boundary tracks.")
            state.active_boundary_tracks.clear()

        for det in initial_detections:
            det_anchor = det["anchor"]
            det_is_person = det["is_person"]
            det_class = det["class_name"]

            matched_boundary_track: Optional[BoundaryTrack] = None
            possible_continuation_id: Optional[str] = None
            min_dist = float("inf")

            if is_adjacent:
                for sid, b_track in state.active_boundary_tracks.items():
                    if sid in used_boundary_sessions:
                        continue
                    if b_track.is_person != det_is_person:
                        continue

                    dist = math.hypot(det_anchor[0] - b_track.anchor[0], det_anchor[1] - b_track.anchor[1])
                    if dist <= self.spatial_reid_distance_px and dist < min_dist:
                        min_dist = dist
                        matched_boundary_track = b_track
                    elif dist <= self.ambiguous_distance_px:
                        possible_continuation_id = sid

            if matched_boundary_track is not None:
                used_boundary_sessions.add(matched_boundary_track.session_id)
                logger.info(
                    f"Successfully stitched cross-clip track for {camera_id}: "
                    f"Session {matched_boundary_track.session_id} (gap: {time_gap:.2f}s, dist: {min_dist:.1f}px)"
                )
                matches.append({
                    "detection": det,
                    "session_id": matched_boundary_track.session_id,
                    "is_continued": True,
                    "initial_anchor": matched_boundary_track.initial_anchor,
                    "duration_offset": matched_boundary_track.duration_so_far,
                    "distance_offset": matched_boundary_track.cumulative_distance,
                    "possible_continuation_id": None,
                })
            else:
                # Assign new sequential session ID
                if det_is_person:
                    state.person_counter += 1
                    sid = f"P-{state.person_counter:02d}"
                else:
                    state.article_counter += 1
                    sid = f"B-{state.article_counter:02d}"

                matches.append({
                    "detection": det,
                    "session_id": sid,
                    "is_continued": False,
                    "initial_anchor": det_anchor,
                    "duration_offset": 0.0,
                    "distance_offset": 0.0,
                    "possible_continuation_id": possible_continuation_id,
                })

        return matches

    def update_boundary_state(
        self,
        camera_id: str,
        clip_end_time: float,
        final_active_tracks: List[Dict[str, Any]],
        person_counter: int,
        article_counter: int,
    ) -> None:
        """Saves trailing state of active tracks at the end of a clip."""
        state = self.get_or_create_camera_state(camera_id)
        state.last_clip_end_time = clip_end_time
        state.person_counter = max(state.person_counter, person_counter)
        state.article_counter = max(state.article_counter, article_counter)
        state.active_boundary_tracks.clear()

        for t in final_active_tracks:
            b_track = BoundaryTrack(
                session_id=t["session_id"],
                class_name=t["class_name"],
                is_person=t["is_person"],
                box=t["box"],
                anchor=t["anchor"],
                initial_anchor=t.get("initial_anchor", t["anchor"]),
                motion_state=t["motion_state"],
                current_zone=t.get("current_zone", "Unknown"),
                initial_zone=t.get("initial_zone", "Unknown"),
                last_seen_time=t["last_seen_time"],
                duration_so_far=t.get("duration_seconds", 0.0),
                cumulative_distance=t.get("cumulative_distance_px", 0.0),
                net_displacement=t.get("net_displacement_px", 0.0),
            )
            state.active_boundary_tracks[t["session_id"]] = b_track

        logger.info(f"Saved {len(state.active_boundary_tracks)} boundary tracks for {camera_id} at {clip_end_time:.1f}s")
