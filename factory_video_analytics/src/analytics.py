"""Rule-based video analytics for motion, people metrics, articles, and counting line."""

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from .tracker import TrackState, TrackerManager

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Represents a timestamped domain event."""
    timestamp: float
    frame_idx: int
    event_type: str
    track_id: int
    class_name: str
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "frame_idx": self.frame_idx,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "details": self.details,
        }


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Returns True if points A, B, C are in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Returns True if line segment p1-p2 intersects segment p3-p4."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_side_of_line(
    p: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Returns cross product indicating which side of directed line segment point p lies on.
    
    Positive (> 0): Left side (Side A)
    Negative (< 0): Right side (Side B)
    """
    return (line_end[0] - line_start[0]) * (p[1] - line_start[1]) - (line_end[1] - line_start[1]) * (p[0] - line_start[0])


class AnalyticsEngine:
    """Computes people metrics, article metrics, motion states, and line crossings."""

    def __init__(
        self,
        motion_threshold_pixels_per_sec: float = 15.0,
        low_motion_duration_sec: float = 5.0,
        meaningful_movement_pixels: float = 30.0,
        counting_line_config: Optional[Dict[str, Any]] = None,
    ):
        self.motion_threshold_pixels_per_sec = motion_threshold_pixels_per_sec
        self.low_motion_duration_sec = low_motion_duration_sec
        self.meaningful_movement_pixels = meaningful_movement_pixels
        self.counting_line_config = counting_line_config or {}

        # Counting line params
        self.line_enabled = self.counting_line_config.get("enabled", False)
        self.line_p1 = tuple(self.counting_line_config.get("point1", [0, 0]))
        self.line_p2 = tuple(self.counting_line_config.get("point2", [0, 0]))
        self.line_classes = set(
            c.lower() for c in self.counting_line_config.get("classes", [])
        )
        self.count_people = self.counting_line_config.get("count_people", False)
        self.debounce_frames = self.counting_line_config.get("debounce_frames", 15)

        # Global event log
        self.events: List[AnalyticsEvent] = []

        # Real-time metrics
        self.max_simultaneous_people: int = 0
        self.current_visible_people: int = 0
        self.current_visible_articles: int = 0
        self.line_crossing_counts: Dict[str, int] = {"A_to_B": 0, "B_to_A": 0}

        # Track-specific motion tracking
        self.track_motion_stats: Dict[int, Dict[str, Any]] = {}

    def process_frame(
        self,
        visible_tracks: List[TrackState],
        frame_idx: int,
        timestamp: float,
    ) -> None:
        """Processes analytics for all visible tracks in the current frame."""
        people_in_frame = 0
        articles_in_frame = 0

        for track in visible_tracks:
            is_person = track.class_name.lower() in ("person", "worker")
            if is_person:
                people_in_frame += 1
            else:
                articles_in_frame += 1

            # Initialize track motion stats if new
            if track.track_id not in self.track_motion_stats:
                self.track_motion_stats[track.track_id] = {
                    "moving_time": 0.0,
                    "low_motion_time": 0.0,
                    "last_timestamp": timestamp,
                    "has_started_moving": False,
                }
                self.events.append(
                    AnalyticsEvent(
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        event_type="track_started",
                        track_id=track.track_id,
                        class_name=track.class_name,
                        details=f"Track initialized at ({int(track.center[0])}, {int(track.center[1])})",
                    )
                )

            # Update motion classification
            stats = self.track_motion_stats[track.track_id]
            dt = max(0.0, timestamp - stats["last_timestamp"])
            stats["last_timestamp"] = timestamp

            velocity = track.estimate_velocity()
            is_moving = velocity >= self.motion_threshold_pixels_per_sec

            if is_moving:
                stats["moving_time"] += dt
                if not stats["has_started_moving"] and track.cumulative_distance > 10.0:
                    stats["has_started_moving"] = True
                    self.events.append(
                        AnalyticsEvent(
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            event_type="object_started_moving",
                            track_id=track.track_id,
                            class_name=track.class_name,
                            details=f"Velocity {velocity:.1f} px/s, cumulative dist {track.cumulative_distance:.1f} px",
                        )
                    )

                if track.motion_state in ("low_motion", "extended_low_motion"):
                    track.motion_state = "moving"
                    track.low_motion_start_time = None
                    self.events.append(
                        AnalyticsEvent(
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            event_type="low_motion_ended",
                            track_id=track.track_id,
                            class_name=track.class_name,
                            details="Motion resumed",
                        )
                    )
                else:
                    track.motion_state = "moving"

            else:
                stats["low_motion_time"] += dt
                if track.motion_state != "low_motion" and track.motion_state != "extended_low_motion":
                    track.motion_state = "low_motion"
                    track.low_motion_start_time = timestamp
                    self.events.append(
                        AnalyticsEvent(
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            event_type="low_motion_started",
                            track_id=track.track_id,
                            class_name=track.class_name,
                            details=f"Low motion detected (velocity {velocity:.1f} px/s)",
                        )
                    )
                elif track.low_motion_start_time is not None:
                    low_motion_dur = timestamp - track.low_motion_start_time
                    if (
                        low_motion_dur >= self.low_motion_duration_sec
                        and not track.extended_low_motion_logged
                    ):
                        track.motion_state = "extended_low_motion"
                        track.extended_low_motion_logged = True
                        self.events.append(
                            AnalyticsEvent(
                                timestamp=timestamp,
                                frame_idx=frame_idx,
                                event_type="extended_low_motion",
                                track_id=track.track_id,
                                class_name=track.class_name,
                                details=f"Extended low motion exceeded {self.low_motion_duration_sec:.1f}s",
                            )
                        )

            # Check virtual counting line
            if self.line_enabled:
                self._check_line_crossing(track, frame_idx, timestamp)

        self.current_visible_people = people_in_frame
        self.current_visible_articles = articles_in_frame
        if people_in_frame > self.max_simultaneous_people:
            self.max_simultaneous_people = people_in_frame

    def _check_line_crossing(
        self,
        track: TrackState,
        frame_idx: int,
        timestamp: float,
    ) -> None:
        """Checks if a track's center crossed the virtual counting line."""
        is_person = track.class_name.lower() in ("person", "worker")
        if is_person and not self.count_people:
            return

        if not is_person and self.line_classes and track.class_name.lower() not in self.line_classes:
            return

        if len(track.recent_history) < 2:
            return

        # Check debounce
        if (frame_idx - track.last_crossing_frame) < self.debounce_frames:
            return

        prev_pos = (track.recent_history[-2][1], track.recent_history[-2][2])
        curr_pos = track.center

        # Check line segment intersection
        if segments_intersect(prev_pos, curr_pos, self.line_p1, self.line_p2):
            side_prev = point_side_of_line(prev_pos, self.line_p1, self.line_p2)
            side_curr = point_side_of_line(curr_pos, self.line_p1, self.line_p2)

            direction = "A_to_B" if side_prev > 0 and side_curr <= 0 else "B_to_A"
            self.line_crossing_counts[direction] += 1
            track.last_crossing_frame = frame_idx

            self.events.append(
                AnalyticsEvent(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    event_type="line_crossing",
                    track_id=track.track_id,
                    class_name=track.class_name,
                    details=f"Direction: {direction}, crossing at ({int(curr_pos[0])}, {int(curr_pos[1])})",
                )
            )
            logger.info(
                f"[Line Crossing] Track {track.track_id} ({track.class_name}) crossed line in direction {direction} at {timestamp:.2f}s"
            )

    def finalize_track_events(self, tracker: TrackerManager) -> None:
        """Adds track_ended events for completed tracks at the end of the video."""
        for track in tracker.all_tracks.values():
            if track.duration >= tracker.min_track_duration_sec and track.total_detections >= tracker.min_detections:
                self.events.append(
                    AnalyticsEvent(
                        timestamp=track.last_seen_time,
                        frame_idx=track.last_seen_frame,
                        event_type="track_ended",
                        track_id=track.track_id,
                        class_name=track.class_name,
                        details=f"Track completed. Duration: {track.duration:.2f}s, distance: {track.cumulative_distance:.1f}px",
                    )
                )

        # Sort all events chronologically
        self.events.sort(key=lambda e: (e.timestamp, e.frame_idx))

    def generate_summary(self, tracker: TrackerManager, total_video_duration: float) -> Dict[str, Any]:
        """Generates comprehensive high-level analytics summary."""
        valid_tracks = tracker.get_valid_tracks()
        person_tracks = tracker.get_valid_person_tracks()
        article_tracks = tracker.get_valid_article_tracks()

        # People metrics
        people_durations = [t.duration for t in person_tracks]
        people_distances = [t.cumulative_distance for t in person_tracks]

        people_summary = {
            "unique_person_tracks": len(person_tracks),
            "max_simultaneous_visible_people": self.max_simultaneous_people,
            "average_visible_duration_sec": round(sum(people_durations) / len(people_durations), 2) if people_durations else 0.0,
            "average_movement_distance_pixels": round(sum(people_distances) / len(people_distances), 2) if people_distances else 0.0,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "first_seen_time_sec": round(t.first_seen_time, 2),
                    "last_seen_time_sec": round(t.last_seen_time, 2),
                    "visible_duration_sec": round(t.duration, 2),
                    "cumulative_distance_pixels": round(t.cumulative_distance, 2),
                    "net_displacement_pixels": round(t.net_displacement, 2),
                    "moving_time_sec": round(self.track_motion_stats.get(t.track_id, {}).get("moving_time", 0.0), 2),
                    "low_motion_time_sec": round(self.track_motion_stats.get(t.track_id, {}).get("low_motion_time", 0.0), 2),
                }
                for t in person_tracks
            ]
        }

        # Article metrics
        articles_summary = {
            "unique_article_tracks": len(article_tracks),
            "articles_with_meaningful_movement": len([
                t for t in article_tracks if t.cumulative_distance >= self.meaningful_movement_pixels
            ]),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "first_seen_time_sec": round(t.first_seen_time, 2),
                    "last_seen_time_sec": round(t.last_seen_time, 2),
                    "visible_duration_sec": round(t.duration, 2),
                    "cumulative_distance_pixels": round(t.cumulative_distance, 2),
                    "net_displacement_pixels": round(t.net_displacement, 2),
                    "meaningful_movement_occurred": t.cumulative_distance >= self.meaningful_movement_pixels,
                }
                for t in article_tracks
            ]
        }

        # Event counts breakdown
        event_counts: Dict[str, int] = {}
        for ev in self.events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1

        return {
            "video_duration_analyzed_sec": round(total_video_duration, 2),
            "total_valid_tracks": len(valid_tracks),
            "people_analytics": people_summary,
            "articles_analytics": articles_summary,
            "line_crossing_analytics": {
                "enabled": self.line_enabled,
                "counts_by_direction": self.line_crossing_counts,
                "total_crossings": sum(self.line_crossing_counts.values()),
            },
            "event_counts_summary": event_counts,
        }
