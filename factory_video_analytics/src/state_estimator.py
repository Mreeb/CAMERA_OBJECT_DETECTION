"""Smoothed state estimator, bottom-center tracking, velocity filtering, and activity states."""

from collections import deque
from dataclasses import dataclass, field
import logging
import math
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MovementConfig:
    smoothing_alpha: float = 0.4
    warmup_seconds: float = 1.5
    window_seconds: float = 2.0
    enter_moving_threshold_px_per_sec: float = 20.0
    exit_moving_threshold_px_per_sec: float = 10.0
    movement_confirmation_seconds: float = 1.0
    stationary_confirmation_seconds: float = 2.0
    meaningful_article_displacement_px: float = 30.0
    low_motion_after_seconds: float = 30.0
    extended_low_motion_after_seconds: float = 120.0


class TrackSmoother:
    """Maintains smoothed spatial coordinates, rolling velocity, and hysteresis motion state."""

    def __init__(self, track_id: int, class_name: str, initial_box: Tuple[float, float, float, float], start_time: float, config: MovementConfig):
        self.track_id = track_id
        self.class_name = class_name
        self.is_person = class_name.lower() in ("person", "worker")
        self.config = config

        self.first_seen_time = start_time
        self.last_seen_time = start_time
        self.total_detections = 1
        self.missing_seconds = 0.0

        # Calculate initial anchor point
        raw_anchor = self._get_anchor(initial_box)
        self.smooth_anchor = raw_anchor
        self.initial_anchor = raw_anchor

        # Rolling history of (timestamp, smooth_x, smooth_y)
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=40)
        self.history.append((start_time, raw_anchor[0], raw_anchor[1]))

        # Trajectory trail for visualizer
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=30)
        self.trail.append(raw_anchor)

        # Activity & motion state
        self.motion_state = "Moving" if not self.is_person else "Unknown"
        self.article_movement_state = "visible_stationary"
        self.current_zone = "Unknown"
        self.initial_zone = "Unknown"
        self.has_transferred_zones = False

        self.current_velocity = 0.0
        self.low_motion_start_time: Optional[float] = None
        self.moving_start_time: Optional[float] = None
        self.time_in_current_state = 0.0
        self.last_state_change_time = start_time

        # Movement metrics
        self.cumulative_distance = 0.0
        self.net_displacement = 0.0

    def _get_anchor(self, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        if self.is_person:
            # Bottom-center for people avoids arm/head movements
            return ((x1 + x2) / 2.0, y2)
        else:
            # Box center for articles
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, box: Tuple[float, float, float, float], timestamp: float, current_zone: Optional[str] = None) -> None:
        """Updates track with a new detection."""
        self.last_seen_time = timestamp
        self.missing_seconds = 0.0
        self.total_detections += 1

        raw_anchor = self._get_anchor(box)

        # Exponential Moving Average smoothing
        alpha = self.config.smoothing_alpha
        prev_x, prev_y = self.smooth_anchor
        smooth_x = alpha * raw_anchor[0] + (1.0 - alpha) * prev_x
        smooth_y = alpha * raw_anchor[1] + (1.0 - alpha) * prev_y
        self.smooth_anchor = (smooth_x, smooth_y)

        # Step distance
        step_dist = math.hypot(smooth_x - prev_x, smooth_y - prev_y)
        # Deadband threshold to ignore micro-jitter (< 1.5 px)
        if step_dist > 1.5:
            self.cumulative_distance += step_dist

        # Net displacement from track origin
        self.net_displacement = math.hypot(
            smooth_x - self.initial_anchor[0],
            smooth_y - self.initial_anchor[1]
        )

        self.history.append((timestamp, smooth_x, smooth_y))
        self.trail.append((smooth_x, smooth_y))

        # Zone tracking
        if current_zone:
            if self.initial_zone == "Unknown":
                self.initial_zone = current_zone
            if self.current_zone != current_zone and self.current_zone != "Unknown":
                self.has_transferred_zones = True
            self.current_zone = current_zone

        # Velocity & Activity State calculation
        self._update_motion_state(timestamp)

    def mark_missing(self, dt: float) -> None:
        """Called when track was not detected in the current inference cycle."""
        self.missing_seconds += dt
        if self.missing_seconds > 0.5:
            self._set_state("Temporarily Lost", self.last_seen_time)

    def _estimate_velocity(self, current_time: float) -> float:
        """Calculates velocity in pixels/sec over rolling time window."""
        # Check warmup period
        age = current_time - self.first_seen_time
        if age < self.config.warmup_seconds:
            return 0.0

        cutoff = current_time - self.config.window_seconds
        # Find oldest point in window
        pts_in_window = [p for p in self.history if p[0] >= cutoff]
        if len(pts_in_window) < 2:
            return 0.0

        t0, x0, y0 = pts_in_window[0]
        t1, x1, y1 = pts_in_window[-1]
        dt = t1 - t0
        if dt < 0.2:
            return 0.0

        net_dist = math.hypot(x1 - x0, y1 - y0)
        return net_dist / dt

    def _update_motion_state(self, current_time: float) -> None:
        age = current_time - self.first_seen_time
        if age < self.config.warmup_seconds:
            self.current_velocity = 0.0
            if self.is_person:
                self._set_state("Unknown", current_time)
            return

        vel = self._estimate_velocity(current_time)
        self.current_velocity = vel

        if self.is_person:
            # Person Activity Hysteresis
            if vel >= self.config.enter_moving_threshold_px_per_sec:
                self.low_motion_start_time = None
                self._set_state("Moving", current_time)
            else:
                if self.low_motion_start_time is None:
                    self.low_motion_start_time = current_time

                low_motion_dur = current_time - self.low_motion_start_time
                if low_motion_dur >= self.config.extended_low_motion_after_seconds:
                    self._set_state("Extended Low Motion", current_time)
                elif low_motion_dur >= 5.0:
                    self._set_state("Low Motion", current_time)
                else:
                    self._set_state("Stationary", current_time)
        else:
            # Article Movement State
            if self.net_displacement >= self.config.meaningful_article_displacement_px or self.has_transferred_zones:
                if vel > 8.0:
                    self.article_movement_state = "moving_confirmed"
                    self._set_state("Moving", current_time)
                else:
                    self.article_movement_state = "stopped_after_movement"
                    self._set_state("Stopped", current_time)
            elif self.net_displacement > 15.0:
                self.article_movement_state = "movement_candidate"
                self._set_state("Candidate", current_time)
            else:
                self.article_movement_state = "visible_stationary"
                self._set_state("Stationary", current_time)

        self.time_in_current_state = current_time - self.last_state_change_time

    def _set_state(self, new_state: str, current_time: float) -> None:
        if self.motion_state != new_state:
            self.motion_state = new_state
            self.last_state_change_time = current_time
            self.time_in_current_state = 0.0
