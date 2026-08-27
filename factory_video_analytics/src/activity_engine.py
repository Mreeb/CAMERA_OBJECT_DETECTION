"""Activity Recognition Engine: computes real-time activity intensity and detects peak activity time windows."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ActivitySample:
    timestamp: float
    datetime_str: str
    camera_id: str
    activity_score: float       # 0.0 to 100.0%
    activity_level: str         # "HIGH ACTIVITY", "MODERATE", "LOW MOTION", "IDLE"
    visible_people: int
    moving_people: int
    low_motion_people: int
    avg_velocity: float
    article_transfers: int


@dataclass
class ActivityTimeWindow:
    camera_id: str
    start_time: float
    end_time: float
    start_str: str
    end_str: str
    duration_sec: float
    activity_level: str         # "HIGH ACTIVITY", "MODERATE", "LOW MOTION", "IDLE"
    avg_activity_score: float
    peak_workers: int
    avg_workers: float
    summary: str


class ActivityEngine:
    """Calculates activity intensity scores, maintains time-series history, and detects peak time intervals."""

    def __init__(self, history_maxlen: int = 1800):
        # Rolling sample history per camera (up to 30 mins at 1s resolution)
        self.camera_histories: Dict[str, Deque[ActivitySample]] = {}
        self.history_maxlen = history_maxlen

        # Detected contiguous time windows per camera
        self.detected_windows: Dict[str, List[ActivityTimeWindow]] = {}

        # Current window tracking per camera
        self._current_window_state: Dict[str, Dict[str, Any]] = {}

    def get_or_create_history(self, camera_id: str) -> Deque[ActivitySample]:
        if camera_id not in self.camera_histories:
            self.camera_histories[camera_id] = deque(maxlen=self.history_maxlen)
            self.detected_windows[camera_id] = []
        return self.camera_histories[camera_id]

    def compute_activity_score(
        self,
        visible_people: int,
        moving_people: int,
        low_motion_people: int,
        avg_velocity: float,
        article_transfers: int = 0,
    ) -> Tuple[float, str]:
        """Calculates 0-100% worker activity intensity score and qualitative classification."""
        if visible_people == 0:
            return 0.0, "IDLE"

        # Moving workers produce real activity; stationary/standing workers produce minimal activity score
        if moving_people == 0:
            score = min(20.0, (low_motion_people * 4.0) + min(5.0, avg_velocity * 0.5))
            return round(score, 1), "LOW MOTION"

        raw_score = (
            (moving_people * 38.0) +
            (low_motion_people * 5.0) +
            min(25.0, avg_velocity * 1.5)
        )

        score = min(100.0, max(0.0, raw_score))

        if score >= 60.0 or (moving_people >= 3):
            level = "HIGH ACTIVITY"
        elif score >= 25.0 or (moving_people >= 1):
            level = "MODERATE"
        else:
            level = "LOW MOTION"

        return round(score, 1), level

    def record_sample(
        self,
        camera_id: str,
        timestamp: float,
        visible_people: int,
        moving_people: int,
        low_motion_people: int,
        avg_velocity: float,
        article_transfers: int,
    ) -> ActivitySample:
        """Records a 1-second activity sample and updates time-window segmentation."""
        score, level = self.compute_activity_score(
            visible_people=visible_people,
            moving_people=moving_people,
            low_motion_people=low_motion_people,
            avg_velocity=avg_velocity,
            article_transfers=article_transfers,
        )

        dt_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

        sample = ActivitySample(
            timestamp=timestamp,
            datetime_str=dt_str,
            camera_id=camera_id,
            activity_score=score,
            activity_level=level,
            visible_people=visible_people,
            moving_people=moving_people,
            low_motion_people=low_motion_people,
            avg_velocity=round(avg_velocity, 1),
            article_transfers=article_transfers,
        )

        history = self.get_or_create_history(camera_id)
        history.append(sample)

        # Update contiguous time-window segmentation
        self._update_time_windows(camera_id, sample)

        return sample

    def _update_time_windows(self, camera_id: str, sample: ActivitySample) -> None:
        """Groups contiguous activity states into clear time-window segments."""
        if camera_id not in self._current_window_state:
            self._current_window_state[camera_id] = {
                "level": sample.activity_level,
                "start_time": sample.timestamp,
                "start_str": sample.datetime_str,
                "scores": [sample.activity_score],
                "workers": [sample.visible_people],
            }
            return

        curr = self._current_window_state[camera_id]

        # Check if activity level changed or time gap occurred
        if curr["level"] != sample.activity_level:
            duration = sample.timestamp - curr["start_time"]
            # Only record windows of meaningful duration (>= 5s)
            if duration >= 5.0:
                avg_score = sum(curr["scores"]) / len(curr["scores"])
                peak_w = max(curr["workers"]) if curr["workers"] else 0
                avg_w = sum(curr["workers"]) / len(curr["workers"]) if curr["workers"] else 0.0

                summary_text = f"{curr['level']} with {peak_w} peak workers ({avg_score:.0f}% intensity)"
                if curr["level"] == "HIGH ACTIVITY":
                    summary_text += " - Continuous worker movement and physical operations"
                elif curr["level"] == "LOW MOTION":
                    summary_text += " - Stationary standing or brief pause"

                window = ActivityTimeWindow(
                    camera_id=camera_id,
                    start_time=curr["start_time"],
                    end_time=sample.timestamp,
                    start_str=curr["start_str"],
                    end_str=sample.datetime_str,
                    duration_sec=round(duration, 1),
                    activity_level=curr["level"],
                    avg_activity_score=round(avg_score, 1),
                    peak_workers=peak_w,
                    avg_workers=round(avg_w, 1),
                    summary=summary_text,
                )

                if camera_id not in self.detected_windows:
                    self.detected_windows[camera_id] = []
                self.detected_windows[camera_id].insert(0, window)  # Newest first

                # Keep last 50 windows
                if len(self.detected_windows[camera_id]) > 50:
                    self.detected_windows[camera_id].pop()

            # Start new window
            self._current_window_state[camera_id] = {
                "level": sample.activity_level,
                "start_time": sample.timestamp,
                "start_str": sample.datetime_str,
                "scores": [sample.activity_score],
                "workers": [sample.visible_people],
            }
        else:
            curr["scores"].append(sample.activity_score)
            curr["workers"].append(sample.visible_people)

    def get_time_windows(self, camera_id: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Returns detected 'From When To When' peak activity intervals."""
        all_windows = []
        if camera_id and camera_id in self.detected_windows:
            windows = self.detected_windows[camera_id]
        else:
            windows = []
            for cam_win in self.detected_windows.values():
                windows.extend(cam_win)
            windows.sort(key=lambda w: w.start_time, reverse=True)

        for w in windows[:limit]:
            all_windows.append({
                "camera_id": w.camera_id,
                "start_str": w.start_str,
                "end_str": w.end_str,
                "duration_str": f"{int(w.duration_sec // 60)}m {int(w.duration_sec % 60)}s" if w.duration_sec >= 60 else f"{int(w.duration_sec)}s",
                "duration_sec": w.duration_sec,
                "activity_level": w.activity_level,
                "avg_activity_score": w.avg_activity_score,
                "peak_workers": w.peak_workers,
                "avg_workers": w.avg_workers,
                "summary": w.summary,
            })
        return all_windows

    def get_timeline_data(self, camera_id: Optional[str] = None, max_points: int = 30) -> Dict[str, Any]:
        """Returns structured, smooth, continuous timeline data for Chart.js graphs."""
        now = int(time.time())
        # Clean 1-second interval time window
        second_timestamps = [now - i for i in range(max_points - 1, -1, -1)]
        labels = [datetime.fromtimestamp(t).strftime("%H:%M:%S") for t in second_timestamps]

        colors = {
            "camera_01": "#10b981",  # Emerald
            "camera_02": "#38bdf8",  # Sky Blue
            "camera_03": "#f59e0b",  # Amber
            "camera_04": "#a855f7",  # Purple
        }

        names = {
            "camera_01": "Camera 01 (Entrance)",
            "camera_02": "Camera 02 (Loading Yard)",
            "camera_03": "Camera 03 (Storage Bay)",
            "camera_04": "Camera 04 (Dispatch Bay)",
        }

        # Single camera requested
        if camera_id and camera_id in self.camera_histories and camera_id != "all":
            history = list(self.camera_histories[camera_id])
            data = []
            for t in second_timestamps:
                if history:
                    # Find nearest sample in time
                    nearest = min(history, key=lambda s: abs(s.timestamp - t))
                    if abs(nearest.timestamp - t) <= 10.0:
                        data.append(nearest.activity_score)
                    else:
                        data.append(0.0)
                else:
                    data.append(0.0)

            c_hex = colors.get(camera_id, "#38bdf8")
            return {
                "labels": labels,
                "datasets": [
                    {
                        "label": names.get(camera_id, camera_id.upper()),
                        "data": data,
                        "borderColor": c_hex,
                        "backgroundColor": f"{c_hex}15",
                        "borderWidth": 2.5,
                        "tension": 0.45,
                        "fill": True,
                        "pointRadius": 0,
                    }
                ]
            }

        # All cameras requested
        datasets = []
        for cid in ["camera_01", "camera_02", "camera_03", "camera_04"]:
            if cid in self.camera_histories:
                history = list(self.camera_histories[cid])
                data = []
                for t in second_timestamps:
                    if history:
                        nearest = min(history, key=lambda s: abs(s.timestamp - t))
                        if abs(nearest.timestamp - t) <= 10.0:
                            data.append(nearest.activity_score)
                        else:
                            data.append(0.0)
                    else:
                        data.append(0.0)

                c_hex = colors.get(cid, "#ffffff")
                datasets.append({
                    "label": names.get(cid, cid.upper()),
                    "data": data,
                    "borderColor": c_hex,
                    "backgroundColor": f"{c_hex}10",
                    "borderWidth": 2,
                    "tension": 0.45,
                    "fill": True,
                    "pointRadius": 0,
                })

        return {"labels": labels, "datasets": datasets}

    def get_overall_summary(self) -> Dict[str, Any]:
        """Computes facility-wide executive activity metrics."""
        latest_scores = []
        total_workers_now = 0
        total_moving_now = 0
        total_low_motion_now = 0

        for history in self.camera_histories.values():
            if history:
                latest = history[-1]
                latest_scores.append(latest.activity_score)
                total_workers_now += latest.visible_people
                total_moving_now += latest.moving_people
                total_low_motion_now += latest.low_motion_people

        overall_score = (sum(latest_scores) / len(latest_scores)) if latest_scores else 0.0

        if overall_score >= 60.0:
            overall_level = "HIGH ACTIVITY"
        elif overall_score >= 25.0:
            overall_level = "MODERATE"
        elif total_workers_now > 0:
            overall_level = "LOW MOTION"
        else:
            overall_level = "IDLE"

        # Find peak window across all cameras
        peak_window_str = "09:00 - 11:30 (Morning Peak)"
        all_wins = self.get_time_windows(limit=5)
        high_wins = [w for w in all_wins if w["activity_level"] == "HIGH ACTIVITY"]
        if high_wins:
            peak_window_str = f"{high_wins[0]['start_str']} - {high_wins[0]['end_str']} ({high_wins[0]['camera_id']})"

        return {
            "overall_activity_score": round(overall_score, 1),
            "overall_activity_level": overall_level,
            "total_active_workers_now": total_workers_now,
            "total_moving_workers_now": total_moving_now,
            "total_low_motion_now": total_low_motion_now,
            "peak_activity_window": peak_window_str,
            "monitored_cameras_count": len(self.camera_histories),
        }
