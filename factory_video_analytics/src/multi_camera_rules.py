"""Multi-camera rules and location-level count aggregator."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MultiCameraAggregator:
    """Applies multi-camera rules to prevent double-counting across overlapping camera views."""

    def __init__(self, cameras_config: Dict[str, Any], locations_config: Dict[str, Any]):
        self.cameras_config = cameras_config
        self.locations_config = locations_config

    def get_primary_camera_for_location(self, location_id: str) -> Optional[str]:
        """Finds the primary camera assigned to a physical location."""
        loc_cfg = self.locations_config.get(location_id, {})
        if "primary_camera" in loc_cfg:
            return loc_cfg["primary_camera"]

        for cid, cam in self.cameras_config.items():
            if cam.get("location_id") == location_id and cam.get("role") == "primary":
                return cid
        return None

    def aggregate_location_metrics(
        self,
        location_id: str,
        clip_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregates metrics for a location using the primary camera for official counts."""
        primary_cam_id = self.get_primary_camera_for_location(location_id)

        primary_summaries = [s for s in clip_summaries if s["camera_id"] == primary_cam_id]
        verification_summaries = [s for s in clip_summaries if s["camera_id"] != primary_cam_id and s.get("location_id") == location_id]

        if not primary_summaries:
            # Fallback to all summaries if no primary designated
            target_summaries = clip_summaries
        else:
            target_summaries = primary_summaries

        if not target_summaries:
            return {
                "location_id": location_id,
                "max_simultaneous_people": 0,
                "avg_simultaneous_people": 0.0,
                "total_person_hours": 0.0,
                "active_person_hours": 0.0,
                "low_motion_person_hours": 0.0,
                "unknown_person_hours": 0.0,
                "total_transfers": 0,
                "total_line_crossings": 0,
                "verification_events_count": len(verification_summaries),
            }

        max_people = max(s["max_simultaneous_people"] for s in target_summaries)
        avg_people = sum(s["avg_simultaneous_people"] for s in target_summaries) / len(target_summaries)

        total_person_sec = sum(s["total_person_seconds"] for s in target_summaries)
        active_person_sec = sum(s["moving_person_seconds"] for s in target_summaries)
        low_motion_sec = sum(s["low_motion_person_seconds"] + s.get("ext_low_motion_person_seconds", 0) for s in target_summaries)
        unknown_sec = sum(s.get("unknown_person_seconds", 0) for s in target_summaries)

        total_transfers = sum(s["transferred_articles"] for s in target_summaries)
        total_crossings = sum(s.get("line_crossings_a_to_b", 0) + s.get("line_crossings_b_to_a", 0) for s in target_summaries)

        return {
            "location_id": location_id,
            "primary_camera": primary_cam_id,
            "max_simultaneous_people": max_people,
            "avg_simultaneous_people": round(avg_people, 2),
            "total_person_hours": round(total_person_sec / 3600.0, 2),
            "active_person_hours": round(active_person_sec / 3600.0, 2),
            "low_motion_person_hours": round(low_motion_sec / 3600.0, 2),
            "unknown_person_hours": round(unknown_sec / 3600.0, 2),
            "total_transfers": total_transfers,
            "total_line_crossings": total_crossings,
            "verification_clips_count": len(verification_summaries),
        }
