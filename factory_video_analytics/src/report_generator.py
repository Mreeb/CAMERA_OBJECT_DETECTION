"""Deterministic plain-language and structured report generator for shift, daily, and hourly analytics."""

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates verifiable plain-language narratives and structured exports from SQLite metrics."""

    def __init__(self, database: DatabaseManager, reports_dir: str | Path = "data/reports"):
        self.database = database
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_shift_report(
        self,
        shift_name: str = "Morning Shift (09:00 - 17:00)",
        date_str: Optional[str] = None,
        location_id: str = "loading_yard",
    ) -> Dict[str, Any]:
        """Calculates shift metrics and compiles a plain-language narrative."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        conn = self.database._get_connection()
        cursor = conn.cursor()

        # Query all clip summaries for this location
        cursor.execute("""
            SELECT cs.*, c.role
            FROM clip_summaries cs
            JOIN cameras c ON cs.camera_id = c.id
            WHERE cs.location_id = ?
            ORDER BY cs.start_time ASC;
        """, (location_id,))
        rows = cursor.fetchall()
        summaries = [dict(r) for r in rows]

        if not summaries:
            # Fallback if no location filter matches
            cursor.execute("SELECT * FROM clip_summaries ORDER BY start_time ASC;")
            summaries = [dict(r) for r in cursor.fetchall()]

        if not summaries:
            empty_narrative = (
                f"No processed video data found for {shift_name} on {date_str} in location '{location_id}'."
            )
            return {
                "shift_name": shift_name,
                "date_str": date_str,
                "location_id": location_id,
                "max_people": 0,
                "avg_people": 0.0,
                "total_person_hours": 0.0,
                "active_pct": 0.0,
                "low_motion_pct": 0.0,
                "unknown_pct": 0.0,
                "total_transfers": 0,
                "total_line_crossings": 0,
                "ext_low_motion_count": 0,
                "plain_language_report": empty_narrative,
            }

        # Filter to Primary cameras for official counts
        primary_summaries = [s for s in summaries if s.get("role") == "primary"] or summaries

        max_people = max(s["max_simultaneous_people"] for s in primary_summaries)
        avg_people = sum(s["avg_simultaneous_people"] for s in primary_summaries) / len(primary_summaries)

        total_person_sec = sum(s["total_person_seconds"] for s in primary_summaries)
        active_person_sec = sum(s["moving_person_seconds"] for s in primary_summaries)
        low_motion_sec = sum(s["low_motion_person_seconds"] + s.get("ext_low_motion_person_seconds", 0) for s in primary_summaries)
        unknown_sec = sum(s.get("unknown_person_seconds", 0) for s in primary_summaries)

        total_person_hours = total_person_sec / 3600.0
        active_hours = active_person_sec / 3600.0
        low_motion_hours = low_motion_sec / 3600.0
        unknown_hours = unknown_sec / 3600.0

        if total_person_sec > 0:
            active_pct = (active_person_sec / total_person_sec) * 100.0
            low_motion_pct = (low_motion_sec / total_person_sec) * 100.0
            unknown_pct = (unknown_sec / total_person_sec) * 100.0
        else:
            active_pct, low_motion_pct, unknown_pct = 0.0, 0.0, 0.0

        total_transfers = sum(s["transferred_articles"] for s in primary_summaries)
        total_crossings = sum(s.get("line_crossings_a_to_b", 0) + s.get("line_crossings_b_to_a", 0) for s in primary_summaries)
        total_sessions = sum(s.get("new_tracker_sessions", 0) for s in primary_summaries)

        # Count extended low motion events
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM events
            WHERE event_type = 'extended_low_motion_detected';
        """)
        ext_low_cnt = cursor.fetchone()["cnt"]

        # Data quality checks
        warnings = []
        if total_sessions > max_people * 15 and max_people > 0:
            warnings.append(
                f"Elevated tracker session rate ({total_sessions} sessions for peak {max_people} people) "
                "due to frequent camera edge entries or brief occlusions."
            )

        start_time = min(s["start_time"] for s in primary_summaries)
        end_time = max(s["end_time"] for s in primary_summaries)

        # Compile Deterministic Plain-Language Narrative
        narrative = (
            f"During the {shift_name} ({date_str}), a maximum of {max_people} people were visible "
            f"simultaneously in the monitored {location_id} area. The cameras recorded {total_person_hours:.1f} "
            f"observed person-hours. Approximately {active_pct:.1f}% of classifiable visible time showed regular "
            f"movement or confirmed handling activity, while {low_motion_pct:.1f}% was low motion. "
            f"The remaining {unknown_pct:.1f}% could not be classified because of occlusion or initial track settling. "
            f"The system recorded {total_transfers} confirmed article transfers between storage and loading areas, "
            f"and {total_crossings} directional counting line crossings. "
            f"{ext_low_cnt} extended-low-motion periods (>120s) were flagged for review. "
            f"These periods do not necessarily mean that no work was being performed.\n\n"
            f"Note on Distinct Workers: The system recorded {total_sessions} tracker sessions. "
            f"Tracker sessions do not represent unique individuals because identities were not preserved across all "
            f"occlusions and cameras. The maximum simultaneously visible count was {max_people}."
        )

        shift_record = {
            "shift_name": shift_name,
            "date_str": date_str,
            "location_id": location_id,
            "start_time": start_time,
            "end_time": end_time,
            "max_people": max_people,
            "avg_people": round(avg_people, 2),
            "total_person_hours": round(total_person_hours, 2),
            "active_person_hours": round(active_hours, 2),
            "low_motion_person_hours": round(low_motion_hours, 2),
            "unknown_person_hours": round(unknown_hours, 2),
            "active_pct": round(active_pct, 1),
            "low_motion_pct": round(low_motion_pct, 1),
            "unknown_pct": round(unknown_pct, 1),
            "total_transfers": total_transfers,
            "total_line_crossings": total_crossings,
            "ext_low_motion_count": ext_low_cnt,
            "data_quality_warnings": "; ".join(warnings) if warnings else "None",
            "plain_language_report": narrative,
        }

        # Store in SQLite
        self.database.insert_shift_summary(shift_record)
        return shift_record

    def export_shift_report_files(self, shift_data: Dict[str, Any], prefix: str = "shift_report") -> Dict[str, str]:
        """Exports report in TXT, JSON, and CSV formats."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Plain Text
        txt_path = self.reports_dir / f"{prefix}_{timestamp_str}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== FACTORY VIDEO ANALYTICS SHIFT REPORT ===\n")
            f.write(f"Shift: {shift_data['shift_name']} | Date: {shift_data['date_str']} | Location: {shift_data['location_id']}\n\n")
            f.write(shift_data["plain_language_report"])
            f.write("\n\n--- METRICS TABLE ---\n")
            for k, v in shift_data.items():
                if k != "plain_language_report":
                    f.write(f"{k:30s}: {v}\n")

        # 2. JSON
        json_path = self.reports_dir / f"{prefix}_{timestamp_str}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(shift_data, f, indent=2)

        # 3. CSV
        csv_path = self.reports_dir / f"{prefix}_{timestamp_str}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in shift_data.items():
                if k != "plain_language_report":
                    writer.writerow([k, v])

        logger.info(f"Exported shift reports to {self.reports_dir}")
        return {
            "txt_path": str(txt_path),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }
