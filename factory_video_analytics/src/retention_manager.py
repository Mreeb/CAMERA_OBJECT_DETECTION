"""Safe automated retention and cleanup policies for clips, evidence, and database logs."""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class RetentionManager:
    """Enforces retention periods for completed clips, evidence, and database records."""

    def __init__(
        self,
        database: DatabaseManager,
        retention_config: Dict[str, Any],
        completed_dir: str | Path = "data/completed",
        evidence_dir: str | Path = "data/evidence",
    ):
        self.database = database
        self.completed_dir = Path(completed_dir)
        self.evidence_dir = Path(evidence_dir)

        self.original_clips_days = int(retention_config.get("original_clips_days", 30))
        self.evidence_clips_days = int(retention_config.get("evidence_clips_days", 90))
        self.snapshots_days = int(retention_config.get("snapshots_days", 90))
        self.structured_events_days = int(retention_config.get("structured_events_days", 365))
        self.delete_original_after_processing = bool(retention_config.get("delete_original_after_processing", False))

    def run_cleanup_cycle(self) -> Dict[str, int]:
        """Runs a safe retention cleanup pass."""
        now = time.time()
        deleted_counts = {
            "completed_clips": 0,
            "evidence_clips": 0,
            "snapshots": 0,
            "database_events": 0,
        }

        # 1. Cleanup completed source clips
        if self.completed_dir.exists():
            cutoff = now - (self.original_clips_days * 86400)
            for f in self.completed_dir.rglob("*.mp4"):
                try:
                    if self.delete_original_after_processing or f.stat().st_mtime < cutoff:
                        f.unlink()
                        deleted_counts["completed_clips"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old clip {f}: {e}")

        # 2. Cleanup evidence clips
        clips_dir = self.evidence_dir / "clips"
        if clips_dir.exists():
            cutoff = now - (self.evidence_clips_days * 86400)
            for f in clips_dir.glob("*.mp4"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        deleted_counts["evidence_clips"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old evidence clip {f}: {e}")

        # 3. Cleanup snapshots
        snap_dir = self.evidence_dir / "snapshots"
        if snap_dir.exists():
            cutoff = now - (self.snapshots_days * 86400)
            for f in snap_dir.glob("*.jpg"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        deleted_counts["snapshots"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old snapshot {f}: {e}")

        # 4. Cleanup old database events
        conn = self.database._get_connection()
        cutoff_events = now - (self.structured_events_days * 86400)
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE timestamp < ?;", (cutoff_events,))
            deleted_counts["database_events"] = cursor.rowcount

        logger.info(f"Retention cleanup cycle complete: {deleted_counts}")
        return deleted_counts
