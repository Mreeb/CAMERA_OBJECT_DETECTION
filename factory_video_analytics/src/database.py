"""SQLite Database Manager for scheduled clip-based batch analytics, queue, and evidence."""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite schema, persistent queue, track sessions, events, and reports."""

    def __init__(self, db_path: str | Path = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for high concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.conn

    def _init_tables(self) -> None:
        conn = self._get_connection()
        with conn:
            # 1. Cameras
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'primary',
                    clock_offset_sec REAL NOT NULL DEFAULT 0.0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Locations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Clips
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    datetime_start TEXT NOT NULL,
                    datetime_end TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    is_estimated_timestamp INTEGER NOT NULL DEFAULT 0,
                    model_version TEXT NOT NULL DEFAULT 'yoloe-v2',
                    config_version TEXT NOT NULL DEFAULT 'v2.0',
                    processed_at REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id)
                );
            """)

            # 4. Processing Jobs (Persistent Queue)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    priority INTEGER NOT NULL DEFAULT 10,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    error_message TEXT,
                    started_at REAL,
                    completed_at REAL,
                    processing_time_sec REAL,
                    speed_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (clip_id) REFERENCES clips(id)
                );
            """)

            # 5. Track Sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS track_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    internal_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    is_person INTEGER NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    max_motion_state TEXT NOT NULL,
                    initial_zone TEXT,
                    final_zone TEXT,
                    net_displacement_px REAL NOT NULL DEFAULT 0.0,
                    total_distance_px REAL NOT NULL DEFAULT 0.0,
                    is_continued_from_prev INTEGER NOT NULL DEFAULT 0,
                    possible_continuation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (clip_id) REFERENCES clips(id)
                );
            """)

            # 6. Events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id INTEGER,
                    camera_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    track_id INTEGER,
                    class_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    datetime_str TEXT,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    zone TEXT,
                    details TEXT,
                    snapshot_path TEXT,
                    evidence_clip_path TEXT,
                    human_verification TEXT DEFAULT 'unreviewed',
                    human_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 7. Clip Summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clip_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id INTEGER NOT NULL UNIQUE,
                    camera_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    max_simultaneous_people INTEGER NOT NULL,
                    avg_simultaneous_people REAL NOT NULL,
                    total_person_seconds REAL NOT NULL,
                    moving_person_seconds REAL NOT NULL,
                    low_motion_person_seconds REAL NOT NULL,
                    ext_low_motion_person_seconds REAL NOT NULL,
                    unknown_person_seconds REAL NOT NULL,
                    total_articles_seen INTEGER NOT NULL,
                    stationary_articles INTEGER NOT NULL,
                    moved_articles INTEGER NOT NULL,
                    transferred_articles INTEGER NOT NULL,
                    line_crossings_a_to_b INTEGER NOT NULL,
                    line_crossings_b_to_a INTEGER NOT NULL,
                    new_tracker_sessions INTEGER NOT NULL,
                    fragmentation_rate REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (clip_id) REFERENCES clips(id)
                );
            """)

            # 8. Shift Summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shift_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shift_name TEXT NOT NULL,
                    date_str TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    max_people INTEGER NOT NULL,
                    avg_people REAL NOT NULL,
                    total_person_hours REAL NOT NULL,
                    active_person_hours REAL NOT NULL,
                    low_motion_person_hours REAL NOT NULL,
                    unknown_person_hours REAL NOT NULL,
                    active_pct REAL NOT NULL,
                    low_motion_pct REAL NOT NULL,
                    unknown_pct REAL NOT NULL,
                    total_transfers INTEGER NOT NULL,
                    total_line_crossings INTEGER NOT NULL,
                    ext_low_motion_count INTEGER NOT NULL,
                    data_quality_warnings TEXT,
                    plain_language_report TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 9. Evidence Records
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_id INTEGER,
                    camera_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    datetime_str TEXT,
                    snapshot_path TEXT,
                    clip_path TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    operator_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 10. Zone Definitions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS zone_definitions (
                    camera_id TEXT PRIMARY KEY,
                    zones_json TEXT NOT NULL,
                    counting_line_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Auto-migrate columns if table existed from earlier version
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(events);")
            event_cols = [c[1] for c in cursor.fetchall()]
            if "clip_id" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN clip_id INTEGER;")
            if "evidence_clip_path" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN evidence_clip_path TEXT;")
            if "human_verification" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN human_verification TEXT DEFAULT 'unreviewed';")
            if "human_notes" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN human_notes TEXT;")

            # Indexes for high performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_camera ON clips(camera_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_hash ON clips(file_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_clip ON events(clip_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);")

        logger.info(f"Initialized SQLite database tables at {self.db_path}")

    # Camera & Location Operations
    def register_camera(self, camera_id: str, name: str, location_id: str, role: str = "primary", clock_offset: float = 0.0) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO cameras (id, name, location_id, role, clock_offset_sec, enabled)
                VALUES (?, ?, ?, ?, ?, 1);
            """, (camera_id, name, location_id, role, clock_offset))

    def register_location(self, location_id: str, name: str, description: str = "") -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO locations (id, name, description)
                VALUES (?, ?, ?);
            """, (location_id, name, description))

    # Ingestion & Clip Operations
    def clip_exists_by_hash(self, file_hash: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM clips WHERE file_hash = ? LIMIT 1;", (file_hash,))
        return cursor.fetchone() is not None

    def insert_clip_and_job(self, clip_data: Dict[str, Any], priority: int = 10) -> int:
        """Atomically registers a clip and enqueues its processing job."""
        conn = self._get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clips (
                    camera_id, location_id, filename, file_hash, file_path,
                    start_time, end_time, datetime_start, datetime_end,
                    duration_sec, frame_count, fps, status, is_estimated_timestamp,
                    model_version, config_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?);
            """, (
                clip_data["camera_id"],
                clip_data.get("location_id", "default"),
                clip_data["filename"],
                clip_data["file_hash"],
                clip_data["file_path"],
                clip_data["start_time"],
                clip_data["end_time"],
                clip_data.get("datetime_start", ""),
                clip_data.get("datetime_end", ""),
                clip_data["duration_sec"],
                clip_data["frame_count"],
                clip_data["fps"],
                1 if clip_data.get("is_estimated_timestamp", False) else 0,
                clip_data.get("model_version", "yoloe-v2"),
                clip_data.get("config_version", "v2.0"),
            ))
            clip_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO processing_jobs (clip_id, camera_id, status, priority, attempts)
                VALUES (?, ?, 'QUEUED', ?, 0);
            """, (clip_id, clip_data["camera_id"], priority))

            return clip_id

    # Queue Operations
    def fetch_next_job(self) -> Optional[Dict[str, Any]]:
        """Atomically fetches and locks the highest priority QUEUED job."""
        conn = self._get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT j.id as job_id, j.clip_id, j.camera_id, j.priority, j.attempts,
                       c.filename, c.file_path, c.start_time, c.end_time, c.duration_sec,
                       c.frame_count, c.fps, c.location_id
                FROM processing_jobs j
                JOIN clips c ON j.clip_id = c.id
                WHERE j.status = 'QUEUED'
                ORDER BY j.priority DESC, c.start_time ASC, j.id ASC
                LIMIT 1;
            """)
            row = cursor.fetchone()
            if not row:
                return None

            job = dict(row)
            # Mark job as PROCESSING
            now = time.time()
            cursor.execute("""
                UPDATE processing_jobs
                SET status = 'PROCESSING', started_at = ?, attempts = attempts + 1
                WHERE id = ?;
            """, (now, job["job_id"]))

            cursor.execute("UPDATE clips SET status = 'PROCESSING' WHERE id = ?;", (job["clip_id"],))
            return job

    def mark_job_completed(self, job_id: int, clip_id: int, processing_time: float, speed_ratio: float) -> None:
        conn = self._get_connection()
        now = time.time()
        with conn:
            conn.execute("""
                UPDATE processing_jobs
                SET status = 'COMPLETED', completed_at = ?, processing_time_sec = ?, speed_ratio = ?
                WHERE id = ?;
            """, (now, processing_time, speed_ratio, job_id))

            conn.execute("UPDATE clips SET status = 'COMPLETED', processed_at = ? WHERE id = ?;", (now, clip_id))

    def mark_job_failed(self, job_id: int, clip_id: int, error_msg: str, can_retry: bool = False) -> None:
        conn = self._get_connection()
        now = time.time()
        new_status = "QUEUED" if can_retry else "FAILED"
        with conn:
            conn.execute("""
                UPDATE processing_jobs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE id = ?;
            """, (new_status, now, error_msg, job_id))

            conn.execute("UPDATE clips SET status = ? WHERE id = ?;", (new_status, clip_id))

    def recover_interrupted_jobs(self) -> int:
        """Resets orphaned PROCESSING jobs back to QUEUED after server restart."""
        conn = self._get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE processing_jobs SET status = 'QUEUED' WHERE status = 'PROCESSING';")
            cursor.execute("UPDATE clips SET status = 'QUEUED' WHERE status = 'PROCESSING';")
            return cursor.rowcount

    def get_queue_metrics(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                SUM(CASE WHEN status = 'QUEUED' THEN 1 ELSE 0 END) as waiting_count,
                SUM(CASE WHEN status = 'PROCESSING' THEN 1 ELSE 0 END) as processing_count,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_count,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_count,
                AVG(CASE WHEN status = 'COMPLETED' THEN processing_time_sec END) as avg_processing_time,
                AVG(CASE WHEN status = 'COMPLETED' THEN speed_ratio END) as avg_speed_ratio,
                MIN(CASE WHEN status = 'QUEUED' THEN created_at END) as oldest_waiting_created
            FROM processing_jobs;
        """)
        row = cursor.fetchone()
        data = dict(row) if row else {}
        return {
            "waiting_clips": data.get("waiting_count") or 0,
            "processing_clips": data.get("processing_count") or 0,
            "completed_clips": data.get("completed_count") or 0,
            "failed_clips": data.get("failed_count") or 0,
            "avg_processing_time_sec": round(data.get("avg_processing_time") or 0.0, 2),
            "avg_speed_ratio": round(data.get("avg_speed_ratio") or 0.0, 2),
            "oldest_waiting_created": data.get("oldest_waiting_created"),
        }

    # Structured Metrics, Tracks, and Events
    def insert_track_sessions(self, clip_id: int, camera_id: str, sessions: List[Dict[str, Any]]) -> None:
        conn = self._get_connection()
        with conn:
            for s in sessions:
                conn.execute("""
                    INSERT INTO track_sessions (
                        clip_id, camera_id, session_id, internal_id, class_name, is_person,
                        first_seen, last_seen, duration_sec, max_motion_state, initial_zone,
                        final_zone, net_displacement_px, total_distance_px, is_continued_from_prev,
                        possible_continuation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    clip_id, camera_id, s["session_id"], s["internal_id"], s["class_name"],
                    1 if s["is_person"] else 0, s["first_seen"], s["last_seen"], s["duration_sec"],
                    s["max_motion_state"], s.get("initial_zone"), s.get("final_zone"),
                    s.get("net_displacement_px", 0.0), s.get("total_distance_px", 0.0),
                    1 if s.get("is_continued_from_prev", False) else 0,
                    s.get("possible_continuation_id"),
                ))

    def insert_event(self, event_data: Dict[str, Any]) -> int:
        conn = self._get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (
                    clip_id, camera_id, session_id, track_id, class_name, timestamp,
                    datetime_str, event_type, state, zone, details, snapshot_path,
                    evidence_clip_path, human_verification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed');
            """, (
                event_data.get("clip_id"),
                event_data["camera_id"],
                event_data["session_id"],
                event_data.get("track_id", 0),
                event_data["class_name"],
                event_data["timestamp"],
                event_data.get("datetime_str", datetime.fromtimestamp(event_data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")),
                event_data["event_type"],
                event_data["state"],
                event_data.get("zone", "Unknown"),
                event_data.get("details", ""),
                event_data.get("snapshot_path"),
                event_data.get("evidence_clip_path"),
            ))
            return cursor.lastrowid

    def insert_clip_summary(self, summary: Dict[str, Any]) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO clip_summaries (
                    clip_id, camera_id, location_id, start_time, end_time, duration_sec,
                    max_simultaneous_people, avg_simultaneous_people, total_person_seconds,
                    moving_person_seconds, low_motion_person_seconds, ext_low_motion_person_seconds,
                    unknown_person_seconds, total_articles_seen, stationary_articles,
                    moved_articles, transferred_articles, line_crossings_a_to_b,
                    line_crossings_b_to_a, new_tracker_sessions, fragmentation_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                summary["clip_id"], summary["camera_id"], summary.get("location_id", "default"),
                summary["start_time"], summary["end_time"], summary["duration_sec"],
                summary["max_simultaneous_people"], summary["avg_simultaneous_people"],
                summary["total_person_seconds"], summary["moving_person_seconds"],
                summary["low_motion_person_seconds"], summary["ext_low_motion_person_seconds"],
                summary["unknown_person_seconds"], summary["total_articles_seen"],
                summary["stationary_articles"], summary["moved_articles"],
                summary["transferred_articles"], summary.get("line_crossings_a_to_b", 0),
                summary.get("line_crossings_b_to_a", 0), summary["new_tracker_sessions"],
                summary["fragmentation_rate"],
            ))

    def insert_shift_summary(self, summary: Dict[str, Any]) -> int:
        conn = self._get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO shift_summaries (
                    shift_name, date_str, location_id, start_time, end_time,
                    max_people, avg_people, total_person_hours, active_person_hours,
                    low_motion_person_hours, unknown_person_hours, active_pct, low_motion_pct,
                    unknown_pct, total_transfers, total_line_crossings, ext_low_motion_count,
                    data_quality_warnings, plain_language_report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                summary["shift_name"], summary["date_str"], summary["location_id"],
                summary["start_time"], summary["end_time"], summary["max_people"],
                summary["avg_people"], summary["total_person_hours"], summary["active_person_hours"],
                summary["low_motion_person_hours"], summary["unknown_person_hours"],
                summary["active_pct"], summary["low_motion_pct"], summary["unknown_pct"],
                summary["total_transfers"], summary["total_line_crossings"],
                summary["ext_low_motion_count"], summary.get("data_quality_warnings", ""),
                summary["plain_language_report"],
            ))
            return cursor.lastrowid

    def get_recent_events(self, camera_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if camera_id:
            cursor.execute("""
                SELECT * FROM events WHERE camera_id = ? ORDER BY timestamp DESC LIMIT ?;
            """, (camera_id, limit))
        else:
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?;", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_evidence_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.*, c.filename as clip_filename
            FROM events e
            LEFT JOIN clips c ON e.clip_id = c.id
            WHERE e.snapshot_path IS NOT NULL OR e.evidence_clip_path IS NOT NULL
            ORDER BY e.timestamp DESC LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def update_human_verification(self, event_id: int, status: str, notes: str = "") -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                UPDATE events
                SET human_verification = ?, human_notes = ?
                WHERE id = ?;
            """, (status, notes, event_id))

    def get_latest_shift_summary(self, location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if location_id:
            cursor.execute("SELECT * FROM shift_summaries WHERE location_id = ? ORDER BY id DESC LIMIT 1;", (location_id,))
        else:
            cursor.execute("SELECT * FROM shift_summaries ORDER BY id DESC LIMIT 1;")
        row = cursor.fetchone()
        return dict(row) if row else None
