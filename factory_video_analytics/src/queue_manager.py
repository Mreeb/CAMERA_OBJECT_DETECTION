"""Persistent processing queue manager and worker scheduler."""

import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

from .clip_processor import ClipProcessor
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages worker pool, dequeuing jobs, error handling, and recovery."""

    def __init__(
        self,
        database: DatabaseManager,
        clip_processor: ClipProcessor,
        max_workers: int = 1,
        poll_interval_sec: float = 1.0,
        failed_dir: str | Path = "data/failed",
    ):
        self.database = database
        self.processor = clip_processor
        self.max_workers = max_workers
        self.poll_interval = poll_interval_sec
        self.failed_dir = Path(failed_dir)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._workers: List[threading.Thread] = []

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Recover interrupted jobs from previous shutdown
        recovered = self.database.recover_interrupted_jobs()
        if recovered > 0:
            logger.info(f"Recovered {recovered} interrupted jobs on startup.")

        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"QueueWorker-{i+1}", daemon=True)
            t.start()
            self._workers.append(t)

        logger.info(f"QueueManager started with {self.max_workers} worker threads.")

    def stop(self) -> None:
        self._running = False
        for t in self._workers:
            if t.is_alive():
                t.join(timeout=3.0)
        self._workers.clear()
        logger.info("QueueManager stopped.")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                job = self.database.fetch_next_job()
                if job is None:
                    time.sleep(self.poll_interval)
                    continue

                self._execute_job(job)
            except Exception as e:
                logger.error(f"Unexpected error in queue worker: {e}", exc_info=True)
                time.sleep(self.poll_interval)

    def _execute_job(self, job: Dict[str, Any]) -> None:
        job_id = job["job_id"]
        clip_id = job["clip_id"]
        file_path = Path(job["file_path"])

        try:
            self.processor.process_clip(job)
        except Exception as e:
            logger.error(f"Job #{job_id} for clip {file_path.name} failed: {e}", exc_info=True)
            attempts = job.get("attempts", 1)
            can_retry = attempts < 3
            self.database.mark_job_failed(job_id, clip_id, str(e), can_retry=can_retry)

            if not can_retry:
                # Move to failed directory
                try:
                    if file_path.exists():
                        dest = self.failed_dir / file_path.name
                        shutil.move(str(file_path), str(dest))
                except Exception as move_err:
                    logger.warning(f"Could not move failed clip to {self.failed_dir}: {move_err}")
