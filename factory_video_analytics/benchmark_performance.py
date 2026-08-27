"""Benchmark hardware throughput and compute multi-camera capacity for batch analytics."""

import argparse
import json
import logging
from pathlib import Path
import time
import torch
import yaml

from src.clip_processor import ClipProcessor
from src.cross_clip_stitcher import CrossClipStitcher
from src.database import DatabaseManager
from src.detector import YOLOEDetector
from src.evidence_manager import EvidenceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(
    video_path: str = "Videos/VID-20260820-WA0003.mp4",
    fps_targets: list[float] = [5.0, 8.0, 10.0],
):
    path = Path(video_path)
    if not path.exists():
        path = Path("../Videos/VID-20260820-WA0003.mp4")
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db = DatabaseManager("data/benchmark_test.db")
    stitcher = CrossClipStitcher()
    evidence_mgr = EvidenceManager("data/benchmark_evidence")

    detector = YOLOEDetector(
        checkpoint=cfg["detection"]["checkpoint"],
        fallback_checkpoint=cfg["detection"]["fallback_checkpoint"],
        prompts=cfg["detection"]["prompts"],
        person_confidence=cfg["detection"]["person_confidence"],
        article_confidence=cfg["detection"]["article_confidence"],
        device="auto",
    )

    results = []

    for target_fps in fps_targets:
        processor = ClipProcessor(
            detector=detector,
            database=db,
            cross_clip_stitcher=stitcher,
            evidence_manager=evidence_mgr,
            zones_config=cfg.get("zones", []),
            counting_line_config=cfg.get("counting_line", {}),
            movement_config=cfg.get("movement", {}),
            worker_activity_config=cfg.get("worker_activity", {}),
            completed_base_dir="data/benchmark_completed",
            inference_fps=target_fps,
            archive_completed=False,
        )

        job_data = {
            "job_id": int(target_fps * 10),
            "clip_id": int(target_fps * 10),
            "camera_id": "camera_02",
            "location_id": "loading_yard",
            "file_path": str(path.resolve()),
            "start_time": 1787245200.0,
            "duration_sec": 132.64,
        }

        start_wall = time.time()
        summary = processor.process_clip(job_data)
        elapsed_wall = time.time() - start_wall

        video_dur = job_data["duration_sec"]
        speed_ratio = video_dur / max(0.01, elapsed_wall)
        clips_per_hour = (3600.0 / elapsed_wall) * (video_dur / 300.0)  # normalized to 5-min clips

        # 4 Cameras generate 4 * (3600 / 300) = 48 five-minute clips per hour
        # Real-time required speed ratio for N cameras is N * 1.0 = 4.0x
        can_support_4_cams = speed_ratio >= 4.0
        can_support_5_cams = speed_ratio >= 5.0

        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        bench_item = {
            "target_inference_fps": target_fps,
            "video_duration_sec": round(video_dur, 2),
            "processing_time_sec": round(elapsed_wall, 2),
            "speed_to_video_ratio": round(speed_ratio, 2),
            "clips_5min_per_hour": round(clips_per_hour, 1),
            "vram_allocated_mb": round(vram_mb, 1),
            "supports_4_cameras": can_support_4_cams,
            "supports_5_cameras": can_support_5_cams,
            "capacity_margin_4_cams_pct": round(((speed_ratio - 4.0) / 4.0) * 100.0, 1),
        }
        results.append(bench_item)
        logger.info(
            f"Benchmark @ {target_fps} FPS: {speed_ratio:.2f}x real-time "
            f"({elapsed_wall:.2f}s for {video_dur:.1f}s video). Supports 4 cams: {can_support_4_cams}"
        )

    # Save benchmark report
    with open("data/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark complete. Results written to data/benchmark_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="../Videos/VID-20260820-WA0003.mp4")
    args = parser.parse_args()
    run_benchmark(args.video)
