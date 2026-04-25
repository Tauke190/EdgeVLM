import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, write_csv, write_json
from tools.offline_benchmark_runner import discover_videos, resolve_video_paths, sanitize_name
from tools.offline_runtime_demo import build_raw_config, run_offline_runtime


SUMMARY_FIELDNAMES = [
    "video_path",
    "run_dir",
    "pipeline_mode",
    "frames_read",
    "output_ready_frames",
    "active_frames",
    "frames_written",
    "motion_active_frames",
    "person_active_frames",
    "person_detector_frames",
    "motion_event_count",
    "person_event_count",
    "sia_activation_count",
    "effective_fps",
    "inference_mean_ms",
    "active_loop_mean_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full motion-person-SiA benchmark harness for the shared runtime."
    )
    parser.add_argument(
        "--config",
        default="configs/full_pipeline_runtime.json",
        help="Path to a runtime config JSON file. Default: configs/full_pipeline_runtime.json",
    )
    parser.add_argument("--video", help="Run one specific input video.")
    parser.add_argument("--video-dir", help="Run every supported video under a directory.")
    parser.add_argument("--glob", default="*", help="Filename pattern inside --video-dir. Default: '*'")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under --video-dir.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--output-dir", help="Optional explicit suite output directory.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from each source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    return parser.parse_args()


def build_summary_row(video_path, result):
    metrics = result["metrics"]
    return {
        "video_path": str(video_path),
        "run_dir": str(result["run_dir"]),
        "pipeline_mode": metrics.get("pipeline_mode"),
        "frames_read": metrics.get("frames_read"),
        "output_ready_frames": metrics.get("output_ready_frames"),
        "active_frames": metrics.get("active_frames"),
        "frames_written": metrics.get("frames_written"),
        "motion_active_frames": metrics.get("motion_active_frames"),
        "person_active_frames": metrics.get("person_active_frames"),
        "person_detector_frames": metrics.get("person_detector_frames"),
        "motion_event_count": metrics.get("motion_event_count"),
        "person_event_count": metrics.get("person_event_count"),
        "sia_activation_count": metrics.get("sia_activation_count"),
        "effective_fps": metrics.get("effective_fps"),
        "inference_mean_ms": metrics.get("timings", {}).get("inference", {}).get("mean_ms"),
        "active_loop_mean_ms": metrics.get("timings", {}).get("active_loop", {}).get("mean_ms"),
    }


def main():
    args = parse_args()
    raw_config = build_raw_config(args)
    if raw_config.get("pipeline_mode") != "motion_person_sia":
        raise RuntimeError(
            "full_pipeline_benchmark.py requires pipeline_mode='motion_person_sia'."
        )

    videos = resolve_video_paths(args)
    suite_dir = Path(args.output_dir or "results/full_pipeline/manual_run")
    ensure_dir(suite_dir)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    summary_rows = []
    for index, video_path in enumerate(videos, start=1):
        run_dir = suite_dir / f"{index:03d}_{sanitize_name(video_path.stem)}"
        video_config = dict(raw_config)
        video_config["video_path"] = str(video_path)
        print(f"[{index}/{len(videos)}] Running full pipeline benchmark for: {video_path}")
        result = run_offline_runtime(
            video_config,
            invoked_command,
            run_name=f"full_pipeline_{sanitize_name(video_path.stem)}",
            run_dir=run_dir,
        )
        summary_rows.append(build_summary_row(video_path, result))

    summary = {
        "suite_dir": str(suite_dir),
        "invoked_command": invoked_command,
        "video_count_completed": len(summary_rows),
        "videos": summary_rows,
    }
    write_csv(suite_dir / "benchmark_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    write_json(suite_dir / "benchmark_summary.json", summary)
    print(f"Full pipeline benchmark complete. Summary saved to: {suite_dir}")
    print(f"CSV summary: {suite_dir / 'benchmark_summary.csv'}")
    print(f"JSON summary: {suite_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
