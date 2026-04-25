import argparse
from pathlib import Path
import shlex
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, load_json, timestamp_slug, write_csv, write_json
from tools.offline_runtime_demo import build_raw_config, run_offline_runtime


DEFAULT_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teammate-facing offline benchmark harness for the shared runtime."
    )
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video", help="Run one specific input video.")
    parser.add_argument("--video-dir", help="Run every supported video under a directory.")
    parser.add_argument(
        "--glob",
        default="*",
        help="Filename pattern to apply inside --video-dir. Default: '*'",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --video-dir.",
    )
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--output-dir", help="Optional explicit suite output directory.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from each source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that already have a matching completed run in the suite output directory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=60,
        help="Print progress every N processed frames for each video. Default: 60",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining videos if one input fails.",
    )
    return parser.parse_args()


def sanitize_name(value):
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "video"


def discover_videos(video_dir, pattern, recursive):
    root = Path(video_dir)
    if not root.is_dir():
        raise RuntimeError(f"Video directory '{video_dir}' does not exist or is not a directory.")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def resolve_video_paths(args):
    if bool(args.video) == bool(args.video_dir):
        raise RuntimeError("Provide exactly one of --video or --video-dir.")
    if args.video:
        path = Path(args.video)
        if not path.is_file():
            raise RuntimeError(f"Video '{args.video}' does not exist or is not a file.")
        return [path]

    videos = discover_videos(args.video_dir, args.glob, args.recursive)
    if not videos:
        raise RuntimeError(
            f"No supported videos found under '{args.video_dir}' with pattern '{args.glob}'."
        )
    return videos


def suite_output_dir(args, raw_config):
    if args.output_dir:
        path = Path(args.output_dir)
        ensure_dir(path)
        return path
    output_root = Path(args.output_root or raw_config.get("output_root", "results/runtime"))
    path = output_root / f"{timestamp_slug()}_offline_benchmark_suite"
    ensure_dir(path)
    return path


def progress_label(frame_index, total_frames, max_frames):
    if max_frames:
        effective_total = min(total_frames, max_frames) if total_frames else max_frames
    else:
        effective_total = total_frames
    if effective_total:
        return f"{frame_index}/{effective_total}"
    return str(frame_index)


def output_video_path_for_run(run_dir, config_payload):
    output_name = config_payload.get("output_video_name", "runtime_offline.mp4")
    return Path(run_dir) / output_name


def load_existing_run(run_dir):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        return None
    return {
        "config": load_json(config_path),
        "metrics": load_json(metrics_path),
    }


def matching_completed_run(existing_run, expected_config, run_dir):
    if existing_run is None:
        return False
    existing_config = existing_run["config"]
    existing_metrics = existing_run["metrics"]
    checks = [
        existing_config.get("video_path") == expected_config.get("video_path"),
        existing_config.get("weights_path") == expected_config.get("weights_path"),
        existing_config.get("max_frames") == expected_config.get("max_frames"),
        existing_config.get("pipeline_mode", "always_on") == expected_config.get("pipeline_mode", "always_on"),
        bool(existing_config.get("render_enabled", True)) == bool(expected_config.get("render_enabled", True)),
        existing_config.get("precision", expected_config.get("precision")) == expected_config.get("precision"),
        bool(existing_config.get("autocast", expected_config.get("autocast", False)))
        == bool(expected_config.get("autocast", False)),
        existing_config.get("person_detector", expected_config.get("person_detector"))
        == expected_config.get("person_detector"),
        existing_config.get("person_weights", expected_config.get("person_weights"))
        == expected_config.get("person_weights"),
        existing_config.get("person_threshold", expected_config.get("person_threshold"))
        == expected_config.get("person_threshold"),
        existing_config.get("person_precision", expected_config.get("person_precision"))
        == expected_config.get("person_precision"),
        existing_config.get("person_stride", expected_config.get("person_stride"))
        == expected_config.get("person_stride"),
        existing_config.get("person_cooldown_frames", expected_config.get("person_cooldown_frames"))
        == expected_config.get("person_cooldown_frames"),
        existing_config.get("person_hit_threshold", expected_config.get("person_hit_threshold"))
        == expected_config.get("person_hit_threshold"),
        existing_config.get("person_resize_width", expected_config.get("person_resize_width"))
        == expected_config.get("person_resize_width"),
        existing_config.get("person_min_box_area", expected_config.get("person_min_box_area"))
        == expected_config.get("person_min_box_area"),
        existing_metrics.get("frames_read") is not None,
    ]
    if not all(checks):
        return False
    if expected_config.get("render_enabled", True):
        return output_video_path_for_run(run_dir, expected_config).is_file()
    return True


def summary_row(video_path, run_dir, metrics, output_video_path, status="ok", error=""):
    return {
        "video_path": str(video_path),
        "run_dir": str(run_dir),
        "pipeline_mode": metrics.get("pipeline_mode"),
        "frames_read": metrics.get("frames_read"),
        "active_frames": metrics.get("active_frames"),
        "clips_processed": metrics.get("clips_processed"),
        "frames_written": metrics.get("frames_written"),
        "effective_fps": metrics.get("effective_fps"),
        "inference_mean_ms": metrics.get("timings", {}).get("inference", {}).get("mean_ms"),
        "postprocess_mean_ms": metrics.get("timings", {}).get("postprocess", {}).get("mean_ms"),
        "render_mean_ms": metrics.get("timings", {}).get("render", {}).get("mean_ms"),
        "active_loop_mean_ms": metrics.get("timings", {}).get("active_loop", {}).get("mean_ms"),
        "output_video": str(output_video_path) if output_video_path else "disabled",
        "status": status,
        "error": error,
    }


def main():
    args = parse_args()
    raw_config = build_raw_config(args)
    video_paths = resolve_video_paths(args)
    suite_dir = suite_output_dir(args, raw_config)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    rows = []
    failures = []
    for index, video_path in enumerate(video_paths, start=1):
        video_label = sanitize_name(video_path.stem)
        run_dir = suite_dir / f"{index:03d}_{video_label}"
        video_config = dict(raw_config)
        video_config["video_path"] = str(video_path)
        print(f"[{index}/{len(video_paths)}] Running offline benchmark for: {video_path}")
        if args.no_render:
            print("  Rendering disabled. Output video will not be written for this run.")

        last_progress_time = 0.0

        def progress_callback(payload):
            nonlocal last_progress_time
            event = payload["event"]
            if event == "start":
                total_hint = payload["frame_count_hint"]
                max_frames = payload["max_frames"]
                total_msg = f", source frames={total_hint}" if total_hint else ""
                max_msg = f", max_frames={max_frames}" if max_frames else ""
                print(f"  Run directory: {payload['run_dir']}{total_msg}{max_msg}")
                return
            if event == "frame":
                frame_index = payload["frame_index"]
                should_print = frame_index == 1 or frame_index % max(1, args.progress_every) == 0
                now = time.perf_counter()
                if not should_print and now - last_progress_time < 10.0:
                    return
                last_progress_time = now
                label = progress_label(
                    frame_index,
                    payload["frame_count_hint"],
                    payload["max_frames"],
                )
                print(
                    "  Progress: "
                    f"frame {label}, "
                    f"active_frames={payload['active_frames']}, "
                    f"clips_processed={payload['clips_processed']}, "
                    f"frames_written={payload['frames_written']}"
                )
                return
            if event == "complete":
                output_video = payload["output_video_path"] or "disabled"
                print(
                    "  Completed: "
                    f"frames_read={payload['frames_read']}, "
                    f"active_frames={payload['active_frames']}, "
                    f"clips_processed={payload['clips_processed']}, "
                    f"frames_written={payload['frames_written']}, "
                    f"effective_fps={payload['effective_fps']}, "
                    f"output_video={output_video}"
                )

        try:
            existing_run = load_existing_run(run_dir) if args.resume else None
            if matching_completed_run(existing_run, video_config, run_dir):
                output_video = (
                    output_video_path_for_run(run_dir, video_config)
                    if video_config.get("render_enabled", True)
                    else None
                )
                print(f"  Resume mode: skipping existing completed run in {run_dir}")
                rows.append(
                    summary_row(
                        video_path,
                        run_dir,
                        existing_run["metrics"],
                        output_video,
                        status="skipped_existing",
                        error="",
                    )
                )
                continue

            result = run_offline_runtime(
                video_config,
                invoked_command,
                run_name=f"offline_runtime_demo_{video_label}",
                run_dir=run_dir,
                progress_callback=progress_callback,
            )
            rows.append(
                summary_row(
                    video_path,
                    result["run_dir"],
                    result["metrics"],
                    result["output_video_path"],
                    status="ok",
                    error="",
                )
            )
        except Exception as exc:
            failures.append({"video_path": str(video_path), "error": str(exc)})
            rows.append(
                {
                    "video_path": str(video_path),
                    "run_dir": str(run_dir),
                    "frames_read": None,
                    "active_frames": None,
                    "clips_processed": None,
                    "frames_written": None,
                    "effective_fps": None,
                    "inference_mean_ms": None,
                    "postprocess_mean_ms": None,
                    "render_mean_ms": None,
                    "active_loop_mean_ms": None,
                    "output_video": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                break

    summary = {
        "suite_dir": str(suite_dir),
        "invoked_command": invoked_command,
        "video_count_requested": len(video_paths),
        "video_count_completed": sum(row["status"] in {"ok", "skipped_existing"} for row in rows),
        "video_count_skipped": sum(row["status"] == "skipped_existing" for row in rows),
        "video_count_failed": sum(row["status"] == "error" for row in rows),
        "failures": failures,
        "videos": rows,
    }
    write_csv(
        suite_dir / "benchmark_summary.csv",
        [
            "video_path",
            "run_dir",
            "pipeline_mode",
            "frames_read",
            "active_frames",
            "clips_processed",
            "frames_written",
            "effective_fps",
            "inference_mean_ms",
            "postprocess_mean_ms",
            "render_mean_ms",
            "active_loop_mean_ms",
            "output_video",
            "status",
            "error",
        ],
        rows,
    )
    write_json(suite_dir / "benchmark_summary.json", summary)
    print(f"Offline benchmark suite complete. Summary saved to: {suite_dir}")
    print(f"CSV summary: {suite_dir / 'benchmark_summary.csv'}")
    print(f"JSON summary: {suite_dir / 'benchmark_summary.json'}")
    if failures and not args.continue_on_error:
        raise RuntimeError(f"Benchmark suite stopped after failure on '{failures[0]['video_path']}'.")


if __name__ == "__main__":
    main()
