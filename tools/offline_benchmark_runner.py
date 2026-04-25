import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, timestamp_slug, write_csv, write_json
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
        try:
            result = run_offline_runtime(
                video_config,
                invoked_command,
                run_name=f"offline_runtime_demo_{video_label}",
                run_dir=run_dir,
            )
            metrics = result["metrics"]
            rows.append(
                {
                    "video_path": str(video_path),
                    "run_dir": str(result["run_dir"]),
                    "frames_read": metrics["frames_read"],
                    "active_frames": metrics["active_frames"],
                    "clips_processed": metrics["clips_processed"],
                    "frames_written": metrics["frames_written"],
                    "effective_fps": metrics["effective_fps"],
                    "inference_mean_ms": metrics["timings"]["inference"]["mean_ms"],
                    "postprocess_mean_ms": metrics["timings"]["postprocess"]["mean_ms"],
                    "render_mean_ms": metrics["timings"]["render"]["mean_ms"],
                    "active_loop_mean_ms": metrics["timings"]["active_loop"]["mean_ms"],
                    "output_video": str(result["output_video_path"]) if result["output_video_path"] else "disabled",
                    "status": "ok",
                    "error": "",
                }
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
        "video_count_completed": sum(row["status"] == "ok" for row in rows),
        "video_count_failed": sum(row["status"] == "error" for row in rows),
        "failures": failures,
        "videos": rows,
    }
    write_csv(
        suite_dir / "benchmark_summary.csv",
        [
            "video_path",
            "run_dir",
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
