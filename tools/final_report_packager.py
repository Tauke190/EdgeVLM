import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, load_json, write_csv, write_json, write_run_summary
from tools.offline_runtime_demo import run_offline_runtime


MODE_CONFIGS = [
    ("always_on", "configs/runtime_offline_always_on.json"),
    ("motion_only", "configs/runtime_offline_motion_only.json"),
    ("person_only", "configs/runtime_offline_person_only.json"),
    ("motion_person_sia", "configs/runtime_offline_motion_person_sia.json"),
]

FINAL_TABLE_FIELDNAMES = [
    "mode",
    "video_path",
    "frames_read",
    "output_ready_frames",
    "active_frames",
    "active_fraction_output_ready",
    "frames_written",
    "motion_active_frames",
    "person_active_frames",
    "person_detector_frames",
    "motion_event_count",
    "person_event_count",
    "sia_activation_count",
    "sia_stride_wait_frames",
    "sia_trigger_reason_counts",
    "effective_fps",
    "inference_mean_ms",
    "active_loop_mean_ms",
    "motion_to_sia_latency_frames",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final multi-mode comparisons and package presentation-friendly artifacts."
    )
    parser.add_argument("--video", required=True, help="Input video path to compare across preset modes.")
    parser.add_argument(
        "--output-dir",
        default="results/final_package",
        help="Output directory for final package artifacts. Default: results/final_package",
    )
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    return parser.parse_args()


def load_config(path):
    return load_json(path)


def active_fraction(metrics):
    output_ready = metrics.get("output_ready_frames") or 0
    active = metrics.get("active_frames") or 0
    if output_ready <= 0:
        return None
    return round(active / output_ready, 4)


def build_table_row(mode_name, video_path, result):
    metrics = result["metrics"]
    return {
        "mode": mode_name,
        "video_path": str(video_path),
        "frames_read": metrics.get("frames_read"),
        "output_ready_frames": metrics.get("output_ready_frames"),
        "active_frames": metrics.get("active_frames"),
        "active_fraction_output_ready": active_fraction(metrics),
        "frames_written": metrics.get("frames_written"),
        "motion_active_frames": metrics.get("motion_active_frames"),
        "person_active_frames": metrics.get("person_active_frames"),
        "person_detector_frames": metrics.get("person_detector_frames"),
        "motion_event_count": metrics.get("motion_event_count"),
        "person_event_count": metrics.get("person_event_count"),
        "sia_activation_count": metrics.get("sia_activation_count"),
        "sia_stride_wait_frames": metrics.get("sia_stride_wait_frames"),
        "sia_trigger_reason_counts": metrics.get("sia_trigger_reason_counts"),
        "effective_fps": metrics.get("effective_fps"),
        "inference_mean_ms": metrics.get("timings", {}).get("inference", {}).get("mean_ms"),
        "active_loop_mean_ms": metrics.get("timings", {}).get("active_loop", {}).get("mean_ms"),
        "motion_to_sia_latency_frames": metrics.get("motion_to_sia_latency_frames"),
    }


def format_pct(delta):
    if delta is None:
        return "n/a"
    return f"{delta:+.1f}%"


def pct_delta(baseline, candidate):
    if baseline in (None, 0) or candidate is None:
        return None
    return (candidate - baseline) / baseline * 100.0


def build_talking_points(rows):
    by_mode = {row["mode"]: row for row in rows}
    always_on = by_mode.get("always_on")
    motion_only = by_mode.get("motion_only")
    person_only = by_mode.get("person_only")
    full = by_mode.get("motion_person_sia")

    lines = [
        "# Final Talking Points",
        "",
        "## System Summary",
        "- The final shared runtime supports four comparable modes on the same video path: always-on, motion-only, person-only, and motion+person+SiA.",
        "- Motion and person gates now run inside the shared modular pipeline rather than only in prototype scripts.",
        "",
        "## Quantitative Highlights",
    ]

    if always_on and full:
        lines.append(
            f"- Compared with always-on, the motion+person+SiA mode reduced the expensive-tier active fraction from {always_on['active_fraction_output_ready']} to {full['active_fraction_output_ready']}."
        )
        lines.append(
            f"- SiA activations changed from {always_on['sia_activation_count']} in always-on to {full['sia_activation_count']} in motion+person+SiA."
        )
        lines.append(
            f"- Effective expensive-tier throughput changed by {format_pct(pct_delta(always_on['effective_fps'], full['effective_fps']))} relative to always-on."
        )
    if motion_only and person_only:
        lines.append(
            f"- Motion-only kept the model active for {motion_only['active_fraction_output_ready']} of output-ready frames, while person-only kept it active for {person_only['active_fraction_output_ready']}."
        )
    if full:
        lines.append(
            f"- The full pipeline recorded trigger reasons {full['sia_trigger_reason_counts']} and mean active-loop time {full['active_loop_mean_ms']} ms."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- The staged pipeline preserves continuous output playback while controlling when the expensive SiA tier is allowed to run.",
            "- Motion detection provides the cheap wake-up signal, person detection confirms semantic relevance, and SiA runs only when enough evidence and temporal context are available.",
            "- The current shared runtime is sufficient to demonstrate the staged-pipeline idea on Jetson without needing to claim DeepStream integration.",
        ]
    )
    return lines


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise RuntimeError(f"Video '{args.video}' does not exist or is not a file.")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    runs_dir = output_dir / "mode_runs"
    ensure_dir(runs_dir)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    rows = []
    detailed = []
    for index, (mode_name, config_path) in enumerate(MODE_CONFIGS, start=1):
        raw_config = load_config(config_path)
        raw_config["video_path"] = str(video_path)
        if args.weights:
            raw_config["weights_path"] = args.weights
        if args.max_frames is not None:
            raw_config["max_frames"] = args.max_frames
        if args.no_render:
            raw_config["render_enabled"] = False
        run_dir = runs_dir / f"{index:02d}_{mode_name}"
        print(f"[{index}/{len(MODE_CONFIGS)}] Running mode comparison for: {mode_name}")
        result = run_offline_runtime(
            raw_config,
            invoked_command,
            run_name=f"final_package_{mode_name}",
            run_dir=run_dir,
        )
        rows.append(build_table_row(mode_name, video_path, result))
        detailed.append(
            {
                "mode": mode_name,
                "config_path": config_path,
                "run_dir": str(run_dir),
                "metrics": result["metrics"],
            }
        )

    write_csv(output_dir / "final_metrics_table.csv", FINAL_TABLE_FIELDNAMES, rows)
    write_json(
        output_dir / "final_metrics_table.json",
        {
            "video_path": str(video_path),
            "invoked_command": invoked_command,
            "rows": rows,
            "detailed_runs": detailed,
        },
    )

    talking_points_lines = build_talking_points(rows)
    write_run_summary(output_dir / "final_talking_points.md", talking_points_lines)
    write_run_summary(
        output_dir / "README.md",
        [
            "# Final Package",
            "",
            f"Source video: {video_path}",
            f"Command: {invoked_command}",
            "",
            "Artifacts:",
            "- final_metrics_table.csv",
            "- final_metrics_table.json",
            "- final_talking_points.md",
            "- mode_runs/",
        ],
    )

    print(f"Final package saved to: {output_dir}")
    print(f"Metrics table: {output_dir / 'final_metrics_table.csv'}")
    print(f"Talking points: {output_dir / 'final_talking_points.md'}")


if __name__ == "__main__":
    main()
