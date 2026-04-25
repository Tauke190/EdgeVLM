import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and compare the trusted offline baseline and the shared offline runtime on the same clip."
    )
    parser.add_argument("--baseline-config", required=True, help="Path to the baseline config JSON.")
    parser.add_argument("--runtime-config", required=True, help="Path to the runtime config JSON.")
    parser.add_argument("--video", required=True, help="Input video path for both runs.")
    parser.add_argument("--max-frames", type=int, default=120, help="Frame cap for both runs.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering for both runs. Default is disabled to isolate core parity.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the parity report as JSON.",
    )
    return parser.parse_args()


def run_command(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")


def latest_run_dir(root: Path):
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No run directories found under '{root}'.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_metrics(run_dir: Path):
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stage_rows(run_dir: Path):
    with open(run_dir / "stage_timings.csv", "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_active_frame(rows):
    for row in rows:
        if row["active_iteration"] == "1":
            return int(row["frame_index"])
    return None


def compare_stage_rows(baseline_rows, runtime_rows):
    mismatches = []
    shared_count = min(len(baseline_rows), len(runtime_rows))
    for index in range(shared_count):
        baseline_row = baseline_rows[index]
        runtime_row = runtime_rows[index]
        if (
            baseline_row["active_iteration"] != runtime_row["active_iteration"]
            or baseline_row["detections"] != runtime_row["detections"]
        ):
            mismatches.append(
                {
                    "row_index": index + 1,
                    "baseline_frame_index": int(baseline_row["frame_index"]),
                    "runtime_frame_index": int(runtime_row["frame_index"]),
                    "baseline_active_iteration": int(baseline_row["active_iteration"]),
                    "runtime_active_iteration": int(runtime_row["active_iteration"]),
                    "baseline_detections": int(baseline_row["detections"]),
                    "runtime_detections": int(runtime_row["detections"]),
                }
            )
    return mismatches


def build_report(baseline_run, runtime_run):
    baseline_metrics = load_metrics(baseline_run)
    runtime_metrics = load_metrics(runtime_run)
    baseline_rows = load_stage_rows(baseline_run)
    runtime_rows = load_stage_rows(runtime_run)
    mismatches = compare_stage_rows(baseline_rows, runtime_rows)

    report = {
        "baseline_run": str(baseline_run),
        "runtime_run": str(runtime_run),
        "baseline": {
            "frames_read": baseline_metrics.get("frames_read"),
            "clips_processed": baseline_metrics.get("clips_processed"),
            "frames_written": baseline_metrics.get("frames_written"),
            "effective_fps": baseline_metrics.get("effective_fps"),
            "first_active_frame": first_active_frame(baseline_rows),
            "active_rows": sum(row["active_iteration"] == "1" for row in baseline_rows),
            "detection_sum": sum(int(row["detections"]) for row in baseline_rows),
        },
        "runtime": {
            "frames_read": runtime_metrics.get("frames_read"),
            "clips_processed": runtime_metrics.get("clips_processed"),
            "frames_written": runtime_metrics.get("frames_written"),
            "effective_fps": runtime_metrics.get("effective_fps"),
            "first_active_frame": first_active_frame(runtime_rows),
            "active_rows": sum(row["active_iteration"] == "1" for row in runtime_rows),
            "detection_sum": sum(int(row["detections"]) for row in runtime_rows),
        },
        "row_count_match": len(baseline_rows) == len(runtime_rows),
        "shared_row_count": min(len(baseline_rows), len(runtime_rows)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
    }
    return report


def print_report(report):
    print("Runtime Parity Check")
    print(f"Baseline run: {report['baseline_run']}")
    print(f"Runtime run:  {report['runtime_run']}")
    print("")
    print(
        "Baseline summary: "
        f"frames_read={report['baseline']['frames_read']}, "
        f"clips_processed={report['baseline']['clips_processed']}, "
        f"frames_written={report['baseline']['frames_written']}, "
        f"first_active_frame={report['baseline']['first_active_frame']}, "
        f"detection_sum={report['baseline']['detection_sum']}"
    )
    print(
        "Runtime summary:  "
        f"frames_read={report['runtime']['frames_read']}, "
        f"clips_processed={report['runtime']['clips_processed']}, "
        f"frames_written={report['runtime']['frames_written']}, "
        f"first_active_frame={report['runtime']['first_active_frame']}, "
        f"detection_sum={report['runtime']['detection_sum']}"
    )
    print(
        "Row comparison: "
        f"row_count_match={report['row_count_match']}, "
        f"shared_row_count={report['shared_row_count']}, "
        f"mismatch_count={report['mismatch_count']}"
    )
    if report["mismatches"]:
        print("First mismatches:")
        for mismatch in report["mismatches"]:
            print(
                "  "
                f"row={mismatch['row_index']} "
                f"baseline(frame={mismatch['baseline_frame_index']}, active={mismatch['baseline_active_iteration']}, det={mismatch['baseline_detections']}) "
                f"runtime(frame={mismatch['runtime_frame_index']}, active={mismatch['runtime_active_iteration']}, det={mismatch['runtime_detections']})"
            )


def main():
    args = parse_args()
    baseline_root = REPO_ROOT / "results" / "baseline"
    runtime_root = REPO_ROOT / "results" / "runtime"
    baseline_before = latest_run_dir(baseline_root) if baseline_root.exists() and any(baseline_root.iterdir()) else None
    runtime_before = latest_run_dir(runtime_root) if runtime_root.exists() and any(runtime_root.iterdir()) else None

    common_flags = ["--video", args.video, "--max-frames", str(args.max_frames)]
    if not args.render:
        common_flags.append("--no-render")

    run_command(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "tools/baseline_runner.py",
            "--config",
            args.baseline_config,
            *common_flags,
        ]
    )
    run_command(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "tools/offline_runtime_demo.py",
            "--config",
            args.runtime_config,
            *common_flags,
        ]
    )

    baseline_run = latest_run_dir(baseline_root)
    runtime_run = latest_run_dir(runtime_root)
    if baseline_before is not None and baseline_run == baseline_before:
        raise RuntimeError("Baseline parity run did not produce a new output directory.")
    if runtime_before is not None and runtime_run == runtime_before:
        raise RuntimeError("Runtime parity run did not produce a new output directory.")

    report = build_report(baseline_run, runtime_run)
    print_report(report)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
