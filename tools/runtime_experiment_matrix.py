import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, load_json, safe_pct_change, write_csv, write_json, write_run_summary
from tools.live_runtime_demo import run_live_runtime
from tools.offline_runtime_demo import run_offline_runtime


OFFLINE_MODE_CONFIGS = {
    "always_on": "configs/runtime_offline_always_on.json",
    "motion_only": "configs/runtime_offline_motion_only.json",
    "person_only": "configs/runtime_offline_person_only.json",
    "motion_person_sia": "configs/runtime_offline_motion_person_sia.json",
}

LIVE_MODE_BASE_CONFIGS = {
    "always_on": "configs/runtime_live_always_on.json",
    "motion_only": "configs/runtime_live_always_on.json",
    "person_only": "configs/runtime_live_always_on.json",
    "motion_person_sia": "configs/runtime_live.json",
}

DEFAULT_MODES = ["always_on", "motion_only", "person_only", "motion_person_sia"]
DEFAULT_PRECISIONS = ["fp32", "fp16"]

SUMMARY_FIELDNAMES = [
    "experiment_id",
    "source_mode",
    "pipeline_mode",
    "backend_name",
    "optimization_label",
    "precision",
    "autocast",
    "video_path",
    "simulate_live",
    "drop_frames",
    "source_fps",
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
    "effective_input_fps",
    "effective_active_fps",
    "inference_mean_ms",
    "inference_p95_ms",
    "postprocess_mean_ms",
    "render_mean_ms",
    "active_loop_mean_ms",
    "active_loop_p95_ms",
    "delta_vs_baseline_inference_mean_ms_pct",
    "delta_vs_baseline_active_loop_mean_ms_pct",
    "delta_vs_baseline_effective_active_fps_pct",
    "run_dir",
    "status",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a comparable experiment matrix across gate permutations and runtime variants."
    )
    parser.add_argument("--video", required=True, help="Input video path used for every experiment.")
    parser.add_argument(
        "--source-modes",
        default="offline,live_replay",
        help="Comma-separated source modes to run. Supported: offline, live_replay",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated pipeline modes to run.",
    )
    parser.add_argument(
        "--precisions",
        default=",".join(DEFAULT_PRECISIONS),
        help="Comma-separated precision modes to run. Supported: fp32, fp16",
    )
    parser.add_argument(
        "--output-dir",
        default="results/runtime/experiment_matrix",
        help="Output directory for experiment matrix artifacts.",
    )
    parser.add_argument("--weights", help="Optional override for the model checkpoint.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    parser.add_argument(
        "--live-target-fps",
        type=float,
        default=None,
        help="Optional FPS override when using live_replay.",
    )
    parser.add_argument(
        "--live-drop-frames",
        action="store_true",
        help="Drop replay frames when the live_replay path falls behind wall clock.",
    )
    parser.add_argument(
        "--backend-name",
        default="pytorch",
        help="Backend label recorded into the experiment metadata. Default: pytorch",
    )
    parser.add_argument(
        "--trt-engine-path",
        help="TensorRT engine path required when --backend-name=tensorrt.",
    )
    parser.add_argument(
        "--variant-prefix",
        default="shared_runtime",
        help="Prefix used when composing optimization_label values.",
    )
    return parser.parse_args()


def parse_csv_arg(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def active_fraction(metrics):
    output_ready = metrics.get("output_ready_frames") or 0
    active = metrics.get("active_frames") or 0
    if output_ready <= 0:
        return None
    return round(active / output_ready, 4)


def effective_input_fps(metrics):
    return metrics.get("effective_input_fps", metrics.get("effective_fps"))


def effective_active_fps(metrics):
    return metrics.get("effective_active_fps", metrics.get("effective_fps"))


def base_config_for(source_mode, pipeline_mode):
    if source_mode == "offline":
        path = OFFLINE_MODE_CONFIGS[pipeline_mode]
    elif source_mode == "live_replay":
        path = LIVE_MODE_BASE_CONFIGS[pipeline_mode]
    else:
        raise ValueError(f"Unsupported source mode '{source_mode}'.")
    return load_json(path)


def build_raw_config(args, source_mode, pipeline_mode, precision, video_path):
    raw_config = base_config_for(source_mode, pipeline_mode)
    raw_config["pipeline_mode"] = pipeline_mode
    raw_config["precision"] = precision
    raw_config["autocast"] = precision == "fp16"
    raw_config["video_path"] = str(video_path)
    raw_config["backend_name"] = args.backend_name
    if args.trt_engine_path:
        raw_config["trt_engine_path"] = args.trt_engine_path
    raw_config["optimization_label"] = f"{args.variant_prefix}_{precision}"
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.no_render:
        raw_config["render_enabled"] = False

    if source_mode == "live_replay":
        raw_config["mode"] = "live"
        raw_config["simulate_live"] = True
        raw_config["drop_frames"] = bool(args.live_drop_frames)
        raw_config["show_preview"] = False
        if args.live_target_fps is not None:
            raw_config["source_fps_override"] = args.live_target_fps

    return raw_config


def build_summary_row(experiment_id, source_mode, metrics, run_dir, status="ok", error=""):
    timings = metrics.get("timings", {})
    return {
        "experiment_id": experiment_id,
        "source_mode": source_mode,
        "pipeline_mode": metrics.get("pipeline_mode"),
        "backend_name": metrics.get("backend_name"),
        "optimization_label": metrics.get("optimization_label"),
        "precision": metrics.get("precision"),
        "autocast": metrics.get("autocast"),
        "video_path": metrics.get("video_path"),
        "simulate_live": metrics.get("simulate_live", False),
        "drop_frames": metrics.get("drop_frames", False),
        "source_fps": metrics.get("source_fps"),
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
        "effective_input_fps": effective_input_fps(metrics),
        "effective_active_fps": effective_active_fps(metrics),
        "inference_mean_ms": timings.get("inference", {}).get("mean_ms"),
        "inference_p95_ms": timings.get("inference", {}).get("p95_ms"),
        "postprocess_mean_ms": timings.get("postprocess", {}).get("mean_ms"),
        "render_mean_ms": timings.get("render", {}).get("mean_ms"),
        "active_loop_mean_ms": timings.get("active_loop", {}).get("mean_ms"),
        "active_loop_p95_ms": timings.get("active_loop", {}).get("p95_ms"),
        "delta_vs_baseline_inference_mean_ms_pct": None,
        "delta_vs_baseline_active_loop_mean_ms_pct": None,
        "delta_vs_baseline_effective_active_fps_pct": None,
        "run_dir": str(run_dir),
        "status": status,
        "error": error,
    }


def apply_baseline_deltas(rows):
    baselines = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["source_mode"], row["backend_name"])
        if row["pipeline_mode"] == "always_on" and row["precision"] == "fp32":
            baselines[key] = row

    for row in rows:
        baseline = baselines.get((row["source_mode"], row["backend_name"]))
        if baseline is None or row["status"] != "ok":
            continue
        row["delta_vs_baseline_inference_mean_ms_pct"] = safe_pct_change(
            baseline["inference_mean_ms"],
            row["inference_mean_ms"],
        )
        row["delta_vs_baseline_active_loop_mean_ms_pct"] = safe_pct_change(
            baseline["active_loop_mean_ms"],
            row["active_loop_mean_ms"],
        )
        row["delta_vs_baseline_effective_active_fps_pct"] = safe_pct_change(
            baseline["effective_active_fps"],
            row["effective_active_fps"],
        )


def build_notes(rows):
    lines = [
        "# Runtime Experiment Matrix",
        "",
        "This package records comparable runs across source modes, gate permutations, and optimization variants.",
        "",
        "Key comparison rule:",
        "- treat `always_on + fp32` as the baseline for each source mode and backend",
        "- compare later FP16 and eventual INT8 rows against that baseline using the delta columns in `matrix_summary.csv`",
        "",
        "Artifacts:",
        "- matrix_summary.csv",
        "- matrix_summary.json",
        "- runs/",
    ]
    successful = [row for row in rows if row["status"] == "ok"]
    if successful:
        lines.extend(
            [
                "",
                "Recorded dimensions:",
                "- source mode",
                "- pipeline mode",
                "- backend name",
                "- optimization label",
                "- precision and autocast",
                "- active fraction, activation counts, and gate activity",
                "- inference, postprocess, render, and active-loop timing",
            ]
        )
    return lines


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise RuntimeError(f"Video '{args.video}' does not exist or is not a file.")

    source_modes = parse_csv_arg(args.source_modes)
    modes = parse_csv_arg(args.modes)
    precisions = parse_csv_arg(args.precisions)
    supported_source_modes = {"offline", "live_replay"}
    supported_precisions = {"fp32", "fp16"}

    invalid_source_modes = sorted(set(source_modes) - supported_source_modes)
    invalid_precisions = sorted(set(precisions) - supported_precisions)
    invalid_modes = sorted(set(modes) - set(DEFAULT_MODES))
    if invalid_source_modes:
        raise RuntimeError(f"Unsupported source modes: {invalid_source_modes}")
    if invalid_precisions:
        raise RuntimeError(f"Unsupported precisions: {invalid_precisions}")
    if args.backend_name == "tensorrt":
        if not args.trt_engine_path:
            raise RuntimeError("--trt-engine-path is required when --backend-name=tensorrt.")
        invalid_trt_precisions = sorted(set(precisions) - {"fp16"})
        if invalid_trt_precisions:
            raise RuntimeError(
                f"TensorRT experiment matrix currently supports only fp16 precision, got: {invalid_trt_precisions}"
            )
    if invalid_modes:
        raise RuntimeError(f"Unsupported pipeline modes: {invalid_modes}")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    runs_dir = output_dir / "runs"
    ensure_dir(runs_dir)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    rows = []
    detailed_runs = []
    failures = []
    experiments = [
        (source_mode, pipeline_mode, precision)
        for source_mode in source_modes
        for pipeline_mode in modes
        for precision in precisions
    ]

    for index, (source_mode, pipeline_mode, precision) in enumerate(experiments, start=1):
        experiment_id = f"{index:02d}_{source_mode}_{pipeline_mode}_{precision}"
        raw_config = build_raw_config(args, source_mode, pipeline_mode, precision, video_path)
        run_dir = runs_dir / experiment_id
        print(f"[{index}/{len(experiments)}] Running {experiment_id}")
        try:
            if source_mode == "offline":
                result = run_offline_runtime(
                    raw_config,
                    invoked_command,
                    run_name=experiment_id,
                    run_dir=run_dir,
                )
            else:
                result = run_live_runtime(
                    raw_config,
                    invoked_command,
                    run_name=experiment_id,
                    run_dir=run_dir,
                )
            row = build_summary_row(experiment_id, source_mode, result["metrics"], result["run_dir"])
            rows.append(row)
            detailed_runs.append(
                {
                    "experiment_id": experiment_id,
                    "source_mode": source_mode,
                    "pipeline_mode": pipeline_mode,
                    "precision": precision,
                    "run_dir": str(result["run_dir"]),
                    "metrics": result["metrics"],
                }
            )
        except Exception as exc:
            error_text = str(exc)
            failures.append({"experiment_id": experiment_id, "error": error_text})
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "source_mode": source_mode,
                    "pipeline_mode": pipeline_mode,
                    "backend_name": args.backend_name,
                    "optimization_label": f"{args.variant_prefix}_{precision}",
                    "precision": precision,
                    "autocast": precision == "fp16",
                    "video_path": str(video_path),
                    "simulate_live": source_mode == "live_replay",
                    "drop_frames": bool(args.live_drop_frames),
                    "source_fps": args.live_target_fps,
                    "frames_read": None,
                    "output_ready_frames": None,
                    "active_frames": None,
                    "active_fraction_output_ready": None,
                    "frames_written": None,
                    "motion_active_frames": None,
                    "person_active_frames": None,
                    "person_detector_frames": None,
                    "motion_event_count": None,
                    "person_event_count": None,
                    "sia_activation_count": None,
                    "sia_stride_wait_frames": None,
                    "effective_input_fps": None,
                    "effective_active_fps": None,
                    "inference_mean_ms": None,
                    "inference_p95_ms": None,
                    "postprocess_mean_ms": None,
                    "render_mean_ms": None,
                    "active_loop_mean_ms": None,
                    "active_loop_p95_ms": None,
                    "delta_vs_baseline_inference_mean_ms_pct": None,
                    "delta_vs_baseline_active_loop_mean_ms_pct": None,
                    "delta_vs_baseline_effective_active_fps_pct": None,
                    "run_dir": str(run_dir),
                    "status": "error",
                    "error": error_text,
                }
            )

    apply_baseline_deltas(rows)

    payload = {
        "video_path": str(video_path),
        "invoked_command": invoked_command,
        "backend_name": args.backend_name,
        "variant_prefix": args.variant_prefix,
        "rows": rows,
        "detailed_runs": detailed_runs,
        "failures": failures,
    }
    write_csv(output_dir / "matrix_summary.csv", SUMMARY_FIELDNAMES, rows)
    write_json(output_dir / "matrix_summary.json", payload)
    write_run_summary(output_dir / "README.md", build_notes(rows))

    print(f"Runtime experiment matrix saved to: {output_dir}")
    print(f"CSV summary: {output_dir / 'matrix_summary.csv'}")
    print(f"JSON summary: {output_dir / 'matrix_summary.json'}")
    if failures:
        raise RuntimeError(f"Experiment matrix completed with failures: {failures[0]['experiment_id']}")


if __name__ == "__main__":
    main()
