import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, infer_git_commit, safe_pct_change, write_csv, write_json, write_run_summary
from tools.live_runtime_demo import run_live_runtime
from tools.offline_runtime_demo import run_offline_runtime
from tools.runtime_experiment_matrix import DEFAULT_MODES, build_summary_row, parse_csv_arg, base_config_for


SUMMARY_FIELDNAMES = [
    "experiment_id",
    "comparison_label",
    "source_mode",
    "pipeline_mode",
    "backend_name",
    "optimization_label",
    "precision",
    "autocast",
    "weights_path",
    "actions_json",
    "trt_engine_path",
    "video_path",
    "max_frames",
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
    "delta_vs_pytorch_fp32_inference_mean_ms_pct",
    "delta_vs_pytorch_fp32_active_loop_mean_ms_pct",
    "delta_vs_pytorch_fp32_effective_active_fps_pct",
    "delta_vs_pytorch_fp16_inference_mean_ms_pct",
    "delta_vs_pytorch_fp16_active_loop_mean_ms_pct",
    "delta_vs_pytorch_fp16_effective_active_fps_pct",
    "run_dir",
    "status",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible backend comparison package across PyTorch and TensorRT shared-runtime variants."
    )
    parser.add_argument("--video", required=True, help="Input video path used for every comparison run.")
    parser.add_argument(
        "--source-modes",
        default="offline,live_replay",
        help="Comma-separated source modes to run. Supported: offline, live_replay",
    )
    parser.add_argument(
        "--modes",
        default="always_on,motion_person_sia",
        help="Comma-separated pipeline modes to run.",
    )
    parser.add_argument(
        "--pytorch-precisions",
        default="fp32,fp16",
        help="Comma-separated PyTorch precisions to include. Supported: fp32, fp16",
    )
    parser.add_argument(
        "--include-tensorrt",
        action="store_true",
        help="Include the TensorRT FP16 backend in the comparison package.",
    )
    parser.add_argument(
        "--trt-engine-path",
        help="TensorRT engine path used when --include-tensorrt is set.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/runtime/backend_compare",
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument("--weights", help="Optional checkpoint override.")
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
    return parser.parse_args()


def validate_args(args):
    video_path = Path(args.video)
    if not video_path.is_file():
        raise RuntimeError(f"Video '{args.video}' does not exist or is not a file.")

    source_modes = parse_csv_arg(args.source_modes)
    modes = parse_csv_arg(args.modes)
    pytorch_precisions = parse_csv_arg(args.pytorch_precisions)
    supported_source_modes = {"offline", "live_replay"}
    supported_precisions = {"fp32", "fp16"}

    invalid_source_modes = sorted(set(source_modes) - supported_source_modes)
    invalid_modes = sorted(set(modes) - set(DEFAULT_MODES))
    invalid_precisions = sorted(set(pytorch_precisions) - supported_precisions)
    if invalid_source_modes:
        raise RuntimeError(f"Unsupported source modes: {invalid_source_modes}")
    if invalid_modes:
        raise RuntimeError(f"Unsupported pipeline modes: {invalid_modes}")
    if invalid_precisions:
        raise RuntimeError(f"Unsupported PyTorch precisions: {invalid_precisions}")
    if args.include_tensorrt and not args.trt_engine_path:
        raise RuntimeError("--trt-engine-path is required when --include-tensorrt is set.")
    return video_path, source_modes, modes, pytorch_precisions


def build_raw_config(args, source_mode, pipeline_mode, backend_name, precision, video_path):
    raw_config = base_config_for(source_mode, pipeline_mode)
    raw_config["pipeline_mode"] = pipeline_mode
    raw_config["backend_name"] = backend_name
    raw_config["precision"] = precision
    raw_config["autocast"] = precision == "fp16"
    raw_config["video_path"] = str(video_path)
    raw_config["optimization_label"] = f"backend_compare_{backend_name}_{precision}"
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.no_render:
        raw_config["render_enabled"] = False
    if backend_name == "tensorrt":
        raw_config["trt_engine_path"] = args.trt_engine_path
    if source_mode == "live_replay":
        raw_config["mode"] = "live"
        raw_config["simulate_live"] = True
        raw_config["drop_frames"] = bool(args.live_drop_frames)
        raw_config["show_preview"] = False
        if args.live_target_fps is not None:
            raw_config["source_fps_override"] = args.live_target_fps
    return raw_config


def build_variant_specs(pytorch_precisions, include_tensorrt):
    variants = [("pytorch", precision) for precision in pytorch_precisions]
    if include_tensorrt:
        variants.append(("tensorrt", "fp16"))
    return variants


def augment_summary_row(row, raw_config, comparison_label):
    row = dict(row)
    row["comparison_label"] = comparison_label
    row["weights_path"] = raw_config.get("weights_path")
    row["actions_json"] = raw_config.get("actions_json")
    row["trt_engine_path"] = raw_config.get("trt_engine_path")
    row["max_frames"] = raw_config.get("max_frames")
    row["delta_vs_pytorch_fp32_inference_mean_ms_pct"] = None
    row["delta_vs_pytorch_fp32_active_loop_mean_ms_pct"] = None
    row["delta_vs_pytorch_fp32_effective_active_fps_pct"] = None
    row["delta_vs_pytorch_fp16_inference_mean_ms_pct"] = None
    row["delta_vs_pytorch_fp16_active_loop_mean_ms_pct"] = None
    row["delta_vs_pytorch_fp16_effective_active_fps_pct"] = None
    return row


def apply_cross_backend_deltas(rows):
    pytorch_fp32 = {}
    pytorch_fp16 = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["source_mode"], row["pipeline_mode"])
        if row["backend_name"] == "pytorch" and row["precision"] == "fp32":
            pytorch_fp32[key] = row
        if row["backend_name"] == "pytorch" and row["precision"] == "fp16":
            pytorch_fp16[key] = row

    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["source_mode"], row["pipeline_mode"])
        fp32_baseline = pytorch_fp32.get(key)
        fp16_baseline = pytorch_fp16.get(key)
        if fp32_baseline is not None:
            row["delta_vs_pytorch_fp32_inference_mean_ms_pct"] = safe_pct_change(
                fp32_baseline["inference_mean_ms"], row["inference_mean_ms"]
            )
            row["delta_vs_pytorch_fp32_active_loop_mean_ms_pct"] = safe_pct_change(
                fp32_baseline["active_loop_mean_ms"], row["active_loop_mean_ms"]
            )
            row["delta_vs_pytorch_fp32_effective_active_fps_pct"] = safe_pct_change(
                fp32_baseline["effective_active_fps"], row["effective_active_fps"]
            )
        if fp16_baseline is not None:
            row["delta_vs_pytorch_fp16_inference_mean_ms_pct"] = safe_pct_change(
                fp16_baseline["inference_mean_ms"], row["inference_mean_ms"]
            )
            row["delta_vs_pytorch_fp16_active_loop_mean_ms_pct"] = safe_pct_change(
                fp16_baseline["active_loop_mean_ms"], row["active_loop_mean_ms"]
            )
            row["delta_vs_pytorch_fp16_effective_active_fps_pct"] = safe_pct_change(
                fp16_baseline["effective_active_fps"], row["effective_active_fps"]
            )


def build_notes(args, rows, variants):
    lines = [
        "# Runtime Backend Comparison",
        "",
        "This package compares shared-runtime behavior across backend variants using the same video, pipeline modes, and source modes.",
        "",
        "Recorded comparison dimensions:",
        "- source mode",
        "- pipeline mode",
        "- backend name",
        "- precision and autocast",
        "- weights path and action vocabulary path",
        "- TensorRT engine path when present",
        "- active fraction, activation counts, and gate activity",
        "- inference, postprocess, render, and active-loop timing",
        "",
        "Interpretation rules:",
        "- negative latency deltas mean the candidate is faster than the referenced PyTorch baseline",
        "- positive effective-active-FPS deltas mean the candidate is better than the referenced PyTorch baseline",
        "- use the per-run config.json files in runs/ when exact contract details matter",
        "",
        "Variants included:",
    ]
    for backend_name, precision in variants:
        if backend_name == "tensorrt":
            lines.append(f"- {backend_name} {precision} using engine: {args.trt_engine_path}")
        else:
            lines.append(f"- {backend_name} {precision}")
    successful = [row for row in rows if row["status"] == "ok"]
    if successful:
        lines.extend(
            [
                "",
                "Top-level artifacts:",
                "- comparison_summary.csv",
                "- comparison_summary.json",
                "- runs/",
            ]
        )
    return lines


def main():
    args = parse_args()
    video_path, source_modes, modes, pytorch_precisions = validate_args(args)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    runs_dir = output_dir / "runs"
    ensure_dir(runs_dir)
    variants = build_variant_specs(pytorch_precisions, args.include_tensorrt)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    rows = []
    detailed_runs = []
    failures = []
    experiment_index = 0
    total_experiments = len(source_modes) * len(modes) * len(variants)
    for source_mode in source_modes:
        for pipeline_mode in modes:
            for backend_name, precision in variants:
                experiment_index += 1
                comparison_label = f"{backend_name}_{precision}"
                experiment_id = f"{experiment_index:02d}_{source_mode}_{pipeline_mode}_{comparison_label}"
                raw_config = build_raw_config(args, source_mode, pipeline_mode, backend_name, precision, video_path)
                run_dir = runs_dir / experiment_id
                print(f"[{experiment_index}/{total_experiments}] Running {experiment_id}")
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
                    rows.append(augment_summary_row(row, raw_config, comparison_label))
                    detailed_runs.append(
                        {
                            "experiment_id": experiment_id,
                            "comparison_label": comparison_label,
                            "source_mode": source_mode,
                            "pipeline_mode": pipeline_mode,
                            "backend_name": backend_name,
                            "precision": precision,
                            "raw_config": raw_config,
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
                            "comparison_label": comparison_label,
                            "source_mode": source_mode,
                            "pipeline_mode": pipeline_mode,
                            "backend_name": backend_name,
                            "precision": precision,
                            "autocast": precision == "fp16",
                            "weights_path": raw_config.get("weights_path"),
                            "actions_json": raw_config.get("actions_json"),
                            "trt_engine_path": raw_config.get("trt_engine_path"),
                            "video_path": str(video_path),
                            "max_frames": raw_config.get("max_frames"),
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
                            "delta_vs_pytorch_fp32_inference_mean_ms_pct": None,
                            "delta_vs_pytorch_fp32_active_loop_mean_ms_pct": None,
                            "delta_vs_pytorch_fp32_effective_active_fps_pct": None,
                            "delta_vs_pytorch_fp16_inference_mean_ms_pct": None,
                            "delta_vs_pytorch_fp16_active_loop_mean_ms_pct": None,
                            "delta_vs_pytorch_fp16_effective_active_fps_pct": None,
                            "run_dir": str(run_dir),
                            "status": "error",
                            "error": error_text,
                        }
                    )

    apply_cross_backend_deltas(rows)
    payload = {
        "git_commit": infer_git_commit(),
        "video_path": str(video_path),
        "invoked_command": invoked_command,
        "variants": [
            {
                "backend_name": backend_name,
                "precision": precision,
                "trt_engine_path": args.trt_engine_path if backend_name == "tensorrt" else None,
            }
            for backend_name, precision in variants
        ],
        "rows": rows,
        "detailed_runs": detailed_runs,
        "failures": failures,
    }
    write_csv(output_dir / "comparison_summary.csv", SUMMARY_FIELDNAMES, rows)
    write_json(output_dir / "comparison_summary.json", payload)
    write_run_summary(output_dir / "README.md", build_notes(args, rows, variants))

    print(f"Runtime backend comparison saved to: {output_dir}")
    print(f"CSV summary: {output_dir / 'comparison_summary.csv'}")
    print(f"JSON summary: {output_dir / 'comparison_summary.json'}")
    if failures:
        raise RuntimeError(f"Backend comparison completed with failures: {failures[0]['experiment_id']}")


if __name__ == "__main__":
    main()
