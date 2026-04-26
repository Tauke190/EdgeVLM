import argparse
import csv
import json
from pathlib import Path
import shlex
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, infer_git_commit, write_csv, write_json, write_run_summary


FINAL_TABLE_FIELDNAMES = [
    "section",
    "comparison_label",
    "source_mode",
    "pipeline_mode",
    "backend_name",
    "precision",
    "video_path",
    "frames_read",
    "output_ready_frames",
    "active_frames",
    "active_fraction_output_ready",
    "sia_active_fraction",
    "effective_input_fps",
    "effective_active_fps",
    "sia_active_fps",
    "inference_mean_ms",
    "postprocess_mean_ms",
    "render_mean_ms",
    "active_loop_mean_ms",
    "motion_event_count",
    "person_event_count",
    "sia_activation_count",
    "sia_stride_wait_frames",
    "motion_to_sia_latency_frames",
    "motion_to_sia_latency_mean_frames",
    "motion_to_sia_latency_median_frames",
    "sia_trigger_reason_counts",
    "delta_vs_pytorch_fp32_inference_mean_ms_pct",
    "delta_vs_pytorch_fp32_active_loop_mean_ms_pct",
    "delta_vs_pytorch_fp32_effective_active_fps_pct",
    "delta_vs_pytorch_fp16_inference_mean_ms_pct",
    "delta_vs_pytorch_fp16_active_loop_mean_ms_pct",
    "delta_vs_pytorch_fp16_effective_active_fps_pct",
    "run_dir",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble a final presentation package from existing backend-comparison and full-pipeline artifacts."
    )
    parser.add_argument(
        "--backend-compare-summary",
        default="results/runtime/backend_compare_hit_smoke/comparison_summary.json",
        help="Path to backend comparison JSON summary.",
    )
    parser.add_argument(
        "--full-pipeline-summary",
        default="results/full_pipeline/surveillance_long/benchmark_summary.json",
        help="Path to full-pipeline benchmark JSON summary.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_package",
        help="Output directory for final package artifacts.",
    )
    parser.add_argument(
        "--title",
        default="Final Package",
        help="Title written into README and talking points.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def maybe_int(value):
    if value in (None, "", "None"):
        return None
    return int(value)


def maybe_json_like(value):
    if value in (None, "", "None"):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        normalized = value.replace("'", '"')
        return json.loads(normalized)
    except Exception:
        return value


def summarize_latencies(latency_values):
    if not latency_values:
        return None, None
    return round(sum(latency_values) / len(latency_values), 3), round(statistics.median(latency_values), 3)


def derive_sia_active_fraction(active_frames, output_ready_frames):
    if active_frames in (None, "") or output_ready_frames in (None, "", 0):
        return None
    active_frames = int(active_frames)
    output_ready_frames = int(output_ready_frames)
    if output_ready_frames <= 0:
        return None
    return round(active_frames / output_ready_frames, 4)


def derive_sia_active_fps(active_loop_mean_ms):
    if active_loop_mean_ms in (None, "", 0):
        return None
    active_loop_mean_ms = float(active_loop_mean_ms)
    if active_loop_mean_ms <= 0.0:
        return None
    return round(1000.0 / active_loop_mean_ms, 3)


def normalize_backend_row(row):
    latency_values = []
    active_frames = maybe_int(row.get("active_frames"))
    output_ready_frames = maybe_int(row.get("output_ready_frames"))
    active_loop_mean_ms = maybe_float(row.get("active_loop_mean_ms"))
    return {
        "section": "backend_compare",
        "comparison_label": row.get("comparison_label"),
        "source_mode": row.get("source_mode"),
        "pipeline_mode": row.get("pipeline_mode"),
        "backend_name": row.get("backend_name"),
        "precision": row.get("precision"),
        "video_path": row.get("video_path"),
        "frames_read": maybe_int(row.get("frames_read")),
        "output_ready_frames": output_ready_frames,
        "active_frames": active_frames,
        "active_fraction_output_ready": maybe_float(row.get("active_fraction_output_ready")),
        "sia_active_fraction": derive_sia_active_fraction(active_frames, output_ready_frames),
        "effective_input_fps": maybe_float(row.get("effective_input_fps")),
        "effective_active_fps": maybe_float(row.get("effective_active_fps")),
        "sia_active_fps": derive_sia_active_fps(active_loop_mean_ms),
        "inference_mean_ms": maybe_float(row.get("inference_mean_ms")),
        "postprocess_mean_ms": maybe_float(row.get("postprocess_mean_ms")),
        "render_mean_ms": maybe_float(row.get("render_mean_ms")),
        "active_loop_mean_ms": active_loop_mean_ms,
        "motion_event_count": maybe_int(row.get("motion_event_count")),
        "person_event_count": maybe_int(row.get("person_event_count")),
        "sia_activation_count": maybe_int(row.get("sia_activation_count")),
        "sia_stride_wait_frames": maybe_int(row.get("sia_stride_wait_frames")),
        "motion_to_sia_latency_frames": latency_values,
        "motion_to_sia_latency_mean_frames": None,
        "motion_to_sia_latency_median_frames": None,
        "sia_trigger_reason_counts": maybe_json_like(row.get("sia_trigger_reason_counts")),
        "delta_vs_pytorch_fp32_inference_mean_ms_pct": maybe_float(row.get("delta_vs_pytorch_fp32_inference_mean_ms_pct")),
        "delta_vs_pytorch_fp32_active_loop_mean_ms_pct": maybe_float(row.get("delta_vs_pytorch_fp32_active_loop_mean_ms_pct")),
        "delta_vs_pytorch_fp32_effective_active_fps_pct": maybe_float(row.get("delta_vs_pytorch_fp32_effective_active_fps_pct")),
        "delta_vs_pytorch_fp16_inference_mean_ms_pct": maybe_float(row.get("delta_vs_pytorch_fp16_inference_mean_ms_pct")),
        "delta_vs_pytorch_fp16_active_loop_mean_ms_pct": maybe_float(row.get("delta_vs_pytorch_fp16_active_loop_mean_ms_pct")),
        "delta_vs_pytorch_fp16_effective_active_fps_pct": maybe_float(row.get("delta_vs_pytorch_fp16_effective_active_fps_pct")),
        "run_dir": row.get("run_dir"),
        "notes": "",
    }


def normalize_full_pipeline_row(summary_payload):
    videos = summary_payload.get("videos", [])
    if not videos:
        raise RuntimeError("Full pipeline summary does not contain any videos.")
    video = videos[0]
    metrics_path = Path(video["run_dir"]) / "metrics.json"
    metrics = load_json(metrics_path)
    latency_values = metrics.get("motion_to_sia_latency_frames") or []
    latency_mean, latency_median = summarize_latencies(latency_values)
    output_ready = metrics.get("output_ready_frames") or 0
    active = metrics.get("active_frames") or 0
    active_fraction = round(active / output_ready, 4) if output_ready > 0 else None
    active_loop_mean_ms = metrics.get("timings", {}).get("active_loop", {}).get("mean_ms")
    return {
        "section": "full_pipeline",
        "comparison_label": "staged_pipeline_long_clip",
        "source_mode": "offline",
        "pipeline_mode": video.get("pipeline_mode"),
        "backend_name": metrics.get("backend_name", "pytorch"),
        "precision": metrics.get("precision"),
        "video_path": video.get("video_path"),
        "frames_read": metrics.get("frames_read"),
        "output_ready_frames": metrics.get("output_ready_frames"),
        "active_frames": metrics.get("active_frames"),
        "active_fraction_output_ready": active_fraction,
        "sia_active_fraction": active_fraction,
        "effective_input_fps": metrics.get("effective_input_fps", metrics.get("effective_fps")),
        "effective_active_fps": metrics.get("effective_active_fps", metrics.get("effective_fps")),
        "sia_active_fps": derive_sia_active_fps(active_loop_mean_ms),
        "inference_mean_ms": metrics.get("timings", {}).get("inference", {}).get("mean_ms"),
        "postprocess_mean_ms": metrics.get("timings", {}).get("postprocess", {}).get("mean_ms"),
        "render_mean_ms": metrics.get("timings", {}).get("render", {}).get("mean_ms"),
        "active_loop_mean_ms": active_loop_mean_ms,
        "motion_event_count": metrics.get("motion_event_count"),
        "person_event_count": metrics.get("person_event_count"),
        "sia_activation_count": metrics.get("sia_activation_count"),
        "sia_stride_wait_frames": metrics.get("sia_stride_wait_frames"),
        "motion_to_sia_latency_frames": latency_values,
        "motion_to_sia_latency_mean_frames": latency_mean,
        "motion_to_sia_latency_median_frames": latency_median,
        "sia_trigger_reason_counts": metrics.get("sia_trigger_reason_counts"),
        "delta_vs_pytorch_fp32_inference_mean_ms_pct": None,
        "delta_vs_pytorch_fp32_active_loop_mean_ms_pct": None,
        "delta_vs_pytorch_fp32_effective_active_fps_pct": None,
        "delta_vs_pytorch_fp16_inference_mean_ms_pct": None,
        "delta_vs_pytorch_fp16_active_loop_mean_ms_pct": None,
        "delta_vs_pytorch_fp16_effective_active_fps_pct": None,
        "run_dir": video.get("run_dir"),
        "notes": "Long surveillance-style clip showing scheduler behavior and expensive-tier duty cycle.",
    }


def build_rows(backend_compare_payload, full_pipeline_payload):
    backend_csv = Path(backend_compare_payload["_csv_path"])
    backend_rows_raw = load_csv_rows(backend_csv)
    backend_rows = [normalize_backend_row(row) for row in backend_rows_raw if row.get("status") == "ok"]
    full_pipeline_row = normalize_full_pipeline_row(full_pipeline_payload)
    return backend_rows + [full_pipeline_row]


def select_backend_row(rows, source_mode, pipeline_mode, backend_name, precision):
    for row in rows:
        if (
            row["section"] == "backend_compare"
            and row["source_mode"] == source_mode
            and row["pipeline_mode"] == pipeline_mode
            and row["backend_name"] == backend_name
            and row["precision"] == precision
        ):
            return row
    return None


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_talking_points(rows):
    live_always_fp32 = select_backend_row(rows, "live_replay", "always_on", "pytorch", "fp32")
    live_always_fp16 = select_backend_row(rows, "live_replay", "always_on", "pytorch", "fp16")
    live_always_trt = select_backend_row(rows, "live_replay", "always_on", "tensorrt", "fp16")
    live_always_trt_int8 = select_backend_row(rows, "live_replay", "always_on", "tensorrt", "int8")
    live_staged_fp32 = select_backend_row(rows, "live_replay", "motion_person_sia", "pytorch", "fp32")
    live_staged_fp16 = select_backend_row(rows, "live_replay", "motion_person_sia", "pytorch", "fp16")
    live_staged_trt = select_backend_row(rows, "live_replay", "motion_person_sia", "tensorrt", "fp16")
    live_staged_trt_int8 = select_backend_row(rows, "live_replay", "motion_person_sia", "tensorrt", "int8")
    long_pipeline = next(row for row in rows if row["section"] == "full_pipeline")

    lines = [
        "# Final Talking Points",
        "",
        "## What The System Does",
        "- The final runtime is a shared staged pipeline with four comparable operating modes: always-on, motion-only, person-only, and motion+person+SiA.",
        "- Motion detection acts as the cheap wake-up signal, person detection confirms semantic relevance, and SiA runs only when the scheduler allows the expensive tier to activate.",
        "- The same shared runtime structure now supports both repeatable offline evaluation and live-like replay validation.",
        "",
        "## What Changed From The Original Demo",
        "- The original repo behavior was a mostly always-on PyTorch plus OpenCV path.",
        "- The current system adds a modular runtime, explicit scheduler states, event logs, a person gate, and a real TensorRT backend path for the SiA vision encoder.",
        "",
        "## Backend Findings",
    ]

    if live_always_fp32 and live_always_fp16 and live_always_trt:
        lines.extend(
            [
                f"- In live-like always-on runs, PyTorch fp32 sustained {fmt(live_always_fp32['sia_active_fps'])} SiA-active FPS with SiA active on {fmt(live_always_fp32['sia_active_fraction'], 4)} of output-ready frames.",
                f"- PyTorch fp16 improved that to {fmt(live_always_fp16['sia_active_fps'])} SiA-active FPS, with SiA still active on {fmt(live_always_fp16['sia_active_fraction'], 4)} of output-ready frames.",
                f"- TensorRT fp16 improved it further to {fmt(live_always_trt['sia_active_fps'])} SiA-active FPS, with SiA still active on {fmt(live_always_trt['sia_active_fraction'], 4)} of output-ready frames.",
                f"- Relative to PyTorch fp16 in that same live-like always-on condition, TensorRT fp16 changed inference latency by {fmt(live_always_trt['delta_vs_pytorch_fp16_inference_mean_ms_pct'], 1)}% while changing SiA-active FPS from {fmt(live_always_fp16['sia_active_fps'])} to {fmt(live_always_trt['sia_active_fps'])}.",
            ]
        )
        if live_always_trt_int8:
            lines.append(
                f"- TensorRT int8 reached {fmt(live_always_trt_int8['sia_active_fps'])} SiA-active FPS in that same live-like always-on condition, showing whether INT8 added anything meaningful beyond TensorRT fp16."
            )

    if live_staged_fp32 and live_staged_fp16 and live_staged_trt:
        lines.extend(
            [
                f"- In live-like staged runs, PyTorch fp32 sustained {fmt(live_staged_fp32['sia_active_fps'])} SiA-active FPS with SiA active on {fmt(live_staged_fp32['sia_active_fraction'], 4)} of output-ready frames.",
                f"- PyTorch fp16 improved that to {fmt(live_staged_fp16['sia_active_fps'])} SiA-active FPS with SiA active on {fmt(live_staged_fp16['sia_active_fraction'], 4)} of output-ready frames.",
                f"- TensorRT fp16 improved staged inference latency further to {fmt(live_staged_trt['inference_mean_ms'])} ms and raised SiA-active FPS to {fmt(live_staged_trt['sia_active_fps'])}, while keeping SiA active on only {fmt(live_staged_trt['sia_active_fraction'], 4)} of output-ready frames.",
                "- That means staged mode should be read as two metrics: how fast SiA runs when active, and how rarely the scheduler needs SiA at all.",
            ]
        )
        if live_staged_trt_int8:
            lines.append(
                f"- TensorRT int8 in live-like staged mode reached {fmt(live_staged_trt_int8['sia_active_fps'])} SiA-active FPS with the same SiA-active fraction of {fmt(live_staged_trt_int8['sia_active_fraction'], 4)}, which makes the INT8 marginal gain easy to compare directly against TensorRT fp16."
            )

    lines.extend(
        [
            "",
            "## Full Staged-Pipeline Findings",
            f"- On the longer surveillance-style clip, the full motion+person+SiA pipeline processed {long_pipeline['frames_read']} frames and produced {long_pipeline['output_ready_frames']} output-ready frames.",
            f"- The expensive SiA tier was active on only {long_pipeline['active_frames']} frames, which is an SiA-active fraction of {fmt(long_pipeline['sia_active_fraction'], 4)}.",
            f"- That long run recorded {long_pipeline['motion_event_count']} motion event, {long_pipeline['person_event_count']} person events, and {long_pipeline['sia_activation_count']} expensive-tier activations.",
            f"- The trigger reasons were {long_pipeline['sia_trigger_reason_counts']}, showing that most reactivations came from minimum-new-frame logic with a smaller number of person-edge wakeups.",
        ]
    )

    if long_pipeline["motion_to_sia_latency_mean_frames"] is not None:
        lines.append(
            f"- Motion-to-SiA latency on that run averaged {fmt(long_pipeline['motion_to_sia_latency_mean_frames'])} frames with median {fmt(long_pipeline['motion_to_sia_latency_median_frames'])} frames."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- The project result is not just that the model can be sped up in isolation. The result is that the staged runtime, scheduler, and backend choices can now be compared on one shared measurement surface.",
            "- TensorRT fp16 is the main proven acceleration branch. It improves SiA-active FPS materially, but the live-like staged path still shows remaining systems overhead beyond raw engine speed.",
            "- DeepStream is not required to defend the current contribution. The main contribution is the measured staged Jetson runtime with explicit tradeoffs, not the name of an SDK.",
            "",
            "## Future Work",
            "- If the next goal is higher active SiA throughput in live-like conditions, the next work should target shared-runtime overhead and possibly the model contract itself, not just another backend switch.",
            "- INT8 can still be revisited as a bounded follow-up branch, but current evidence says it is a smaller gain than TensorRT fp16 and not the main story.",
        ]
    )
    return lines


def build_readme(args, backend_compare_payload, full_pipeline_payload):
    return [
        f"# {args.title}",
        "",
        "This package assembles the final presentation-facing artifacts from existing benchmark outputs rather than rerunning the runtime.",
        "",
        f"Backend comparison source: {args.backend_compare_summary}",
        f"Full pipeline source: {args.full_pipeline_summary}",
        f"Git commit at packaging time: {infer_git_commit()}",
        "",
        "Artifacts:",
        "- final_metrics_table.csv",
        "- final_metrics_table.json",
        "- final_talking_points.md",
        "- figure_data/",
        "- demo_artifacts/",
        "",
        "Notes:",
        "- The backend comparison package records PyTorch fp32, PyTorch fp16, and TensorRT fp16 results across offline and live-like conditions.",
        "- The full-pipeline package records the longer surveillance-style staged run, including event logs and motion-to-SiA latency samples.",
    ]


def build_demo_artifact_manifest(backend_compare_payload, full_pipeline_payload):
    full_pipeline_video = full_pipeline_payload.get("videos", [{}])[0].get("run_dir", "")
    full_pipeline_video_path = f"{full_pipeline_video}/runtime_offline.mp4" if full_pipeline_video else "n/a"
    lines = [
        "# Demo Artifacts",
        "",
        "Reference artifact paths for presentation assembly:",
        f"- Full staged-pipeline long-run video: {full_pipeline_video_path}",
        f"- Backend comparison package: {backend_compare_payload['comparison_summary_json_path']}",
        f"- Full pipeline summary package: {full_pipeline_payload['benchmark_summary_json_path']}",
        "",
        "Suggested presentation assets:",
        "- one long staged-pipeline output clip",
        "- one backend comparison table from the final package",
        "- one event timeline derived from the long full-pipeline event log",
    ]
    return lines


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    backend_compare_payload = load_json(args.backend_compare_summary)
    backend_compare_payload["_csv_path"] = str(Path(args.backend_compare_summary).with_name("comparison_summary.csv"))
    backend_compare_payload["comparison_summary_json_path"] = args.backend_compare_summary
    full_pipeline_payload = load_json(args.full_pipeline_summary)
    full_pipeline_payload["benchmark_summary_json_path"] = args.full_pipeline_summary

    rows = build_rows(backend_compare_payload, full_pipeline_payload)
    demo_artifacts_dir = output_dir / "demo_artifacts"
    ensure_dir(demo_artifacts_dir)

    payload = {
        "title": args.title,
        "git_commit": infer_git_commit(),
        "invoked_command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "backend_compare_summary": args.backend_compare_summary,
        "full_pipeline_summary": args.full_pipeline_summary,
        "rows": rows,
    }

    write_csv(output_dir / "final_metrics_table.csv", FINAL_TABLE_FIELDNAMES, rows)
    write_json(output_dir / "final_metrics_table.json", payload)
    write_run_summary(output_dir / "final_talking_points.md", build_talking_points(rows))
    write_run_summary(output_dir / "README.md", build_readme(args, backend_compare_payload, full_pipeline_payload))
    write_run_summary(demo_artifacts_dir / "README.md", build_demo_artifact_manifest(backend_compare_payload, full_pipeline_payload))

    print(f"Final package saved to: {output_dir}")
    print(f"Metrics table: {output_dir / 'final_metrics_table.csv'}")
    print(f"Talking points: {output_dir / 'final_talking_points.md'}")


if __name__ == "__main__":
    main()
