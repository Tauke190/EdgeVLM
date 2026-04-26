import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, write_csv, write_json, write_run_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export chart-friendly CSVs and simple manifests from final package artifacts."
    )
    parser.add_argument(
        "--final-metrics-json",
        default="results/final_package/final_metrics_table.json",
        help="Path to final package JSON payload.",
    )
    parser.add_argument(
        "--full-pipeline-event-log",
        default="results/full_pipeline/surveillance_long/001_SurvellienceFootage/event_log.csv",
        help="Path to the long full-pipeline event log.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_package/figure_data",
        help="Directory for exported figure-friendly data.",
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


def load_run_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    return load_json(metrics_path)


def timing_mean(metrics, timing_name):
    if not metrics:
        return None
    return maybe_float(metrics.get("timings", {}).get(timing_name, {}).get("mean_ms"))


def export_backend_latency(rows, output_dir):
    fieldnames = [
        "source_mode",
        "pipeline_mode",
        "backend_name",
        "precision",
        "inference_mean_ms",
        "active_loop_mean_ms",
        "sia_active_fps",
        "sia_active_fraction",
        "effective_active_fps",
    ]
    selected = []
    for row in rows:
        if row.get("section") != "backend_compare":
            continue
        selected.append(
            {
                "source_mode": row["source_mode"],
                "pipeline_mode": row["pipeline_mode"],
                "backend_name": row["backend_name"],
                "precision": row["precision"],
                "inference_mean_ms": row["inference_mean_ms"],
                "active_loop_mean_ms": row["active_loop_mean_ms"],
                "sia_active_fps": row.get("sia_active_fps"),
                "sia_active_fraction": row.get("sia_active_fraction"),
                "effective_active_fps": row["effective_active_fps"],
            }
        )
    write_csv(output_dir / "backend_latency_by_variant.csv", fieldnames, selected)


def export_stage_breakdown(rows, output_dir):
    fieldnames = [
        "comparison_label",
        "source_mode",
        "pipeline_mode",
        "backend_name",
        "precision",
        "capture_mean_ms",
        "preprocess_mean_ms",
        "inference_mean_ms",
        "postprocess_mean_ms",
        "label_decode_mean_ms",
        "render_mean_ms",
        "active_loop_mean_ms",
        "sia_active_fps",
        "sia_active_fraction",
        "effective_active_fps",
        "run_dir",
    ]
    selected = []
    for row in rows:
        if row.get("section") != "backend_compare":
            continue
        metrics = load_run_metrics(row.get("run_dir"))
        if metrics is None:
            continue
        selected.append(
            {
                "comparison_label": row["comparison_label"],
                "source_mode": row["source_mode"],
                "pipeline_mode": row["pipeline_mode"],
                "backend_name": row["backend_name"],
                "precision": row["precision"],
                "capture_mean_ms": timing_mean(metrics, "capture"),
                "preprocess_mean_ms": timing_mean(metrics, "preprocess"),
                "inference_mean_ms": timing_mean(metrics, "inference"),
                "postprocess_mean_ms": timing_mean(metrics, "postprocess"),
                "label_decode_mean_ms": timing_mean(metrics, "label_decode"),
                "render_mean_ms": timing_mean(metrics, "render"),
                "active_loop_mean_ms": timing_mean(metrics, "active_loop"),
                "sia_active_fps": row.get("sia_active_fps"),
                "sia_active_fraction": row.get("sia_active_fraction"),
                "effective_active_fps": maybe_float(row["effective_active_fps"]),
                "run_dir": row["run_dir"],
            }
        )
    write_csv(output_dir / "stage_breakdown_by_variant.csv", fieldnames, selected)


def export_pipeline_activity(rows, output_dir):
    fieldnames = [
        "comparison_label",
        "video_path",
        "output_ready_frames",
        "active_frames",
        "active_fraction_output_ready",
        "sia_active_fraction",
        "sia_active_fps",
        "motion_event_count",
        "person_event_count",
        "sia_activation_count",
        "sia_stride_wait_frames",
        "motion_to_sia_latency_mean_frames",
        "motion_to_sia_latency_median_frames",
    ]
    selected = []
    for row in rows:
        if row.get("section") != "full_pipeline":
            continue
        selected.append({key: row.get(key) for key in fieldnames})
    write_csv(output_dir / "pipeline_activity_summary.csv", fieldnames, selected)


def export_event_timeline(event_rows, output_dir):
    fieldnames = ["frame_index", "event", "scheduler_state", "prev_scheduler_state", "motion_active", "person_active", "sia_active", "person_detector_ran", "sia_trigger_reason"]
    selected = [{key: row.get(key) for key in fieldnames} for row in event_rows]
    write_csv(output_dir / "full_pipeline_event_timeline.csv", fieldnames, selected)


def build_manifest(output_dir):
    return [
        "# Figure Data",
        "",
        "Generated chart-friendly exports:",
        "- backend_latency_by_variant.csv",
        "- stage_breakdown_by_variant.csv",
        "- pipeline_activity_summary.csv",
        "- full_pipeline_event_timeline.csv",
        "",
        "Suggested figure uses:",
        "- backend_latency_by_variant.csv for grouped bar charts of inference latency, SiA-active FPS, and SiA-active fraction",
        "- stage_breakdown_by_variant.csv for the midterm-style runtime-stage breakdown tables and tradeoff plots",
        "- pipeline_activity_summary.csv for staged-pipeline duty-cycle and SiA-active fraction tables",
        "- full_pipeline_event_timeline.csv for timeline diagrams showing motion, person, and SiA transitions",
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    payload = load_json(args.final_metrics_json)
    rows = payload.get("rows", [])
    event_rows = load_csv_rows(args.full_pipeline_event_log)

    export_backend_latency(rows, output_dir)
    export_stage_breakdown(rows, output_dir)
    export_pipeline_activity(rows, output_dir)
    export_event_timeline(event_rows, output_dir)
    write_json(
        output_dir / "figure_manifest.json",
        {
            "final_metrics_json": args.final_metrics_json,
            "full_pipeline_event_log": args.full_pipeline_event_log,
            "exported_files": [
                "backend_latency_by_variant.csv",
                "stage_breakdown_by_variant.csv",
                "pipeline_activity_summary.csv",
                "full_pipeline_event_timeline.csv",
            ],
        },
    )
    write_run_summary(output_dir / "README.md", build_manifest(output_dir))
    print(f"Figure exports saved to: {output_dir}")


if __name__ == "__main__":
    main()
