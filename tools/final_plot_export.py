import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import ensure_dir, write_json, write_run_summary


BACKEND_ORDER = ["pytorch_fp32", "pytorch_fp16", "tensorrt_fp16", "tensorrt_int8"]
BACKEND_LABELS = {
    "pytorch_fp32": "PyTorch fp32",
    "pytorch_fp16": "PyTorch fp16",
    "tensorrt_fp16": "TensorRT fp16",
    "tensorrt_int8": "TensorRT int8",
}
CONDITION_ORDER = [
    ("offline", "always_on"),
    ("offline", "motion_person_sia"),
    ("live_replay", "always_on"),
    ("live_replay", "motion_person_sia"),
]
CONDITION_LABELS = {
    ("offline", "always_on"): "Offline Always-On",
    ("offline", "motion_person_sia"): "Offline Staged",
    ("live_replay", "always_on"): "Live-Like Always-On",
    ("live_replay", "motion_person_sia"): "Live-Like Staged",
}
COLORS = {
    "pytorch_fp32": "#6b7280",
    "pytorch_fp16": "#2563eb",
    "tensorrt_fp16": "#16a34a",
    "tensorrt_int8": "#ea580c",
}
STAGE_ORDER = [
    ("capture_mean_ms", "Capture"),
    ("preprocess_mean_ms", "Preprocess"),
    ("inference_mean_ms", "Inference"),
    ("postprocess_mean_ms", "Postprocess"),
    ("label_decode_mean_ms", "Label Decode"),
    ("active_loop_mean_ms", "Active Loop"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready plots from results/final_package artifacts."
    )
    parser.add_argument(
        "--final-metrics-json",
        default="results/final_package/final_metrics_table.json",
        help="Path to final package JSON payload.",
    )
    parser.add_argument(
        "--backend-latency-csv",
        default="results/final_package/figure_data/backend_latency_by_variant.csv",
        help="Path to backend latency export CSV.",
    )
    parser.add_argument(
        "--stage-breakdown-csv",
        default="results/final_package/figure_data/stage_breakdown_by_variant.csv",
        help="Path to runtime stage breakdown export CSV.",
    )
    parser.add_argument(
        "--pipeline-activity-csv",
        default="results/final_package/figure_data/pipeline_activity_summary.csv",
        help="Path to pipeline activity export CSV.",
    )
    parser.add_argument(
        "--event-timeline-csv",
        default="results/final_package/figure_data/full_pipeline_event_timeline.csv",
        help="Path to full-pipeline event timeline CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_package/plots",
        help="Directory for generated plots.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def backend_key(backend_name, precision):
    return f"{backend_name}_{precision}"


def condition_subset(df, condition):
    source_mode, pipeline_mode = condition
    subset = df[(df["source_mode"] == source_mode) & (df["pipeline_mode"] == pipeline_mode)].copy()
    subset["variant_key"] = subset.apply(lambda row: backend_key(row["backend_name"], row["precision"]), axis=1)
    return subset.set_index("variant_key").reindex(BACKEND_ORDER).reset_index()


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_backend_latency(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITION_ORDER):
        source_mode, pipeline_mode = condition
        subset = df[(df["source_mode"] == source_mode) & (df["pipeline_mode"] == pipeline_mode)].copy()
        subset["variant_key"] = subset.apply(lambda row: backend_key(row["backend_name"], row["precision"]), axis=1)
        subset = subset.set_index("variant_key").reindex(BACKEND_ORDER).reset_index()

        x = range(len(subset))
        values = subset["inference_mean_ms"].astype(float).tolist()
        labels = [BACKEND_LABELS[key] for key in subset["variant_key"]]
        colors = [COLORS[key] for key in subset["variant_key"]]
        ax.bar(x, values, color=colors, width=0.65)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Inference Mean (ms)")
        ax.set_title(CONDITION_LABELS[condition])
        style_axes(ax)

        for xpos, value in zip(x, values):
            ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save_figure(fig, output_dir / "backend_inference_latency.png")


def plot_backend_active_loop(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITION_ORDER):
        source_mode, pipeline_mode = condition
        subset = df[(df["source_mode"] == source_mode) & (df["pipeline_mode"] == pipeline_mode)].copy()
        subset["variant_key"] = subset.apply(lambda row: backend_key(row["backend_name"], row["precision"]), axis=1)
        subset = subset.set_index("variant_key").reindex(BACKEND_ORDER).reset_index()

        x = range(len(subset))
        values = subset["active_loop_mean_ms"].astype(float).tolist()
        labels = [BACKEND_LABELS[key] for key in subset["variant_key"]]
        colors = [COLORS[key] for key in subset["variant_key"]]
        ax.bar(x, values, color=colors, width=0.65)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Active Loop Mean (ms)")
        ax.set_title(CONDITION_LABELS[condition])
        style_axes(ax)

        for xpos, value in zip(x, values):
            ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save_figure(fig, output_dir / "backend_active_loop_latency.png")


def plot_backend_fps(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITION_ORDER):
        source_mode, pipeline_mode = condition
        subset = df[(df["source_mode"] == source_mode) & (df["pipeline_mode"] == pipeline_mode)].copy()
        subset["variant_key"] = subset.apply(lambda row: backend_key(row["backend_name"], row["precision"]), axis=1)
        subset = subset.set_index("variant_key").reindex(BACKEND_ORDER).reset_index()

        x = range(len(subset))
        values = subset["sia_active_fps"].astype(float).tolist()
        labels = [BACKEND_LABELS[key] for key in subset["variant_key"]]
        colors = [COLORS[key] for key in subset["variant_key"]]
        ax.bar(x, values, color=colors, width=0.65)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("SiA Active FPS")
        ax.set_title(CONDITION_LABELS[condition])
        style_axes(ax)

        for xpos, value in zip(x, values):
            ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    save_figure(fig, output_dir / "backend_sia_active_fps.png")


def plot_backend_active_fraction(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITION_ORDER):
        subset = condition_subset(df, condition)

        x = range(len(subset))
        values = subset["sia_active_fraction"].astype(float).tolist()
        labels = [BACKEND_LABELS[key] for key in subset["variant_key"]]
        colors = [COLORS[key] for key in subset["variant_key"]]
        ax.bar(x, values, color=colors, width=0.65)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("SiA Active Fraction")
        ax.set_title(CONDITION_LABELS[condition])
        style_axes(ax)
        ax.set_ylim(0.0, 1.05)

        for xpos, value in zip(x, values):
            ax.text(xpos, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    save_figure(fig, output_dir / "backend_sia_active_fraction.png")


def plot_stage_breakdown(stage_df, condition, output_dir, filename):
    subset = condition_subset(stage_df, condition)
    fig, ax = plt.subplots(figsize=(14, 6.5))

    x = list(range(len(STAGE_ORDER)))
    variant_count = len(subset)
    width = 0.18 if variant_count >= 4 else 0.24
    offsets = [(index - (variant_count - 1) / 2.0) * width for index in range(variant_count)]

    for offset, (_, row) in zip(offsets, subset.iterrows()):
        key = row["variant_key"]
        values = [float(row[column]) for column, _ in STAGE_ORDER]
        ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            color=COLORS[key],
            label=BACKEND_LABELS[key],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in STAGE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title(f"Runtime Stage Breakdown: {CONDITION_LABELS[condition]}")
    style_axes(ax)
    ax.legend(frameon=False, ncols=3, loc="upper left")

    save_figure(fig, output_dir / filename)


def plot_tensorrt_tradeoff(stage_df, output_dir):
    conditions = [("live_replay", "always_on"), ("live_replay", "motion_person_sia")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, condition in zip(axes, conditions):
        subset = condition_subset(stage_df, condition).set_index("variant_key")
        fp16 = subset.loc["pytorch_fp16"]
        candidate_keys = [key for key in ("tensorrt_fp16", "tensorrt_int8") if key in subset.index]
        labels = [label for _, label in STAGE_ORDER]
        x = list(range(len(labels)))
        width = 0.35 if len(candidate_keys) > 1 else 0.5
        offsets = [(index - (len(candidate_keys) - 1) / 2.0) * width for index in range(len(candidate_keys))]
        for offset, candidate_key in zip(offsets, candidate_keys):
            candidate_row = subset.loc[candidate_key]
            deltas = []
            colors = []
            for column, _ in STAGE_ORDER:
                baseline = float(fp16[column])
                candidate = float(candidate_row[column])
                delta_pct = ((candidate - baseline) / baseline * 100.0) if baseline else 0.0
                deltas.append(delta_pct)
                colors.append(COLORS[candidate_key] if delta_pct < 0 else "#dc2626")
            ax.bar([pos + offset for pos in x], deltas, color=colors, width=width, label=BACKEND_LABELS[candidate_key])
            for xpos, value in zip(x, deltas):
                valign = "bottom" if value >= 0 else "top"
                ax.text(xpos + offset, value, f"{value:.1f}%", ha="center", va=valign, fontsize=8)

        ax.axhline(0.0, color="#111827", linewidth=1)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_ylabel("TensorRT vs PyTorch fp16 Delta (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        style_axes(ax)
        ax.legend(frameon=False, loc="upper left")

    save_figure(fig, output_dir / "tensorrt_stage_tradeoff_vs_pytorch_fp16.png")


def plot_pipeline_duty_cycle(activity_df, output_dir):
    row = activity_df.iloc[0]
    categories = ["Output-Ready", "SiA Active", "Stride-Wait"]
    values = [
        float(row["output_ready_frames"]),
        float(row["active_frames"]),
        float(row["sia_stride_wait_frames"]),
    ]
    colors = ["#94a3b8", "#dc2626", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, width=0.65)
    ax.set_ylabel("Frames")
    ax.set_title("Long-Run Staged Pipeline Duty Cycle")
    style_axes(ax)

    subtitle = (
        f"Active fraction={float(row['active_fraction_output_ready']):.4f}, "
        f"motion events={int(row['motion_event_count'])}, "
        f"person events={int(row['person_event_count'])}, "
        f"SiA activations={int(row['sia_activation_count'])}"
    )
    ax.text(0.5, 1.04, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom", fontsize=10)

    save_figure(fig, output_dir / "pipeline_duty_cycle.png")


def plot_event_timeline(event_df, output_dir):
    event_df = event_df.copy()
    event_df["frame_index"] = event_df["frame_index"].astype(int)
    event_df["motion_active"] = event_df["motion_active"].map(lambda v: str(v).lower() == "true").astype(int)
    event_df["person_active"] = event_df["person_active"].map(lambda v: str(v).lower() == "true").astype(int)
    event_df["sia_active"] = event_df["sia_active"].map(lambda v: str(v).lower() == "true").astype(int)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.step(event_df["frame_index"], event_df["motion_active"] + 2, where="post", label="Motion Active", linewidth=2, color="#f59e0b")
    ax.step(event_df["frame_index"], event_df["person_active"] + 1, where="post", label="Person Active", linewidth=2, color="#2563eb")
    ax.step(event_df["frame_index"], event_df["sia_active"], where="post", label="SiA Active", linewidth=2, color="#dc2626")

    sia_rows = event_df[event_df["event"] == "sia_active"]
    if not sia_rows.empty:
        ax.scatter(sia_rows["frame_index"], [0.05] * len(sia_rows), marker="o", s=35, color="#111827", label="SiA Trigger")

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["SiA", "Person", "Motion", ""])
    ax.set_xlabel("Frame Index")
    ax.set_title("Full Pipeline Event Timeline")
    style_axes(ax)
    ax.legend(loc="upper left", ncols=4, frameon=False)

    save_figure(fig, output_dir / "full_pipeline_event_timeline.png")


def plot_motion_to_sia_latency(final_metrics_payload, output_dir):
    latency_values = []
    for row in final_metrics_payload.get("rows", []):
        if row.get("section") == "full_pipeline":
            latency_values = row.get("motion_to_sia_latency_frames") or []
            break

    if not latency_values:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latency_values, bins=min(10, max(5, len(set(latency_values)))), color="#7c3aed", edgecolor="white")
    ax.set_xlabel("Motion-to-SiA Latency (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Motion-to-SiA Latency Distribution")
    style_axes(ax)
    ax.text(
        0.98,
        0.95,
        f"mean={sum(latency_values)/len(latency_values):.1f}\nmedian={pd.Series(latency_values).median():.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    save_figure(fig, output_dir / "motion_to_sia_latency_histogram.png")
    return "motion_to_sia_latency_histogram.png"


def write_stage_breakdown_tables(stage_df, output_dir):
    conditions = [("live_replay", "always_on"), ("live_replay", "motion_person_sia")]
    variant_keys = [key for key in BACKEND_ORDER if key in set(stage_df.apply(lambda row: backend_key(row["backend_name"], row["precision"]), axis=1))]
    lines = [
        "# Runtime Stage Breakdown Tables",
        "",
        "These tables mirror the midterm stage-by-stage latency comparison, but now include TensorRT fp16 and TensorRT int8.",
        "",
    ]

    for condition in conditions:
        subset = condition_subset(stage_df, condition).set_index("variant_key")
        lines.append(f"## {CONDITION_LABELS[condition]}")
        lines.append("")
        header = "| Stage | " + " | ".join(BACKEND_LABELS[key] for key in variant_keys) + " |"
        divider = "| --- | " + " | ".join("---:" for _ in variant_keys) + " |"
        lines.append(header)
        lines.append(divider)
        for column, label in STAGE_ORDER:
            values = [f"{float(subset.loc[key, column]):.3f}" for key in variant_keys]
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        lines.append("")
        lines.append(header.replace("Stage", "Metric"))
        lines.append(divider)
        fps_values = [f"{float(subset.loc[key, 'sia_active_fps']):.3f}" for key in variant_keys]
        fraction_values = [f"{float(subset.loc[key, 'sia_active_fraction']):.4f}" for key in variant_keys]
        lines.append(f"| SiA Active FPS | " + " | ".join(fps_values) + " |")
        lines.append(f"| SiA Active Fraction | " + " | ".join(fraction_values) + " |")
        lines.append("")

    (output_dir / "runtime_stage_breakdown_tables.md").write_text("\n".join(lines), encoding="utf-8")


def build_manifest(generated_files):
    lines = [
        "# Plot Exports",
        "",
        "Generated plots:",
    ]
    for name in generated_files:
        lines.append(f"- {name}")
    lines.extend(
        [
            "",
            "Suggested usage:",
            "- backend_inference_latency.png for backend acceleration summary",
            "- backend_active_loop_latency.png for end-to-end active-path overhead comparison",
            "- backend_sia_active_fps.png for throughput while SiA is active",
            "- backend_sia_active_fraction.png for how often the scheduler needed SiA",
            "- runtime_stage_breakdown_live_like_always_on.png for the midterm-style live-like always-on stage comparison",
            "- runtime_stage_breakdown_live_like_staged.png for the midterm-style staged live-like stage comparison",
            "- tensorrt_stage_tradeoff_vs_pytorch_fp16.png for the TensorRT tradeoff explanation by stage",
            "- pipeline_duty_cycle.png for staged-pipeline duty-cycle explanation",
            "- full_pipeline_event_timeline.png for scheduler visualization",
            "- motion_to_sia_latency_histogram.png for responsiveness visualization",
        ]
    )
    return lines


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    backend_df = pd.read_csv(args.backend_latency_csv)
    stage_df = pd.read_csv(args.stage_breakdown_csv)
    activity_df = pd.read_csv(args.pipeline_activity_csv)
    event_df = pd.read_csv(args.event_timeline_csv)
    final_metrics_payload = load_json(args.final_metrics_json)

    plot_backend_latency(backend_df, output_dir)
    plot_backend_active_loop(backend_df, output_dir)
    plot_backend_fps(backend_df, output_dir)
    plot_backend_active_fraction(backend_df, output_dir)
    plot_stage_breakdown(stage_df, ("live_replay", "always_on"), output_dir, "runtime_stage_breakdown_live_like_always_on.png")
    plot_stage_breakdown(stage_df, ("live_replay", "motion_person_sia"), output_dir, "runtime_stage_breakdown_live_like_staged.png")
    plot_tensorrt_tradeoff(stage_df, output_dir)
    plot_pipeline_duty_cycle(activity_df, output_dir)
    plot_event_timeline(event_df, output_dir)
    write_stage_breakdown_tables(stage_df, output_dir)

    generated_files = [
        "backend_inference_latency.png",
        "backend_active_loop_latency.png",
        "backend_sia_active_fps.png",
        "backend_sia_active_fraction.png",
        "runtime_stage_breakdown_live_like_always_on.png",
        "runtime_stage_breakdown_live_like_staged.png",
        "tensorrt_stage_tradeoff_vs_pytorch_fp16.png",
        "pipeline_duty_cycle.png",
        "full_pipeline_event_timeline.png",
    ]
    latency_plot = plot_motion_to_sia_latency(final_metrics_payload, output_dir)
    if latency_plot:
        generated_files.append(latency_plot)

    write_json(
        output_dir / "plot_manifest.json",
        {
            "final_metrics_json": args.final_metrics_json,
            "backend_latency_csv": args.backend_latency_csv,
            "stage_breakdown_csv": args.stage_breakdown_csv,
            "pipeline_activity_csv": args.pipeline_activity_csv,
            "event_timeline_csv": args.event_timeline_csv,
            "generated_files": generated_files,
            "generated_tables": ["runtime_stage_breakdown_tables.md"],
        },
    )
    write_run_summary(output_dir / "README.md", build_manifest(generated_files))
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
