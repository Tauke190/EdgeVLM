import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_utils import load_run_metrics, safe_pct_change, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two saved baseline run directories.")
    parser.add_argument("--baseline-run", required=True, help="Path to the baseline run directory.")
    parser.add_argument("--candidate-run", required=True, help="Path to the candidate run directory.")
    parser.add_argument(
        "--output-json",
        help="Optional path to save the comparison summary as JSON.",
    )
    return parser.parse_args()


def timing_snapshot(metrics, stage):
    return metrics.get("timings", {}).get(stage, {})


def build_stage_comparison(baseline_metrics, candidate_metrics, stage):
    baseline_stage = timing_snapshot(baseline_metrics, stage)
    candidate_stage = timing_snapshot(candidate_metrics, stage)
    baseline_mean = baseline_stage.get("mean_ms")
    candidate_mean = candidate_stage.get("mean_ms")
    baseline_p95 = baseline_stage.get("p95_ms")
    candidate_p95 = candidate_stage.get("p95_ms")
    return {
        "baseline_mean_ms": baseline_mean,
        "candidate_mean_ms": candidate_mean,
        "delta_mean_ms": round(candidate_mean - baseline_mean, 3)
        if baseline_mean is not None and candidate_mean is not None
        else None,
        "delta_mean_pct": safe_pct_change(baseline_mean, candidate_mean),
        "baseline_p95_ms": baseline_p95,
        "candidate_p95_ms": candidate_p95,
        "delta_p95_ms": round(candidate_p95 - baseline_p95, 3)
        if baseline_p95 is not None and candidate_p95 is not None
        else None,
        "delta_p95_pct": safe_pct_change(baseline_p95, candidate_p95),
    }


def format_pct(value):
    return f"{value:+.2f}%" if value is not None else "n/a"


def format_ms(value):
    return f"{value:.3f}" if value is not None else "n/a"


def print_table(comparison):
    print("Run Comparison")
    print(f"Baseline:  {comparison['baseline_run']}")
    print(f"Candidate: {comparison['candidate_run']}")
    print("")
    print("| Metric | Baseline | Candidate | Delta |")
    print("| --- | ---: | ---: | ---: |")
    print(
        f"| Effective FPS | {comparison['baseline_effective_fps'] or 'n/a'} | "
        f"{comparison['candidate_effective_fps'] or 'n/a'} | {format_pct(comparison['effective_fps_delta_pct'])} |"
    )
    for stage, values in comparison["stages"].items():
        print(
            f"| {stage} mean ms | {format_ms(values['baseline_mean_ms'])} | "
            f"{format_ms(values['candidate_mean_ms'])} | {format_pct(values['delta_mean_pct'])} |"
        )
        print(
            f"| {stage} p95 ms | {format_ms(values['baseline_p95_ms'])} | "
            f"{format_ms(values['candidate_p95_ms'])} | {format_pct(values['delta_p95_pct'])} |"
        )


def main():
    args = parse_args()
    baseline_run = Path(args.baseline_run)
    candidate_run = Path(args.candidate_run)
    baseline_metrics = load_run_metrics(baseline_run)
    candidate_metrics = load_run_metrics(candidate_run)

    comparison = {
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "baseline_effective_fps": baseline_metrics.get("effective_fps"),
        "candidate_effective_fps": candidate_metrics.get("effective_fps"),
        "effective_fps_delta_pct": safe_pct_change(
            baseline_metrics.get("effective_fps"),
            candidate_metrics.get("effective_fps"),
        ),
        "stages": {},
    }
    for stage in ["capture", "preprocess", "inference", "postprocess", "render", "loop"]:
        comparison["stages"][stage] = build_stage_comparison(baseline_metrics, candidate_metrics, stage)

    print_table(comparison)
    if args.output_json:
        write_json(args.output_json, comparison)


if __name__ == "__main__":
    main()
