import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the measured offline FP16 benchmark for SiA.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "fp16_runtime.json"),
        help="Path to an FP16 benchmark config JSON file.",
    )
    parser.add_argument("--video", help="Optional override for the input video path.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the results root directory.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument(
        "--actions",
        help="Optional comma-separated action override.",
    )
    parser.add_argument("--top-k-labels", type=int, help="Optional cap on labels rendered per box.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable drawing and video writing so timing focuses on non-render stages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "baseline_runner.py"),
        "--config",
        args.config,
        "--precision",
        "fp16",
    ]
    if args.video:
        command.extend(["--video", args.video])
    if args.weights:
        command.extend(["--weights", args.weights])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.actions:
        command.extend(["--actions", args.actions])
    if args.top_k_labels is not None:
        command.extend(["--top-k-labels", str(args.top_k_labels)])
    if args.no_render:
        command.append("--no-render")

    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
