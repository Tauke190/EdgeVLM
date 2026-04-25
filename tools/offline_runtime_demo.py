import argparse
from pathlib import Path
import shlex
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import AlwaysOnSIAPipeline, RuntimeConfig, open_capture
from tools.baseline_utils import infer_git_commit, load_json, make_run_dir, write_json, write_run_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Offline runtime demo built on the shared runtime core.")
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video", help="Optional override for the input video path.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_config = load_json(args.config)
    if args.video:
        raw_config["video_path"] = args.video
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.output_root:
        raw_config["output_root"] = args.output_root
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.no_render:
        raw_config["render_enabled"] = False

    config = RuntimeConfig.from_dict(raw_config)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    capture = open_capture(config)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    writer_fps = source_fps if source_fps and source_fps > 0 else config.output_fps

    run_dir = make_run_dir(config.output_root, "offline_runtime_demo")
    output_video_path = run_dir / config.output_video_name
    writer = None
    if config.render_enabled:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*config.video_codec),
            writer_fps,
            (frame_width, frame_height),
        )

    pipeline = AlwaysOnSIAPipeline(config)
    start_wall = time.perf_counter()
    frame_count = 0
    active_frames = 0

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_count += 1
            if config.max_frames and frame_count > config.max_frames:
                break

            result = pipeline.process_frame(frame, (frame_height, frame_width))
            if result["active"]:
                active_frames += 1
            if writer is not None:
                writer.write(result["rendered_frame"])
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_s = time.perf_counter() - start_wall
    metrics = {
        "mode": config.mode,
        "video_path": config.video_path,
        "weights_path": config.weights_path,
        "git_commit": infer_git_commit(),
        "frames_read": frame_count,
        "active_frames": active_frames,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else None,
        "source_fps": round(source_fps, 3) if source_fps and source_fps > 0 else None,
        "render_enabled": config.render_enabled,
    }
    write_json(run_dir / "config.json", raw_config)
    write_json(run_dir / "metrics.json", metrics)
    write_run_summary(
        run_dir / "run_summary.txt",
        [
            f"Run directory: {run_dir}",
            f"Command: {invoked_command}",
            f"Video: {config.video_path}",
            f"Weights: {config.weights_path}",
            f"Frames read: {frame_count}",
            f"Active frames: {active_frames}",
            f"Elapsed seconds: {metrics['elapsed_s']}",
            f"Effective FPS: {metrics['effective_fps']}",
            f"Output video: {output_video_path if writer is not None else 'disabled'}",
        ],
    )
    print(f"Offline runtime demo complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
