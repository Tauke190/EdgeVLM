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
    parser = argparse.ArgumentParser(description="Live runtime demo built on the shared runtime core.")
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video-device", type=int, help="Optional override for camera device index.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the camera.")
    parser.add_argument("--max-seconds", type=float, help="Optional cap on run duration.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing and preview.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_config = load_json(args.config)
    if args.video_device is not None:
        raw_config["video_device"] = args.video_device
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.output_root:
        raw_config["output_root"] = args.output_root
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.max_seconds is not None:
        raw_config["max_seconds"] = args.max_seconds
    if args.no_render:
        raw_config["render_enabled"] = False
        raw_config["show_preview"] = False

    config = RuntimeConfig.from_dict(raw_config)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])

    capture = open_capture(config)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = capture.get(cv2.CAP_PROP_FPS)

    run_dir = make_run_dir(config.output_root, "live_runtime_demo")
    output_video_path = run_dir / config.output_video_name
    writer = None
    if config.render_enabled:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*config.video_codec),
            config.output_fps,
            (frame_width, frame_height),
        )
    if config.show_preview:
        cv2.namedWindow("SiA Live Runtime", cv2.WINDOW_NORMAL)

    pipeline = AlwaysOnSIAPipeline(config)
    print(f"Live runtime initialized. Source: camera:{config.video_device}")
    print(f"Camera resolution: {frame_width}x{frame_height}")
    if source_fps and source_fps > 0:
        print(f"Reported camera FPS: {round(source_fps, 3)}")
    print(f"Recording output video: {'yes' if writer is not None else 'no'}")
    if writer is not None:
        print(f"Output path: {output_video_path}")
    print("Starting live frame processing...")
    start_wall = time.perf_counter()
    frame_count = 0
    active_frames = 0
    first_frame_logged = False
    writer_frames = 0
    first_recorded_frame_logged = False

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_count += 1
            if not first_frame_logged:
                print(f"First frame received from camera. Frame index: {frame_count}")
                first_frame_logged = True
            elapsed_so_far = time.perf_counter() - start_wall
            if config.max_frames and frame_count > config.max_frames:
                break
            if config.max_seconds and elapsed_so_far > config.max_seconds:
                break

            result = pipeline.process_frame(frame, (frame_height, frame_width))
            if result["active"]:
                active_frames += 1
            if writer is not None:
                target_frames = max(1, int((elapsed_so_far if elapsed_so_far > 0 else 0.0) * config.output_fps))
                while writer_frames < target_frames:
                    writer.write(result["rendered_frame"])
                    writer_frames += 1
                    if not first_recorded_frame_logged:
                        print(
                            "Recording started. "
                            f"Output FPS target: {config.output_fps}. "
                            f"First recorded output frame index: {writer_frames}"
                        )
                        first_recorded_frame_logged = True
            if config.show_preview:
                cv2.imshow("SiA Live Runtime", result["rendered_frame"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if config.show_preview:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start_wall
    metrics = {
        "mode": config.mode,
        "video_device": config.video_device,
        "weights_path": config.weights_path,
        "git_commit": infer_git_commit(),
        "frames_read": frame_count,
        "active_frames": active_frames,
        "frames_written": writer_frames,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else None,
        "render_enabled": config.render_enabled,
        "show_preview": config.show_preview,
        "output_fps": config.output_fps if writer is not None else None,
        "output_duration_s": round(writer_frames / config.output_fps, 3)
        if writer is not None and config.output_fps > 0
        else None,
    }
    write_json(run_dir / "config.json", raw_config)
    write_json(run_dir / "metrics.json", metrics)
    write_run_summary(
        run_dir / "run_summary.txt",
        [
            f"Run directory: {run_dir}",
            f"Command: {invoked_command}",
            f"Video device: {config.video_device}",
            f"Weights: {config.weights_path}",
            f"Frames read: {frame_count}",
            f"Active frames: {active_frames}",
            f"Frames written: {writer_frames}",
            f"Elapsed seconds: {metrics['elapsed_s']}",
            f"Effective FPS: {metrics['effective_fps']}",
            f"Output FPS: {metrics['output_fps']}",
            f"Output duration seconds: {metrics['output_duration_s']}",
            f"Output video: {output_video_path if writer is not None else 'disabled'}",
        ],
    )
    print(f"Live runtime demo complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
