import argparse
from pathlib import Path
import queue
import shlex
import sys
import threading
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import AlwaysOnSIAPipeline, RuntimeConfig, open_capture
from runtime.visualize import draw_predictions, resolve_color
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
    writer_fps = source_fps if source_fps and source_fps > 0 else config.output_fps

    run_dir = make_run_dir(config.output_root, "live_runtime_demo")
    output_video_path = run_dir / config.output_video_name
    writer = None
    if config.render_enabled:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*config.video_codec),
            writer_fps,
            (frame_width, frame_height),
        )
    if config.show_preview:
        cv2.namedWindow("SiA Live Runtime", cv2.WINDOW_NORMAL)

    pipeline = AlwaysOnSIAPipeline(config)
    overlay_color = resolve_color(config.color)
    frame_queue = queue.Queue(maxsize=max(64, config.buffer_max_len * 4))
    stop_event = threading.Event()
    state_lock = threading.Lock()
    shared_state = {
        "frames_captured": 0,
        "frames_written": 0,
        "first_frame_logged": False,
        "first_recorded_frame_logged": False,
        "latest_preview_frame": None,
        "latest_annotation": None,
    }

    def push_frame_for_inference(frame):
        try:
            frame_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def capture_worker():
        while not stop_event.is_set():
            ret, frame = capture.read()
            if not ret:
                stop_event.set()
                break

            with state_lock:
                shared_state["frames_captured"] += 1
                if not shared_state["first_frame_logged"]:
                    print(
                        "First frame received from camera. "
                        f"Frame index: {shared_state['frames_captured']}"
                    )
                    shared_state["first_frame_logged"] = True

            push_frame_for_inference(frame.copy())

            output_frame = frame.copy()
            with state_lock:
                latest_annotation = shared_state["latest_annotation"]
                if latest_annotation is not None:
                    output_frame = draw_predictions(
                        output_frame,
                        latest_annotation["boxes"],
                        latest_annotation["labels"],
                        latest_annotation["scores"],
                        overlay_color,
                        config.font_scale,
                        config.line_thickness,
                    )
                shared_state["latest_preview_frame"] = output_frame.copy()

            if writer is not None:
                writer.write(output_frame)
                with state_lock:
                    shared_state["frames_written"] += 1
                    if not shared_state["first_recorded_frame_logged"]:
                        print(
                            "Recording started. "
                            f"Output FPS: {round(writer_fps, 3)}. "
                            f"First recorded output frame index: {shared_state['frames_written']}"
                        )
                        shared_state["first_recorded_frame_logged"] = True

    print(f"Live runtime initialized. Source: camera:{config.video_device}")
    print(f"Camera resolution: {frame_width}x{frame_height}")
    if source_fps and source_fps > 0:
        print(f"Reported camera FPS: {round(source_fps, 3)}")
    print(f"Recording output video: {'yes' if writer is not None else 'no'}")
    if writer is not None:
        print(f"Output path: {output_video_path}")
    print("Starting live frame processing...")
    start_wall = time.perf_counter()
    active_frames = 0
    capture_thread = threading.Thread(target=capture_worker, name="live-capture", daemon=True)
    capture_thread.start()

    try:
        while not stop_event.is_set():
            elapsed_so_far = time.perf_counter() - start_wall
            with state_lock:
                frames_captured = shared_state["frames_captured"]
            if config.max_frames and frames_captured >= config.max_frames:
                break
            if config.max_seconds and elapsed_so_far > config.max_seconds:
                break

            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if config.show_preview:
                    with state_lock:
                        preview_frame = (
                            None
                            if shared_state["latest_preview_frame"] is None
                            else shared_state["latest_preview_frame"].copy()
                        )
                    if preview_frame is not None:
                        cv2.imshow("SiA Live Runtime", preview_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                continue

            result = pipeline.process_frame(frame, (frame_height, frame_width))
            if result["active"]:
                active_frames += 1
                with state_lock:
                    shared_state["latest_annotation"] = {
                        "boxes": result["boxes"],
                        "labels": result["labels"],
                        "scores": result["scores"],
                    }
            if config.show_preview:
                with state_lock:
                    preview_frame = (
                        None
                        if shared_state["latest_preview_frame"] is None
                        else shared_state["latest_preview_frame"].copy()
                    )
                if preview_frame is not None:
                    cv2.imshow("SiA Live Runtime", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        capture.release()
        if writer is not None:
            writer.release()
        if config.show_preview:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start_wall
    with state_lock:
        frames_read = shared_state["frames_captured"]
        frames_written = shared_state["frames_written"]
    metrics = {
        "mode": config.mode,
        "video_device": config.video_device,
        "weights_path": config.weights_path,
        "git_commit": infer_git_commit(),
        "frames_read": frames_read,
        "active_frames": active_frames,
        "frames_written": frames_written,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frames_read / elapsed_s, 3) if elapsed_s > 0 else None,
        "render_enabled": config.render_enabled,
        "show_preview": config.show_preview,
        "output_fps": round(writer_fps, 3) if writer is not None else None,
        "output_duration_s": round(frames_written / writer_fps, 3)
        if writer is not None and writer_fps > 0
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
            f"Frames read: {frames_read}",
            f"Active frames: {active_frames}",
            f"Frames written: {frames_written}",
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
