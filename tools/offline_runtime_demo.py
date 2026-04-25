import argparse
from pathlib import Path
import shlex
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import (
    AlwaysOnSIAPipeline,
    RuntimeConfig,
    RuntimeMetricsCollector,
    STAGE_TIMING_FIELDNAMES,
    open_capture,
)
from tools.baseline_utils import (
    ensure_dir,
    infer_git_commit,
    load_json,
    make_run_dir,
    to_builtin,
    write_csv,
    write_json,
    write_run_summary,
)
from tools.system_monitor import SystemMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Offline runtime demo built on the shared runtime core.")
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video", help="Optional override for the input video path.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--output-dir", help="Optional explicit run directory for this invocation.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    return parser.parse_args()


def build_raw_config(args):
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
    return raw_config


def run_offline_runtime(
    raw_config,
    invoked_command,
    run_name="offline_runtime_demo",
    run_dir=None,
    progress_callback=None,
):
    config = RuntimeConfig.from_dict(raw_config)

    capture = open_capture(config)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer_fps = source_fps if source_fps and source_fps > 0 else config.output_fps

    if run_dir is None:
        run_dir = make_run_dir(config.output_root, run_name)
    else:
        run_dir = Path(run_dir)
        ensure_dir(run_dir)
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
    collector = RuntimeMetricsCollector()
    system_metrics_path = run_dir / "system_metrics.csv"
    system_monitor = SystemMonitor(
        system_metrics_path,
        sample_interval_s=config.system_metrics_interval_s,
    )
    system_monitor.start()
    start_wall = time.perf_counter()
    frame_count = 0
    active_frames = 0
    frames_written = 0
    clips_processed = 0

    try:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "video_path": config.video_path,
                    "run_dir": str(run_dir),
                    "frame_count_hint": source_frame_count if source_frame_count > 0 else None,
                    "max_frames": config.max_frames,
                    "render_enabled": config.render_enabled,
                }
            )
        while True:
            loop_start = time.perf_counter()
            capture_start = time.perf_counter()
            ret, frame = capture.read()
            capture_time = time.perf_counter() - capture_start
            if not ret:
                break
            if config.max_frames and frame_count >= config.max_frames:
                break
            frame_count += 1

            result = pipeline.process_frame(frame, (frame_height, frame_width))
            if result["active"]:
                active_frames += 1
                clips_processed += 1
            render_time = result["timings"]["render_s"]
            if writer is not None and result["active"]:
                write_start = time.perf_counter()
                writer.write(result["rendered_frame"])
                render_time += time.perf_counter() - write_start
                frames_written += 1

            loop_time = time.perf_counter() - loop_start
            collector.record_frame(
                frame_index=frame_count,
                active_iteration=result["active"],
                capture_s=capture_time,
                preprocess_s=result["timings"]["preprocess_s"],
                inference_s=result["timings"]["inference_s"],
                postprocess_s=result["timings"]["postprocess_s"],
                postprocess_filter_s=result["timings"]["postprocess_filter_s"],
                postprocess_nms_s=result["timings"]["postprocess_nms_s"],
                postprocess_threshold_s=result["timings"]["postprocess_threshold_s"],
                label_decode_s=result["timings"]["label_decode_s"],
                render_s=render_time,
                loop_s=loop_time,
                detections=result["detections"],
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "frame",
                        "frame_index": frame_count,
                        "active_frames": active_frames,
                        "clips_processed": clips_processed,
                        "frames_written": frames_written,
                        "frame_count_hint": source_frame_count if source_frame_count > 0 else None,
                        "max_frames": config.max_frames,
                    }
                )
    finally:
        system_monitor.stop()
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_s = time.perf_counter() - start_wall
    metrics = {
        "mode": config.mode,
        "video_path": config.video_path,
        "weights_path": config.weights_path,
        "git_commit": infer_git_commit(),
        "device": str(pipeline.core.device),
        "precision": config.precision,
        "autocast": config.autocast,
        "num_actions": len(pipeline.core.captions),
        "num_frames_per_clip": config.num_frames,
        "buffer_max_len": config.buffer_max_len,
        "sample_indices": pipeline.buffer.sample_indices.tolist(),
        "threshold": config.threshold,
        "max_frames": config.max_frames,
        "top_k_labels": config.top_k_labels,
        "sync_cuda_timing": config.sync_cuda_timing,
        "frames_read": frame_count,
        "active_frames": active_frames,
        "frames_written": frames_written,
        "clips_processed": clips_processed,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(clips_processed / elapsed_s, 3) if elapsed_s > 0 else None,
        "source_fps": round(source_fps, 3) if source_fps and source_fps > 0 else None,
        "render_enabled": config.render_enabled,
        "monitor_source": system_monitor.source,
        "timings": collector.summarized_timings(),
    }
    write_json(
        run_dir / "config.json",
        {
            **raw_config,
            "resolved_device": str(pipeline.core.device),
            "precision": config.precision,
            "autocast": config.autocast,
            "top_k_labels": config.top_k_labels,
            "sync_cuda_timing": config.sync_cuda_timing,
            "render_enabled": config.render_enabled,
        },
    )
    write_csv(run_dir / "stage_timings.csv", STAGE_TIMING_FIELDNAMES, collector.stage_rows)
    write_json(run_dir / "metrics.json", to_builtin(metrics))
    write_run_summary(
        run_dir / "run_summary.txt",
        [
            f"Run directory: {run_dir}",
            f"Command: {invoked_command}",
            f"Video: {config.video_path}",
            f"Weights: {config.weights_path}",
            f"Device: {pipeline.core.device}",
            f"Precision: {config.precision}",
            f"Autocast enabled: {config.autocast}",
            f"Actions: {len(pipeline.core.captions)}",
            f"Top-k labels per box: {config.top_k_labels if config.top_k_labels is not None else 'all'}",
            f"CUDA timing synchronization: {config.sync_cuda_timing}",
            f"Monitor source: {system_monitor.source}",
            f"Frames read: {frame_count}",
            f"Active frames: {active_frames}",
            f"Frames written: {frames_written}",
            f"Clips processed: {clips_processed}",
            f"Elapsed seconds: {metrics['elapsed_s']}",
            f"Effective FPS: {metrics['effective_fps']}",
            f"Inference mean ms: {metrics['timings']['inference']['mean_ms']}",
            f"Inference p95 ms: {metrics['timings']['inference']['p95_ms']}",
            f"Postprocess mean ms: {metrics['timings']['postprocess']['mean_ms']}",
            f"Postprocess filter mean ms: {metrics['timings']['postprocess_filter']['mean_ms']}",
            f"Postprocess NMS mean ms: {metrics['timings']['postprocess_nms']['mean_ms']}",
            f"Postprocess threshold mean ms: {metrics['timings']['postprocess_threshold']['mean_ms']}",
            f"Label decode mean ms: {metrics['timings']['label_decode']['mean_ms']}",
            f"Render mean ms: {metrics['timings']['render']['mean_ms']}",
            f"Loop mean ms: {metrics['timings']['loop']['mean_ms']}",
            f"Active loop mean ms: {metrics['timings']['active_loop']['mean_ms']}",
            f"Active loop p95 ms: {metrics['timings']['active_loop']['p95_ms']}",
            f"Output video: {output_video_path if writer is not None else 'disabled'}",
        ],
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "complete",
                "run_dir": str(run_dir),
                "frames_read": frame_count,
                "active_frames": active_frames,
                "clips_processed": clips_processed,
                "frames_written": frames_written,
                "effective_fps": metrics["effective_fps"],
                "output_video_path": str(output_video_path) if writer is not None else None,
            }
        )
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "config": config,
        "output_video_path": output_video_path if writer is not None else None,
    }


def main():
    args = parse_args()
    raw_config = build_raw_config(args)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    result = run_offline_runtime(
        raw_config,
        invoked_command,
        run_name="offline_runtime_demo",
        run_dir=args.output_dir,
    )
    print(f"Offline runtime demo complete. Artifacts saved to: {result['run_dir']}")
    print(f"Metrics: {result['run_dir'] / 'metrics.json'}")
    print(f"Stage timings: {result['run_dir'] / 'stage_timings.csv'}")
    print(f"System metrics: {result['run_dir'] / 'system_metrics.csv'}")


if __name__ == "__main__":
    main()
