import argparse
from collections import Counter
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

EVENT_LOG_FIELDNAMES = [
    "frame_index",
    "event",
    "scheduler_state",
    "prev_scheduler_state",
    "motion_active",
    "person_active",
    "sia_active",
    "person_detector_ran",
    "sia_trigger_reason",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline runtime demo built on the shared runtime core.")
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video", help="Optional override for the input video path.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], help="Optional precision override.")
    parser.add_argument("--backend-name", choices=["pytorch", "tensorrt"], help="Optional runtime backend override.")
    parser.add_argument("--trt-engine-path", help="Optional TensorRT engine override when using the tensorrt backend.")
    parser.add_argument("--sia-target-fps", type=float, help="Optional override for the SiA activation FPS cap. Use 0 to disable.")
    parser.add_argument("--adaptive-sia-target-fps", action="store_true", help="Enable adaptive SiA FPS capping. Starts uncapped during warmup.")
    parser.add_argument("--adaptive-sia-warmup-frames", type=int, help="Number of SiA-active frames to observe before enabling the adaptive cap.")
    parser.add_argument("--adaptive-sia-utilization", type=float, help="Target fraction of measured active throughput to use as the adaptive cap.")
    parser.add_argument("--adaptive-sia-smoothing", type=float, help="EMA smoothing factor for adaptive active-loop timing.")
    parser.add_argument("--adaptive-sia-min-fps", type=float, help="Minimum adaptive SiA FPS cap after warmup.")
    parser.add_argument("--adaptive-sia-max-fps", type=float, help="Maximum adaptive SiA FPS cap after warmup.")
    parser.add_argument("--motion-min-on-time", type=int, help="Minimum number of frames to keep the motion gate open before allowing it to close.")
    parser.add_argument("--person-min-on-time", type=int, help="Minimum number of frames to keep the person gate open before allowing it to close.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--output-dir", help="Optional explicit run directory for this invocation.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing.")
    parser.add_argument("--show-active-tiers", action="store_true", help="Overlay tier activity indicators on rendered output.")
    return parser.parse_args()


def build_raw_config(args):
    raw_config = load_json(args.config)
    if args.video:
        raw_config["video_path"] = args.video
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.precision:
        raw_config["precision"] = args.precision
        raw_config["autocast"] = args.precision == "fp16"
    if args.backend_name:
        raw_config["backend_name"] = args.backend_name
    if args.trt_engine_path:
        raw_config["trt_engine_path"] = args.trt_engine_path
    if args.sia_target_fps is not None:
        raw_config["sia_target_fps"] = args.sia_target_fps
    if args.adaptive_sia_target_fps:
        raw_config["adaptive_sia_target_fps"] = True
    if args.adaptive_sia_warmup_frames is not None:
        raw_config["adaptive_sia_warmup_frames"] = args.adaptive_sia_warmup_frames
    if args.adaptive_sia_utilization is not None:
        raw_config["adaptive_sia_utilization"] = args.adaptive_sia_utilization
    if args.adaptive_sia_smoothing is not None:
        raw_config["adaptive_sia_smoothing"] = args.adaptive_sia_smoothing
    if args.adaptive_sia_min_fps is not None:
        raw_config["adaptive_sia_min_fps"] = args.adaptive_sia_min_fps
    if args.adaptive_sia_max_fps is not None:
        raw_config["adaptive_sia_max_fps"] = args.adaptive_sia_max_fps
    if args.motion_min_on_time is not None:
        raw_config["motion_min_on_time"] = args.motion_min_on_time
    if args.person_min_on_time is not None:
        raw_config["person_min_on_time"] = args.person_min_on_time
    if args.output_root:
        raw_config["output_root"] = args.output_root
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.no_render:
        raw_config["render_enabled"] = False
    if args.show_active_tiers:
        raw_config["show_active_tiers"] = True
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
    output_ready_frames = 0
    motion_active_frames = 0
    person_active_frames = 0
    person_detector_runs = 0
    scheduler_state_counts = Counter()
    event_rows = []
    prev_scheduler_state = None
    prev_motion_active = False
    prev_person_active = False
    prev_active = False
    last_motion_start_frame = None
    activation_latency_frames = []
    motion_event_count = 0
    person_event_count = 0
    sia_activation_count = 0
    sia_stride_wait_frames = 0
    sia_trigger_reason_counts = Counter()

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
            if result["output_ready"]:
                output_ready_frames += 1
            if result["gate_state"].get("motion_active"):
                motion_active_frames += 1
            if result["gate_state"].get("person_active"):
                person_active_frames += 1
            if result["gate_state"].get("person_detector_ran"):
                person_detector_runs += 1
            scheduler_state = result.get("scheduler_state", "unknown")
            scheduler_state_counts[scheduler_state] += 1
            if scheduler_state == "sia_stride_wait":
                sia_stride_wait_frames += 1
            motion_active = bool(result["gate_state"].get("motion_active"))
            person_active = bool(result["gate_state"].get("person_active"))
            sia_active = bool(result["active"])
            sia_trigger_reason = result.get("sia_trigger_reason")
            if sia_trigger_reason:
                sia_trigger_reason_counts[sia_trigger_reason] += 1

            if scheduler_state != prev_scheduler_state:
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "scheduler_transition",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": prev_scheduler_state or "",
                        "motion_active": motion_active,
                        "person_active": person_active,
                        "sia_active": sia_active,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": sia_trigger_reason or "",
                        "notes": "",
                    }
                )
                prev_scheduler_state = scheduler_state

            if motion_active and not prev_motion_active:
                motion_event_count += 1
                last_motion_start_frame = frame_count
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "motion_active",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": True,
                        "person_active": person_active,
                        "sia_active": sia_active,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )
            if not motion_active and prev_motion_active:
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "motion_inactive",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": False,
                        "person_active": person_active,
                        "sia_active": sia_active,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )
            if person_active and not prev_person_active:
                person_event_count += 1
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "person_active",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": motion_active,
                        "person_active": True,
                        "sia_active": sia_active,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )
            if not person_active and prev_person_active:
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "person_inactive",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": motion_active,
                        "person_active": False,
                        "sia_active": sia_active,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )
            if sia_active and not prev_active:
                sia_activation_count += 1
                latency_note = ""
                if last_motion_start_frame is not None:
                    latency_frames = frame_count - last_motion_start_frame
                    activation_latency_frames.append(latency_frames)
                    latency_note = f"motion_to_sia_frames={latency_frames}"
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "sia_active",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": motion_active,
                        "person_active": person_active,
                        "sia_active": True,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": sia_trigger_reason or "",
                        "notes": latency_note,
                    }
                )
            if not sia_active and prev_active:
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "sia_inactive",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": motion_active,
                        "person_active": person_active,
                        "sia_active": False,
                        "person_detector_ran": bool(result["gate_state"].get("person_detector_ran")),
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )
            if result["gate_state"].get("person_detector_ran"):
                event_rows.append(
                    {
                        "frame_index": frame_count,
                        "event": "person_detector_run",
                        "scheduler_state": scheduler_state,
                        "prev_scheduler_state": "",
                        "motion_active": motion_active,
                        "person_active": person_active,
                        "sia_active": sia_active,
                        "person_detector_ran": True,
                        "sia_trigger_reason": "",
                        "notes": "",
                    }
                )

            prev_motion_active = motion_active
            prev_person_active = person_active
            prev_active = sia_active
            render_time = result["timings"]["render_s"]
            if config.render_enabled and result["output_ready"]:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_video_path),
                        cv2.VideoWriter_fourcc(*config.video_codec),
                        writer_fps,
                        (frame_width, frame_height),
                    )
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
    active_loop_values = collector.series["active_loop"]
    active_loop_total_s = sum(active_loop_values)
    metrics = {
        "mode": config.mode,
        "pipeline_mode": config.pipeline_mode,
        "backend_name": config.backend_name,
        "optimization_label": config.optimization_label,
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
        "output_ready_frames": output_ready_frames,
        "frames_written": frames_written,
        "clips_processed": clips_processed,
        "motion_active_frames": motion_active_frames,
        "frames_with_motion": motion_active_frames,
        "person_active_frames": person_active_frames,
        "frames_with_person": person_active_frames,
        "person_detector_frames": person_detector_runs,
        "person_detector_runs": person_detector_runs,
        "frames_with_person_detector": person_detector_runs,
        "motion_event_count": motion_event_count,
        "person_event_count": person_event_count,
        "sia_activation_count": sia_activation_count,
        "sia_inference_iterations": pipeline.sia_inference_count,
        "action_inferences": pipeline.sia_inference_count,
        "frames_with_sia_active": active_frames,
        "sia_target_fps": config.sia_target_fps,
        "adaptive_sia_target_fps": config.adaptive_sia_target_fps,
        "adaptive_sia_warmup_frames": config.adaptive_sia_warmup_frames,
        "adaptive_sia_utilization": config.adaptive_sia_utilization,
        "adaptive_sia_smoothing": config.adaptive_sia_smoothing,
        "adaptive_sia_min_fps": config.adaptive_sia_min_fps,
        "adaptive_sia_max_fps": config.adaptive_sia_max_fps,
        "sia_target_fps_effective_final": pipeline.current_sia_target_fps,
        "adaptive_sia_target_fps_updates": pipeline.adaptive_cap_updates,
        "adaptive_sia_active_loop_ema_ms_final": pipeline._adaptive_cap_ema_ms(),
        "sia_stride_wait_frames": sia_stride_wait_frames,
        "sia_trigger_reason_counts": dict(sorted(sia_trigger_reason_counts.items())),
        "motion_to_sia_latency_frames": activation_latency_frames,
        "scheduler_state_counts": dict(sorted(scheduler_state_counts.items())),
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else None,
        "effective_input_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else None,
        "effective_active_fps": round(active_frames / active_loop_total_s, 3)
        if active_loop_total_s > 0
        else None,
        "motion_frame_fraction": round(motion_active_frames / frame_count, 4) if frame_count > 0 else None,
        "person_frame_fraction": round(person_active_frames / frame_count, 4) if frame_count > 0 else None,
        "sia_active_frame_fraction": round(active_frames / frame_count, 4) if frame_count > 0 else None,
        "person_detector_frame_fraction": round(person_detector_runs / frame_count, 4) if frame_count > 0 else None,
        "source_fps": round(source_fps, 3) if source_fps and source_fps > 0 else None,
        "render_enabled": config.render_enabled,
        "show_active_tiers": config.show_active_tiers,
        "monitor_source": system_monitor.source,
        "timings": collector.summarized_timings(),
    }
    write_json(
        run_dir / "config.json",
        {
            **raw_config,
            "resolved_device": str(pipeline.core.device),
            "pipeline_mode": config.pipeline_mode,
            "backend_name": config.backend_name,
            "optimization_label": config.optimization_label,
            "precision": config.precision,
            "autocast": config.autocast,
            "top_k_labels": config.top_k_labels,
            "sia_target_fps": config.sia_target_fps,
            "adaptive_sia_target_fps": config.adaptive_sia_target_fps,
            "adaptive_sia_warmup_frames": config.adaptive_sia_warmup_frames,
            "adaptive_sia_utilization": config.adaptive_sia_utilization,
            "adaptive_sia_smoothing": config.adaptive_sia_smoothing,
            "adaptive_sia_min_fps": config.adaptive_sia_min_fps,
            "adaptive_sia_max_fps": config.adaptive_sia_max_fps,
            "sync_cuda_timing": config.sync_cuda_timing,
            "render_enabled": config.render_enabled,
            "show_active_tiers": config.show_active_tiers,
        },
    )
    write_csv(run_dir / "stage_timings.csv", STAGE_TIMING_FIELDNAMES, collector.stage_rows)
    write_csv(run_dir / "event_log.csv", EVENT_LOG_FIELDNAMES, event_rows)
    write_json(run_dir / "metrics.json", to_builtin(metrics))
    write_run_summary(
        run_dir / "run_summary.txt",
        [
            f"Run directory: {run_dir}",
            f"Command: {invoked_command}",
            f"Video: {config.video_path}",
            f"Weights: {config.weights_path}",
            f"Pipeline mode: {config.pipeline_mode}",
            f"Backend: {config.backend_name}",
            f"Optimization label: {config.optimization_label or 'none'}",
            f"Device: {pipeline.core.device}",
            f"Precision: {config.precision}",
            f"Autocast enabled: {config.autocast}",
            f"Actions: {len(pipeline.core.captions)}",
            f"Top-k labels per box: {config.top_k_labels if config.top_k_labels is not None else 'all'}",
            f"SiA target FPS cap: {config.sia_target_fps}",
            f"Adaptive SiA cap enabled: {config.adaptive_sia_target_fps}",
            f"Adaptive SiA warmup frames: {config.adaptive_sia_warmup_frames}",
            f"Adaptive SiA utilization: {config.adaptive_sia_utilization}",
            f"Adaptive SiA effective final FPS cap: {round(pipeline.current_sia_target_fps, 3)}",
            f"Adaptive SiA cap updates: {pipeline.adaptive_cap_updates}",
            f"CUDA timing synchronization: {config.sync_cuda_timing}",
            f"Monitor source: {system_monitor.source}",
            f"Frames read: {frame_count}",
            f"Active frames: {active_frames}",
            f"Output-ready frames: {output_ready_frames}",
            f"Frames written: {frames_written}",
            f"Clips processed: {clips_processed}",
            f"Motion-active frames: {motion_active_frames}",
            f"Person-active frames: {person_active_frames}",
            f"Person-detector runs: {person_detector_runs}",
            f"Motion events: {motion_event_count}",
            f"Person events: {person_event_count}",
            f"SiA activations: {sia_activation_count}",
            f"SiA stride-wait frames: {sia_stride_wait_frames}",
            f"SiA trigger reason counts: {dict(sorted(sia_trigger_reason_counts.items()))}",
            f"Motion-to-SiA latency frames: {activation_latency_frames}",
            f"Scheduler state counts: {dict(sorted(scheduler_state_counts.items()))}",
            f"Elapsed seconds: {metrics['elapsed_s']}",
            f"Effective input FPS: {metrics['effective_input_fps']}",
            f"Effective active FPS: {metrics['effective_active_fps']}",
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
            f"Output video: {output_video_path if frames_written > 0 else 'not_generated'}",
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
                "output_video_path": str(output_video_path) if frames_written > 0 else None,
            }
        )
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "config": config,
        "output_video_path": output_video_path if frames_written > 0 else None,
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
    print(f"Event log: {result['run_dir'] / 'event_log.csv'}")
    print(f"System metrics: {result['run_dir'] / 'system_metrics.csv'}")


if __name__ == "__main__":
    main()
