import argparse
from collections import Counter
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

from runtime import (
    AlwaysOnSIAPipeline,
    RuntimeConfig,
    RuntimeMetricsCollector,
    STAGE_TIMING_FIELDNAMES,
    open_capture,
)
from runtime.visualize import draw_predictions, resolve_color
from tools.baseline_utils import ensure_dir, infer_git_commit, load_json, make_run_dir, to_builtin, write_csv, write_json, write_run_summary
from tools.offline_runtime_demo import EVENT_LOG_FIELDNAMES
from tools.system_monitor import SystemMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Live runtime demo built on the shared runtime core.")
    parser.add_argument("--config", required=True, help="Path to a runtime config JSON file.")
    parser.add_argument("--video-device", type=int, help="Optional override for camera device index.")
    parser.add_argument("--video", help="Optional live replay source. When provided, the live path can be validated against a file.")
    parser.add_argument("--simulate-live", action="store_true", help="Pace file input against wall clock when --video is provided.")
    parser.add_argument("--drop-frames", action="store_true", help="Drop replay frames when processing falls behind live pacing.")
    parser.add_argument("--target-fps", type=float, help="Optional override for live replay pacing FPS.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], help="Optional precision override.")
    parser.add_argument("--backend-name", choices=["pytorch", "tensorrt"], help="Optional runtime backend override.")
    parser.add_argument("--trt-engine-path", help="Optional TensorRT engine override when using the tensorrt backend.")
    parser.add_argument("--output-root", help="Optional override for the output root.")
    parser.add_argument("--output-dir", help="Optional explicit run directory for this invocation.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source.")
    parser.add_argument("--max-seconds", type=float, help="Optional cap on run duration.")
    parser.add_argument("--no-render", action="store_true", help="Disable output video writing and preview.")
    return parser.parse_args()


def build_raw_config(args):
    raw_config = load_json(args.config)
    if args.video_device is not None:
        raw_config["video_device"] = args.video_device
    if args.video:
        raw_config["video_path"] = args.video
    if args.simulate_live:
        raw_config["simulate_live"] = True
    if args.drop_frames:
        raw_config["drop_frames"] = True
    if args.target_fps is not None:
        raw_config["source_fps_override"] = args.target_fps
    if args.weights:
        raw_config["weights_path"] = args.weights
    if args.precision:
        raw_config["precision"] = args.precision
        raw_config["autocast"] = args.precision == "fp16"
    if args.backend_name:
        raw_config["backend_name"] = args.backend_name
    if args.trt_engine_path:
        raw_config["trt_engine_path"] = args.trt_engine_path
    if args.output_root:
        raw_config["output_root"] = args.output_root
    if args.max_frames is not None:
        raw_config["max_frames"] = args.max_frames
    if args.max_seconds is not None:
        raw_config["max_seconds"] = args.max_seconds
    if args.no_render:
        raw_config["render_enabled"] = False
        raw_config["show_preview"] = False
    return raw_config


def draw_status_banner(frame, status_lines, color):
    rendered = frame.copy()
    y = 24
    for line in status_lines:
        cv2.putText(
            rendered,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 22
    return rendered


def build_status_lines(result):
    gate_state = result["gate_state"]
    motion_active = bool(gate_state.get("motion_active"))
    person_active = bool(gate_state.get("person_active"))
    lines = [
        f"state={result.get('scheduler_state', 'unknown')} active={int(bool(result['active']))}",
        f"motion={int(motion_active)} person={int(person_active)} detector_run={int(bool(gate_state.get('person_detector_ran')))}",
    ]
    trigger_reason = result.get("sia_trigger_reason")
    if trigger_reason:
        lines.append(f"sia_trigger={trigger_reason}")
    return lines


def build_output_frame(frame, result, overlay_color, config):
    if not config.render_enabled and not config.show_preview:
        return None

    output_frame = frame.copy()
    if result["active"]:
        output_frame = draw_predictions(
            output_frame,
            result["boxes"],
            result["labels"],
            result["scores"],
            overlay_color,
            config.font_scale,
            config.line_thickness,
        )
    if config.show_preview:
        output_frame = draw_status_banner(output_frame, build_status_lines(result), overlay_color)
    return output_frame


def run_live_runtime(raw_config, invoked_command, run_name="live_runtime_demo", run_dir=None):
    config = RuntimeConfig.from_dict(raw_config)
    capture = open_capture(config)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    reported_source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = (
        float(config.source_fps_override)
        if config.source_fps_override
        else (reported_source_fps if reported_source_fps and reported_source_fps > 0 else config.output_fps)
    )
    writer_fps = source_fps if source_fps and source_fps > 0 else config.output_fps
    source_name = config.video_path if config.video_path else f"camera:{config.video_device}"

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

    if config.show_preview:
        cv2.namedWindow("SiA Live Runtime", cv2.WINDOW_NORMAL)

    pipeline = AlwaysOnSIAPipeline(config)
    collector = RuntimeMetricsCollector()
    overlay_color = resolve_color(config.color)
    system_metrics_path = run_dir / "system_metrics.csv"
    system_monitor = SystemMonitor(
        system_metrics_path,
        sample_interval_s=config.system_metrics_interval_s,
    )

    frame_queue = queue.Queue(maxsize=max(64, config.buffer_max_len * 4))
    stop_event = threading.Event()
    state_lock = threading.Lock()
    shared_state = {
        "capture_frames_read": 0,
        "frames_dropped": 0,
        "frames_written": 0,
        "latest_output_frame": None,
        "first_frame_logged": False,
        "first_recorded_frame_logged": False,
    }

    active_frames = 0
    output_ready_frames = 0
    motion_active_frames = 0
    person_active_frames = 0
    person_detector_frames = 0
    clips_processed = 0
    scheduler_state_counts = Counter()
    event_rows = []
    prev_scheduler_state = None
    prev_motion_active = False
    prev_person_active = False
    prev_active = False
    motion_event_count = 0
    person_event_count = 0
    sia_activation_count = 0
    sia_stride_wait_frames = 0
    sia_trigger_reason_counts = Counter()
    activation_latency_frames = []
    last_motion_start_frame = None
    source_exhausted = False

    def enqueue_frame(frame, capture_index, capture_s):
        try:
            frame_queue.put_nowait((frame, capture_index, capture_s))
            return
        except queue.Full:
            pass
        try:
            frame_queue.get_nowait()
            with state_lock:
                shared_state["frames_dropped"] += 1
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait((frame, capture_index, capture_s))
        except queue.Full:
            with state_lock:
                shared_state["frames_dropped"] += 1

    def replay_sleep_or_drop(frame_idx, replay_start_time):
        if not (config.video_path and config.simulate_live and source_fps > 0):
            return frame_idx

        frame_interval_s = 1.0 / source_fps
        target_time = replay_start_time + ((frame_idx - 1) * frame_interval_s)
        now = time.perf_counter()
        if now < target_time:
            time.sleep(target_time - now)
            return frame_idx

        if not config.drop_frames:
            return frame_idx

        behind_s = now - target_time
        frames_behind = int(behind_s / frame_interval_s)
        dropped_here = 0
        while frames_behind > 0 and not stop_event.is_set():
            grabbed = capture.grab()
            if not grabbed:
                break
            frame_idx += 1
            dropped_here += 1
            frames_behind -= 1

        if dropped_here:
            with state_lock:
                shared_state["frames_dropped"] += dropped_here
        return frame_idx

    def capture_worker():
        nonlocal source_exhausted
        replay_start_time = time.perf_counter()
        frame_idx = 0
        while not stop_event.is_set():
            frame_idx += 1
            frame_idx = replay_sleep_or_drop(frame_idx, replay_start_time)

            capture_start = time.perf_counter()
            ret, frame = capture.read()
            capture_time = time.perf_counter() - capture_start
            if not ret:
                source_exhausted = True
                stop_event.set()
                break

            with state_lock:
                shared_state["capture_frames_read"] += 1
                capture_count = shared_state["capture_frames_read"]
                if not shared_state["first_frame_logged"]:
                    print(f"First source frame received. Frame index: {capture_count}")
                    shared_state["first_frame_logged"] = True
            stop_after_enqueue = bool(config.max_frames and capture_count >= config.max_frames)

            enqueue_frame(frame, capture_count, capture_time)
            if stop_after_enqueue:
                stop_event.set()
                break

    print(f"Live runtime initialized. Source: {source_name}")
    print(f"Pipeline mode: {config.pipeline_mode}")
    print(f"Source resolution: {frame_width}x{frame_height}")
    print(f"Source FPS target: {round(source_fps, 3) if source_fps else 'unknown'}")
    if config.video_path and config.simulate_live:
        print(f"Replay-as-live pacing: enabled (drop_frames={'yes' if config.drop_frames else 'no'})")
    print(f"Recording output video: {'yes' if writer is not None else 'no'}")
    if writer is not None:
        print(f"Output path: {output_video_path}")
    print("Starting live frame processing...")

    system_monitor.start()
    start_wall = time.perf_counter()
    capture_thread = threading.Thread(target=capture_worker, name="live-capture", daemon=True)
    capture_thread.start()

    try:
        while not stop_event.is_set() or not frame_queue.empty():
            elapsed_so_far = time.perf_counter() - start_wall
            with state_lock:
                capture_frames_read = shared_state["capture_frames_read"]
            if config.max_frames and capture_frames_read >= config.max_frames:
                stop_event.set()
            if config.max_seconds and elapsed_so_far > config.max_seconds:
                stop_event.set()

            try:
                frame, capture_index, capture_time = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                if config.show_preview:
                    with state_lock:
                        preview_frame = (
                            None if shared_state["latest_output_frame"] is None else shared_state["latest_output_frame"].copy()
                        )
                    if preview_frame is not None:
                        cv2.imshow("SiA Live Runtime", preview_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                continue

            loop_start = time.perf_counter()
            result = pipeline.process_frame(frame, (frame_height, frame_width))
            output_frame = build_output_frame(frame, result, overlay_color, config)

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
                person_detector_frames += 1

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
                        "frame_index": capture_index,
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
                last_motion_start_frame = capture_index
                event_rows.append(
                    {
                        "frame_index": capture_index,
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
                        "frame_index": capture_index,
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
                        "frame_index": capture_index,
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
                        "frame_index": capture_index,
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
                    latency_frames = capture_index - last_motion_start_frame
                    activation_latency_frames.append(latency_frames)
                    latency_note = f"motion_to_sia_frames={latency_frames}"
                event_rows.append(
                    {
                        "frame_index": capture_index,
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
                        "frame_index": capture_index,
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
                        "frame_index": capture_index,
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
            if writer is not None and output_frame is not None:
                write_start = time.perf_counter()
                writer.write(output_frame)
                render_time += time.perf_counter() - write_start
                with state_lock:
                    shared_state["frames_written"] += 1
                    if not shared_state["first_recorded_frame_logged"]:
                        print(
                            "Recording started. "
                            f"Output FPS: {round(writer_fps, 3)}. "
                            f"First recorded output frame index: {shared_state['frames_written']}"
                        )
                        shared_state["first_recorded_frame_logged"] = True

            if config.show_preview and output_frame is not None:
                with state_lock:
                    shared_state["latest_output_frame"] = output_frame.copy()

            loop_time = time.perf_counter() - loop_start
            collector.record_frame(
                frame_index=capture_index,
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

            if config.show_preview and output_frame is not None:
                cv2.imshow("SiA Live Runtime", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        system_monitor.stop()
        capture.release()
        if writer is not None:
            writer.release()
        if config.show_preview:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start_wall
    with state_lock:
        frames_read = shared_state["capture_frames_read"]
        frames_written = shared_state["frames_written"]
        frames_dropped = shared_state["frames_dropped"]

    metrics = {
        "mode": config.mode,
        "pipeline_mode": config.pipeline_mode,
        "backend_name": config.backend_name,
        "optimization_label": config.optimization_label,
        "source": source_name,
        "video_device": None if config.video_path else config.video_device,
        "video_path": config.video_path,
        "simulate_live": config.simulate_live,
        "drop_frames": config.drop_frames,
        "source_fps": round(source_fps, 3) if source_fps else None,
        "weights_path": config.weights_path,
        "git_commit": infer_git_commit(),
        "device": str(pipeline.core.device),
        "precision": config.precision,
        "autocast": config.autocast,
        "sia_target_fps": config.sia_target_fps,
        "render_enabled": config.render_enabled,
        "show_preview": config.show_preview,
        "frames_read": frames_read,
        "frames_dropped": frames_dropped,
        "frames_written": frames_written,
        "active_frames": active_frames,
        "output_ready_frames": output_ready_frames,
        "clips_processed": clips_processed,
        "motion_active_frames": motion_active_frames,
        "person_active_frames": person_active_frames,
        "person_detector_frames": person_detector_frames,
        "motion_event_count": motion_event_count,
        "person_event_count": person_event_count,
        "sia_activation_count": sia_activation_count,
        "sia_stride_wait_frames": sia_stride_wait_frames,
        "sia_trigger_reason_counts": dict(sorted(sia_trigger_reason_counts.items())),
        "motion_to_sia_latency_frames": activation_latency_frames,
        "scheduler_state_counts": dict(sorted(scheduler_state_counts.items())),
        "elapsed_s": round(elapsed_s, 3),
        "effective_input_fps": round(frames_read / elapsed_s, 3) if elapsed_s > 0 else None,
        "effective_active_fps": round(clips_processed / elapsed_s, 3) if elapsed_s > 0 else None,
        "output_fps": round(writer_fps, 3) if writer is not None else None,
        "output_duration_s": round(frames_written / writer_fps, 3) if writer is not None and writer_fps > 0 else None,
        "monitor_source": system_monitor.source,
        "timings": collector.summarized_timings(),
        "source_exhausted": source_exhausted,
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
            "sia_target_fps": config.sia_target_fps,
            "render_enabled": config.render_enabled,
            "show_preview": config.show_preview,
            "simulate_live": config.simulate_live,
            "drop_frames": config.drop_frames,
            "source_fps_override": config.source_fps_override,
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
            f"Source: {source_name}",
            f"Pipeline mode: {config.pipeline_mode}",
            f"Backend: {config.backend_name}",
            f"Optimization label: {config.optimization_label or 'none'}",
            f"Device: {pipeline.core.device}",
            f"Precision: {config.precision}",
            f"Autocast enabled: {config.autocast}",
            f"SiA target FPS cap: {config.sia_target_fps}",
            f"Frames read: {frames_read}",
            f"Frames dropped: {frames_dropped}",
            f"Active frames: {active_frames}",
            f"Output-ready frames: {output_ready_frames}",
            f"Clips processed: {clips_processed}",
            f"Frames written: {frames_written}",
            f"Motion-active frames: {motion_active_frames}",
            f"Person-active frames: {person_active_frames}",
            f"Person-detector frames: {person_detector_frames}",
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
            f"Render mean ms: {metrics['timings']['render']['mean_ms']}",
            f"Loop mean ms: {metrics['timings']['loop']['mean_ms']}",
            f"Output video: {output_video_path if frames_written > 0 else 'not_generated'}",
        ],
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
    result = run_live_runtime(
        raw_config,
        invoked_command,
        run_name="live_runtime_demo",
        run_dir=args.output_dir,
    )
    print(f"Live runtime demo complete. Artifacts saved to: {result['run_dir']}")
    print(f"Metrics: {result['run_dir'] / 'metrics.json'}")
    print(f"Stage timings: {result['run_dir'] / 'stage_timings.csv'}")
    print(f"Event log: {result['run_dir'] / 'event_log.csv'}")
    print(f"System metrics: {result['run_dir'] / 'system_metrics.csv'}")


if __name__ == "__main__":
    main()
