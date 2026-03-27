import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sia import PostProcessViz, get_sia
from tools.baseline_utils import (
    infer_git_commit,
    load_actions,
    load_json,
    make_run_dir,
    maybe_cuda_synchronize,
    resolve_color,
    summarize_series,
    to_builtin,
    write_csv,
    write_json,
    write_run_summary,
)
from tools.system_monitor import SystemMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Measured live baseline runner for SiA.")
    parser.add_argument("--config", required=True, help="Path to a live baseline config JSON file.")
    parser.add_argument("--video-device", type=int, help="Optional override for camera device index.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the results root directory.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the camera.")
    parser.add_argument("--max-seconds", type=float, help="Optional cap on run duration.")
    parser.add_argument(
        "--actions",
        type=lambda value: value.split(","),
        help="Optional comma-separated action override.",
    )
    parser.add_argument("--top-k-labels", type=int, help="Optional cap on labels rendered per box.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable drawing, preview, and output writing so timing focuses on non-render stages.",
    )
    return parser.parse_args()


def decode_text_labels(label_ids_per_box, score_values_per_box, captions, top_k_labels=None):
    decoded_labels = []
    decoded_scores = []
    for label_ids, score_values in zip(label_ids_per_box, score_values_per_box):
        if top_k_labels is not None:
            label_ids = label_ids[:top_k_labels]
            score_values = score_values[:top_k_labels]
        decoded_labels.append([captions[index] for index in label_ids])
        decoded_scores.append([float(score) for score in score_values])
    return decoded_labels, decoded_scores


def draw_predictions(frame, boxes, labels, scores, color, font_scale, thickness, disable_empty):
    rendered = frame.copy()
    for box, label_list, score_list in zip(boxes, labels, scores):
        if disable_empty and hasattr(score_list, "shape") and 0 in score_list.shape:
            continue
        box_np = box.detach().cpu().numpy()
        start_point = (int(box_np[0]), int(box_np[1]))
        end_point = (int(box_np[2]), int(box_np[3]))
        cv2.rectangle(rendered, start_point, end_point, color, thickness)
        offset = 20
        for label, score in zip(label_list, score_list):
            text = f"{label} {round(float(score), 2)}"
            cv2.putText(
                rendered,
                text,
                (int(box_np[0]) - 5, int(box_np[1]) + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            offset += 20
    return rendered


def main():
    args = parse_args()
    config = load_json(args.config)

    video_device = args.video_device if args.video_device is not None else int(config.get("video_device", 0))
    weights_path = args.weights or config["weights_path"]
    output_root = args.output_root or config["output_root"]
    captions = load_actions(config["actions_json"], actions_override=args.actions)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    max_frames = args.max_frames or config.get("max_frames")
    max_seconds = args.max_seconds or config.get("max_seconds")
    top_k_labels = args.top_k_labels if args.top_k_labels is not None else config.get("top_k_labels")
    sync_cuda_timing = bool(config.get("sync_cuda_timing", True))

    run_name = config.get("run_name", "live_fp32_baseline")
    run_dir = make_run_dir(output_root, run_name)
    output_video_path = run_dir / config.get("output_video_name", "pred_live.mp4")
    system_metrics_path = run_dir / "system_metrics.csv"

    color = resolve_color(config.get("color", "green"))
    font_scale = config.get("font_scale", 1.0)
    thickness = config.get("line_thickness", 1)
    img_height, img_width = config.get("img_size", [240, 320])
    num_frames = int(config.get("num_frames", 9))
    buffer_max_len = int(config.get("buffer_max_len", 18))
    sample_stride = max(1, buffer_max_len // num_frames)
    sample_indices = np.arange(0, buffer_max_len, sample_stride)[:num_frames]
    threshold = float(config.get("threshold", 0.25))
    output_fps = float(config.get("output_fps", 20))
    disable_empty = bool(config.get("disable_empty", False))
    record_output = bool(config.get("record_output", True))
    show_preview = bool(config.get("show_preview", False))
    if args.no_render:
        record_output = False
        show_preview = False

    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device '{video_device}'.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(config.get("frame_width", 640))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(config.get("frame_height", 480))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if record_output:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*config.get("video_codec", "mp4v")),
            output_fps,
            (frame_width, frame_height),
        )

    write_json(
        run_dir / "config.json",
        {
            **config,
            "weights_path": weights_path,
            "output_root": str(output_root),
            "resolved_device": device,
            "resolved_actions": captions,
            "resolved_video_device": video_device,
            "record_output": record_output,
            "show_preview": show_preview,
            "max_frames": max_frames,
            "max_seconds": max_seconds,
            "top_k_labels": top_k_labels,
            "sync_cuda_timing": sync_cuda_timing,
        },
    )

    model = get_sia(
        size=config.get("model_size", "b"),
        pretrain=config.get("pretrain"),
        det_token_num=int(config.get("det_token_num", 20)),
        text_lora=bool(config.get("text_lora", True)),
        num_frames=num_frames,
    )["sia"]
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True),
        strict=False,
    )
    model.to(device)
    model.eval()

    normalizer = v2.Normalize(
        config.get("normalize_mean", [0.485, 0.456, 0.406]),
        config.get("normalize_std", [0.229, 0.224, 0.225]),
    )
    postprocess = PostProcessViz()

    text_embeds = model.encode_text(captions)
    text_embeds = F.normalize(text_embeds, dim=-1)

    buffer = []
    plotbuffer = []
    mididx = buffer_max_len // 2
    frame_count = 0
    frames_written = 0
    clips_processed = 0
    stage_rows = []
    inference_times = []
    loop_times = []
    capture_times = []
    preprocess_times = []
    postprocess_times = []
    postprocess_filter_times = []
    postprocess_nms_times = []
    postprocess_threshold_times = []
    label_decode_times = []
    render_times = []

    if show_preview:
        preview = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.namedWindow("SiA Live Baseline", cv2.WINDOW_NORMAL)
        cv2.imshow("SiA Live Baseline", preview)
        cv2.waitKey(1)

    system_monitor = SystemMonitor(
        system_metrics_path,
        sample_interval_s=float(config.get("system_metrics_interval_s", 1.0)),
    )
    system_monitor.start()
    start_wall = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            capture_start = time.perf_counter()
            ret, frame = cap.read()
            capture_time = time.perf_counter() - capture_start
            if not ret:
                break

            frame_count += 1
            elapsed_so_far = time.perf_counter() - start_wall
            if max_frames and frame_count > max_frames:
                break
            if max_seconds and elapsed_so_far > max_seconds:
                break

            raw_image = frame
            preprocess_start = time.perf_counter()
            plotbuffer.append(raw_image.transpose(2, 0, 1))
            resized = cv2.resize(raw_image, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            buffer.append(resized.transpose(2, 0, 1))
            preprocess_time = time.perf_counter() - preprocess_start

            inference_time = 0.0
            postprocess_time = 0.0
            postprocess_filter_time = 0.0
            postprocess_nms_time = 0.0
            postprocess_threshold_time = 0.0
            label_decode_time = 0.0
            render_time = 0.0
            detections = 0
            rendered_frame = raw_image.copy()

            if len(buffer) > buffer_max_len:
                buffer.pop(0)
                plotbuffer.pop(0)

                clip_tensor = torch.from_numpy(np.array(buffer)[sample_indices]).float() / 255.0
                clip_tensor = normalizer(clip_tensor).unsqueeze(0).to(device)

                maybe_cuda_synchronize(device, sync_cuda_timing)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.encode_vision(clip_tensor)
                    outputs["pred_logits"] = F.normalize(outputs["pred_logits"], dim=-1) @ text_embeds.T
                maybe_cuda_synchronize(device, sync_cuda_timing)
                inference_time = time.perf_counter() - inference_start

                maybe_cuda_synchronize(device, sync_cuda_timing)
                postprocess_start = time.perf_counter()
                result, postprocess_breakdown = postprocess(
                    outputs,
                    (frame_height, frame_width),
                    human_conf=0.9,
                    thresh=threshold,
                    return_stage_timings=True,
                )
                result = result[0]
                boxes = result["boxes"]
                label_ids = result["labels"]
                raw_scores = result["scores"]
                detections = len(boxes)
                maybe_cuda_synchronize(device, sync_cuda_timing)
                postprocess_time = time.perf_counter() - postprocess_start
                postprocess_filter_time = postprocess_breakdown["human_filter_s"]
                postprocess_nms_time = postprocess_breakdown["nms_s"]
                postprocess_threshold_time = postprocess_breakdown["threshold_s"]

                maybe_cuda_synchronize(device, sync_cuda_timing)
                label_decode_start = time.perf_counter()
                labels, scores = decode_text_labels(label_ids, raw_scores, captions, top_k_labels)
                maybe_cuda_synchronize(device, sync_cuda_timing)
                label_decode_time = time.perf_counter() - label_decode_start

                if writer is not None or show_preview:
                    maybe_cuda_synchronize(device, sync_cuda_timing)
                    render_start = time.perf_counter()
                    rendered_frame = draw_predictions(
                        plotbuffer[mididx].transpose(1, 2, 0).astype(np.uint8),
                        boxes,
                        labels,
                        scores,
                        color,
                        font_scale,
                        thickness,
                        disable_empty,
                    )
                    if writer is not None:
                        writer.write(rendered_frame)
                    if show_preview:
                        cv2.imshow("SiA Live Baseline", rendered_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    maybe_cuda_synchronize(device, sync_cuda_timing)
                    render_time = time.perf_counter() - render_start

                frames_written += 1
                clips_processed += 1
            elif show_preview:
                cv2.imshow("SiA Live Baseline", rendered_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            loop_time = time.perf_counter() - loop_start

            capture_times.append(capture_time)
            preprocess_times.append(preprocess_time)
            loop_times.append(loop_time)
            if inference_time > 0:
                inference_times.append(inference_time)
                postprocess_times.append(postprocess_time)
                postprocess_filter_times.append(postprocess_filter_time)
                postprocess_nms_times.append(postprocess_nms_time)
                postprocess_threshold_times.append(postprocess_threshold_time)
                label_decode_times.append(label_decode_time)
                render_times.append(render_time)

            stage_rows.append(
                {
                    "frame_index": frame_count,
                    "capture_ms": round(capture_time * 1000.0, 3),
                    "preprocess_ms": round(preprocess_time * 1000.0, 3),
                    "inference_ms": round(inference_time * 1000.0, 3),
                    "postprocess_ms": round(postprocess_time * 1000.0, 3),
                    "postprocess_filter_ms": round(postprocess_filter_time * 1000.0, 3),
                    "postprocess_nms_ms": round(postprocess_nms_time * 1000.0, 3),
                    "postprocess_threshold_ms": round(postprocess_threshold_time * 1000.0, 3),
                    "label_decode_ms": round(label_decode_time * 1000.0, 3),
                    "render_ms": round(render_time * 1000.0, 3),
                    "loop_ms": round(loop_time * 1000.0, 3),
                    "detections": detections,
                }
            )
    finally:
        system_monitor.stop()
        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start_wall
    metrics = {
        "video_device": video_device,
        "weights_path": weights_path,
        "git_commit": infer_git_commit(),
        "device": device,
        "num_actions": len(captions),
        "num_frames_per_clip": num_frames,
        "buffer_max_len": buffer_max_len,
        "sample_indices": sample_indices.tolist(),
        "threshold": threshold,
        "max_frames": max_frames,
        "max_seconds": max_seconds,
        "top_k_labels": top_k_labels,
        "sync_cuda_timing": sync_cuda_timing,
        "render_enabled": bool(writer is not None or show_preview),
        "monitor_source": system_monitor.source,
        "frames_read": frame_count,
        "frames_written": frames_written,
        "clips_processed": clips_processed,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frames_written / elapsed_s, 3) if elapsed_s > 0 else None,
        "source_fps": round(source_fps, 3) if source_fps and source_fps > 0 else None,
        "timings": {
            "capture": summarize_series(capture_times),
            "preprocess": summarize_series(preprocess_times),
            "inference": summarize_series(inference_times),
            "postprocess": summarize_series(postprocess_times),
            "postprocess_filter": summarize_series(postprocess_filter_times),
            "postprocess_nms": summarize_series(postprocess_nms_times),
            "postprocess_threshold": summarize_series(postprocess_threshold_times),
            "label_decode": summarize_series(label_decode_times),
            "render": summarize_series(render_times),
            "loop": summarize_series(loop_times),
        },
    }

    write_csv(
        run_dir / "stage_timings.csv",
        [
            "frame_index",
            "capture_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "postprocess_filter_ms",
            "postprocess_nms_ms",
            "postprocess_threshold_ms",
            "label_decode_ms",
            "render_ms",
            "loop_ms",
            "detections",
        ],
        stage_rows,
    )
    write_json(run_dir / "metrics.json", to_builtin(metrics))

    summary_lines = [
        f"Run directory: {run_dir}",
        f"Video device: {video_device}",
        f"Weights: {weights_path}",
        f"Device: {device}",
        f"Actions: {len(captions)}",
        f"Top-k labels per box: {top_k_labels if top_k_labels is not None else 'all'}",
        f"CUDA timing synchronization: {sync_cuda_timing}",
        f"Render enabled: {bool(writer is not None or show_preview)}",
        f"Monitor source: {system_monitor.source}",
        f"Frames read: {frame_count}",
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
        f"Output video: {output_video_path if writer is not None else 'disabled'}",
    ]
    write_run_summary(run_dir / "run_summary.txt", summary_lines)

    print(f"Live baseline run complete. Artifacts saved to: {run_dir}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Stage timings: {run_dir / 'stage_timings.csv'}")
    print(f"System metrics: {run_dir / 'system_metrics.csv'}")
    if writer is not None:
        print(f"Output video: {output_video_path}")


if __name__ == "__main__":
    main()
