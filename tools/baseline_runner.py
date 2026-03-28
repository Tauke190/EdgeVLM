import argparse
from contextlib import nullcontext
import shlex
import time
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Measured offline baseline runner for SiA.")
    parser.add_argument("--config", required=True, help="Path to a baseline config JSON file.")
    parser.add_argument("--video", help="Optional override for the input video path.")
    parser.add_argument("--weights", help="Optional override for the model weights path.")
    parser.add_argument("--output-root", help="Optional override for the results root directory.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on frames read from the source video.")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        help="Optional precision override for model inference.",
    )
    parser.add_argument(
        "--actions",
        type=lambda value: value.split(","),
        help="Optional comma-separated action override.",
    )
    parser.add_argument("--top-k-labels", type=int, help="Optional cap on labels rendered per box.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable drawing and video writing so timing focuses on non-render stages.",
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


def draw_predictions(frame, boxes, labels, scores, color, font_scale, thickness):
    rendered = frame.copy()
    for box, label_list, score_list in zip(boxes, labels, scores):
        box_np = box.detach().cpu().numpy()
        start_point = (int(box_np[0]), int(box_np[1]))
        end_point = (int(box_np[2]), int(box_np[3]))
        cv2.rectangle(rendered, start_point, end_point, color, thickness)
        offset = 0
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
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    config = load_json(args.config)

    video_path = args.video or config["video_path"]
    weights_path = args.weights or config["weights_path"]
    output_root = args.output_root or config["output_root"]
    captions = load_actions(config["actions_json"], actions_override=args.actions)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision or config.get("precision", "fp32")
    autocast_enabled = bool(config.get("autocast", precision == "fp16"))
    max_frames = args.max_frames or config.get("max_frames")
    top_k_labels = args.top_k_labels if args.top_k_labels is not None else config.get("top_k_labels")
    sync_cuda_timing = bool(config.get("sync_cuda_timing", True))
    render_enabled = not args.no_render if args.no_render else False
    if "render_enabled" in config and not args.no_render:
        render_enabled = bool(config.get("render_enabled"))

    run_name = config.get("run_name", "offline_fp32_baseline")
    run_dir = make_run_dir(output_root, run_name)
    output_video_path = run_dir / config.get("output_video_name", "pred_video.mp4")
    system_metrics_path = run_dir / "system_metrics.csv"

    write_json(
        run_dir / "config.json",
        {
            **config,
            "video_path": video_path,
            "weights_path": weights_path,
            "output_root": str(output_root),
            "resolved_device": device,
            "resolved_actions": captions,
            "precision": precision,
            "autocast": autocast_enabled,
            "top_k_labels": top_k_labels,
            "sync_cuda_timing": sync_cuda_timing,
            "render_enabled": render_enabled,
        },
    )

    color = resolve_color(config.get("color", "green"))
    font_scale = config.get("font_scale", 0.3)
    thickness = config.get("line_thickness", 1)
    img_height, img_width = config.get("img_size", [240, 320])
    buffer_max_len = int(config.get("buffer_max_len", 72))
    num_frames = int(config.get("num_frames", 9))
    sample_stride = max(1, buffer_max_len // num_frames)
    sample_indices = np.arange(0, buffer_max_len, sample_stride)[:num_frames]
    threshold = float(config.get("threshold", 0.25))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{video_path}'.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    writer_fps = source_fps if source_fps and source_fps > 0 else config.get("output_fps", 25)
    writer = None
    if render_enabled:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*config.get("video_codec", "mp4v")),
            writer_fps,
            (frame_width, frame_height),
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
    use_fp16 = precision == "fp16" and str(device).startswith("cuda")
    if precision == "fp16" and not use_fp16:
        raise RuntimeError("FP16 precision is only supported on CUDA in this benchmark runner.")
    if use_fp16 and not autocast_enabled:
        model.half()
    model.eval()

    normalizer = v2.Normalize(
        config.get("normalize_mean", [0.485, 0.456, 0.406]),
        config.get("normalize_std", [0.229, 0.224, 0.225]),
    )
    postprocess = PostProcessViz()

    text_autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and autocast_enabled
        else nullcontext()
    )
    with torch.no_grad():
        with text_autocast_context:
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
    active_loop_times = []
    capture_times = []
    preprocess_times = []
    postprocess_times = []
    postprocess_filter_times = []
    postprocess_nms_times = []
    postprocess_threshold_times = []
    label_decode_times = []
    render_times = []

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
            if max_frames and frame_count > max_frames:
                break
            preprocess_start = time.perf_counter()
            raw_image = frame
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
            labels = []
            scores = []
            boxes = []

            if len(buffer) > buffer_max_len:
                buffer.pop(0)
                plotbuffer.pop(0)

                clip_tensor = torch.from_numpy(np.array(buffer)[sample_indices]).float() / 255.0
                clip_tensor = normalizer(clip_tensor).unsqueeze(0).to(device)
                if use_fp16 and not autocast_enabled:
                    clip_tensor = clip_tensor.half()

                maybe_cuda_synchronize(device, sync_cuda_timing)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    inference_autocast_context = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if use_fp16 and autocast_enabled
                        else nullcontext()
                    )
                    with inference_autocast_context:
                        outputs = model.encode_vision(clip_tensor)
                        similarity_text_embeds = text_embeds.to(dtype=outputs["pred_logits"].dtype)
                        outputs["pred_logits"] = F.normalize(outputs["pred_logits"], dim=-1) @ similarity_text_embeds.T
                    outputs = {
                        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                        for key, value in outputs.items()
                    }
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
                detections = len(boxes)
                maybe_cuda_synchronize(device, sync_cuda_timing)
                postprocess_time = time.perf_counter() - postprocess_start
                postprocess_filter_time = postprocess_breakdown["human_filter_s"]
                postprocess_nms_time = postprocess_breakdown["nms_s"]
                postprocess_threshold_time = postprocess_breakdown["threshold_s"]

                maybe_cuda_synchronize(device, sync_cuda_timing)
                label_decode_start = time.perf_counter()
                labels, scores = decode_text_labels(result["labels"], result["scores"], captions, top_k_labels)
                maybe_cuda_synchronize(device, sync_cuda_timing)
                label_decode_time = time.perf_counter() - label_decode_start

                if render_enabled:
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
                    )
                    writer.write(rendered_frame)
                    maybe_cuda_synchronize(device, sync_cuda_timing)
                    render_time = time.perf_counter() - render_start

                frames_written += 1
                clips_processed += 1

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
                active_loop_times.append(loop_time)

            stage_rows.append(
                {
                    "frame_index": frame_count,
                    "active_iteration": int(inference_time > 0),
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

    elapsed_s = time.perf_counter() - start_wall
    metrics = {
        "video_path": video_path,
        "weights_path": weights_path,
        "git_commit": infer_git_commit(),
        "device": device,
        "precision": precision,
        "autocast": autocast_enabled,
        "num_actions": len(captions),
        "num_frames_per_clip": num_frames,
        "buffer_max_len": buffer_max_len,
        "sample_indices": sample_indices.tolist(),
        "threshold": threshold,
        "max_frames": max_frames,
        "top_k_labels": top_k_labels,
        "sync_cuda_timing": sync_cuda_timing,
        "render_enabled": render_enabled,
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
            "active_loop": summarize_series(active_loop_times),
        },
    }

    write_csv(
        run_dir / "stage_timings.csv",
        [
            "frame_index",
            "active_iteration",
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
        f"Command: {invoked_command}",
        f"Video: {video_path}",
        f"Weights: {weights_path}",
        f"Device: {device}",
        f"Precision: {precision}",
        f"Autocast enabled: {autocast_enabled}",
        f"Actions: {len(captions)}",
        f"Top-k labels per box: {top_k_labels if top_k_labels is not None else 'all'}",
        f"CUDA timing synchronization: {sync_cuda_timing}",
        f"Render enabled: {render_enabled}",
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
        f"Active loop mean ms: {metrics['timings']['active_loop']['mean_ms']}",
        f"Active loop p95 ms: {metrics['timings']['active_loop']['p95_ms']}",
        f"Output video: {output_video_path if writer is not None else 'disabled'}",
    ]
    write_run_summary(run_dir / "run_summary.txt", summary_lines)

    print(f"Baseline run complete. Artifacts saved to: {run_dir}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Stage timings: {run_dir / 'stage_timings.csv'}")
    print(f"System metrics: {run_dir / 'system_metrics.csv'}")
    if writer is not None:
        print(f"Output video: {output_video_path}")


if __name__ == "__main__":
    main()
