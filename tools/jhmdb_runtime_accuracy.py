import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import cv2
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import v2

from datasets.HMDB21 import HMDB21
from runtime import AlwaysOnSIAPipeline, RuntimeConfig
from runtime.inference_core import SIARuntimeCore
from tools.baseline_utils import ensure_dir, load_json, to_builtin, write_json
from tools.offline_runtime_demo import (
    PREDICTION_FIELDNAMES,
    SIA_INFERENCE_FRAME_FIELDNAMES,
    _center_frame_metadata,
    _prediction_artifact_rows,
    run_offline_runtime,
)
from util.box_ops import box_cxcywh_to_xyxy


DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "jhmdb" / "videos"
DEFAULT_ANNO = REPO_ROOT / "anno" / "JHMDB-GT.pkl"
DEFAULT_ACTIONS = REPO_ROOT / "gpt" / "GPT_HMDB21.json"
DEFAULT_SIA_CONFIG = REPO_ROOT / "configs" / "runtime_offline_always_on.json"
DEFAULT_FULL_CONFIG = REPO_ROOT / "configs" / "runtime_offline_motion_person_sia.json"
DEFAULT_TRT_FP16_ENGINE = REPO_ROOT / "tmp" / "sia_vision_fp16.engine"
DEFAULT_TRT_INT8_ENGINE = REPO_ROOT / "results" / "tensorrt_vision" / "int8_check" / "sia_vision_int8.engine"

MODE_CONFIGS = {
    "runtime_sia": DEFAULT_SIA_CONFIG,
    "full_pipeline": DEFAULT_FULL_CONFIG,
}

SUMMARY_FIELDS = [
    "mode",
    "backend_name",
    "precision",
    "samples",
    "gt_instances",
    "pred_instances",
    "f_map_50",
    "runtime_frames_with_prediction",
    "runtime_missing_prediction_frames",
    "mean_effective_fps",
    "mean_effective_active_fps",
    "mean_inference_ms",
    "run_dir",
    "status",
    "error",
]

DIRECT_PREDICTION_FIELDS = [
    "mode",
    "backend_name",
    "precision",
    "video",
    "keyframe",
    "x1",
    "y1",
    "x2",
    "y2",
    "label",
    "score",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a JHMDB pilot accuracy sweep using f-mAP@0.5, the primary SiA paper metric."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="Repo-compatible JHMDB AVI root.")
    parser.add_argument("--anno", default=str(DEFAULT_ANNO), help="JHMDB-GT.pkl path.")
    parser.add_argument("--actions-json", default=str(DEFAULT_ACTIONS), help="Action prompt JSON for JHMDB labels.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["direct_sia", "runtime_sia", "full_pipeline"],
        default=["direct_sia"],
        help="Evaluation paths to run. Default: direct_sia.",
    )
    parser.add_argument("--sia-config", default=str(DEFAULT_SIA_CONFIG), help="Runtime config for runtime_sia.")
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG), help="Runtime config for full_pipeline.")
    parser.add_argument("--output-dir", help="Explicit output directory.")
    parser.add_argument("--output-root", default="results/jhmdb_runtime_accuracy", help="Output root.")
    parser.add_argument("--split-index", type=int, default=0, choices=[0, 1, 2], help="JHMDB split index.")
    parser.add_argument("--limit", type=int, default=5, help="Number of JHMDB test videos to run. Use 0 for full split.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset within the selected test split.")
    parser.add_argument(
        "--sample-strategy",
        choices=["round_robin_class", "first"],
        default="round_robin_class",
        help="How to select a limited pilot subset. Default balances action classes.",
    )
    parser.add_argument("--frames", type=int, default=9, help="SiA input frame count.")
    parser.add_argument("--height", type=int, default=240, help="Model input/eval height.")
    parser.add_argument("--width", type=int, default=320, help="Model input/eval width.")
    parser.add_argument("--threshold", type=float, default=0.25, help="SiA action threshold.")
    parser.add_argument(
        "--human-confidence-threshold",
        type=float,
        default=0.0,
        help="Human-box confidence threshold. 0.0 matches val_avak.py JHMDB scoring.",
    )
    parser.add_argument(
        "--top-k-labels",
        type=int,
        default=0,
        help="Labels to keep per box. 0 keeps all threshold-passing labels for metric scoring.",
    )
    parser.add_argument(
        "--buffer-max-len",
        type=int,
        default=9,
        help="Runtime buffer length for short JHMDB clips. Default keeps a 9-frame window.",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=4,
        help="Runtime metric can match the nearest actual SiA prediction center within this many frames.",
    )
    parser.add_argument("--weights", default="weights/avak_b16_11.pt", help="SiA weights path.")
    parser.add_argument("--backend-name", choices=["pytorch", "tensorrt"], help="Single backend override.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], help="Single precision override.")
    parser.add_argument("--trt-engine-path", help="Single TensorRT engine path override.")
    parser.add_argument("--backend-sweep", action="store_true", help="Run PyTorch FP32/FP16 and available TensorRT engines.")
    parser.add_argument("--pytorch-precisions", default="fp32,fp16", help="Comma-separated PyTorch precisions.")
    parser.add_argument("--trt-fp16-engine-path", default=str(DEFAULT_TRT_FP16_ENGINE), help="TensorRT FP16 engine.")
    parser.add_argument("--trt-int8-engine-path", default=str(DEFAULT_TRT_INT8_ENGINE), help="TensorRT INT8 engine.")
    parser.add_argument("--skip-missing-trt", action="store_true", help="Skip missing TensorRT engines in sweeps.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if one variant fails.")
    parser.add_argument("--skip-runs", action="store_true", help="Evaluate existing runtime outputs where present.")
    parser.add_argument(
        "--cold-runtime-per-video",
        action="store_true",
        help="Use the old runtime path that reloads model/backend state once per video.",
    )
    parser.add_argument("--no-render", action="store_true", default=True, help="Disable runtime video rendering.")
    return parser.parse_args()


def timestamp_slug():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_arg(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def backend_variants(args):
    if not args.backend_sweep:
        backend = args.backend_name or "pytorch"
        precision = args.precision or "fp32"
        return [
            {
                "backend_name": backend,
                "precision": precision,
                "trt_engine_path": args.trt_engine_path if backend == "tensorrt" else None,
            }
        ]

    variants = []
    pytorch_precisions = parse_csv_arg(args.pytorch_precisions)
    invalid = sorted(set(pytorch_precisions) - {"fp32", "fp16"})
    if invalid:
        raise RuntimeError(f"Unsupported PyTorch precision(s): {invalid}")
    for precision in pytorch_precisions:
        variants.append({"backend_name": "pytorch", "precision": precision, "trt_engine_path": None})

    for precision, path in [
        ("fp16", Path(args.trt_fp16_engine_path)),
        ("int8", Path(args.trt_int8_engine_path)),
    ]:
        if not path.is_file():
            if args.skip_missing_trt:
                print(f"Skipping TensorRT {precision}: missing engine at {path}")
                continue
            raise RuntimeError(f"Missing TensorRT {precision} engine at {path}")
        variants.append({"backend_name": "tensorrt", "precision": precision, "trt_engine_path": str(path)})
    return variants


def suite_output_dir(args):
    path = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{timestamp_slug()}_jhmdb_accuracy"
    ensure_dir(path)
    return path


def top_k_value(args):
    return None if args.top_k_labels == 0 else args.top_k_labels


def load_dataset(args):
    transforms = v2.Compose(
        [
            v2.Resize((args.height, args.width)),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = HMDB21(
        args.video_dir,
        args.anno,
        transforms=transforms,
        imgsize=(args.height, args.width),
        frames=args.frames,
        split="test",
    )
    split_videos = dataset.anno["test_videos"][args.split_index]
    selected = select_videos(split_videos, args.sample_strategy, args.start_index, args.limit)
    dataset.video_list = selected
    rebuild_jhmdb_bbox_index(dataset, args.video_dir)
    return dataset


def select_videos(video_list, strategy, start_index, limit):
    candidates = list(video_list)[start_index:]
    if limit == 0:
        return candidates
    if strategy == "first":
        return candidates[:limit]
    if strategy != "round_robin_class":
        raise RuntimeError(f"Unsupported sample strategy: {strategy}")

    by_class = {}
    class_order = []
    for video in candidates:
        cls = video.split("/")[0]
        if cls not in by_class:
            by_class[cls] = []
            class_order.append(cls)
        by_class[cls].append(video)

    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for cls in class_order:
            videos = by_class[cls]
            if offset < len(videos):
                selected.append(videos[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def rebuild_jhmdb_bbox_index(dataset, video_dir):
    dataset.bboxes = {}
    for video in dataset.video_list:
        gttubes = dataset.gttubes[video]
        for action in gttubes:
            for tube in gttubes[action]:
                for bbox in tube:
                    img_id, x1, y1, x2, y2 = bbox
                    img_id, x1, y1, x2, y2 = int(img_id), int(x1), int(y1), int(x2), int(y2)
                    file_dir = str(Path(video_dir) / video)
                    dataset.bboxes.setdefault(file_dir, {}).setdefault(img_id, []).append(
                        torch.tensor([x1, y1, x2, y2])
                    )


def labels_from_actions(actions_json):
    with Path(actions_json).open("r", encoding="utf-8") as handle:
        return list(json.load(handle).keys())


def empty_prediction(device="cpu"):
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
        "labels": torch.empty((0,), dtype=torch.int64, device=device),
    }


def target_to_metric_target(target, caption_to_id, width, height, device="cpu"):
    raw_boxes = box_cxcywh_to_xyxy(target["boxes"].to(device))
    if raw_boxes.numel() == 0:
        boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
        labels = torch.empty((0,), dtype=torch.int64, device=device)
        return {"boxes": boxes, "labels": labels}
    boxes = raw_boxes.float().clone()
    boxes[:, 0::2] *= width
    boxes[:, 1::2] *= height
    labels = []
    expanded_boxes = []
    for box, label_group in zip(boxes, target["text_labels"]):
        valid_labels = [caption_to_id[label] for label in label_group if label in caption_to_id]
        for label_id in valid_labels:
            expanded_boxes.append(box)
            labels.append(label_id)
    if not expanded_boxes:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "labels": torch.empty((0,), dtype=torch.int64, device=device),
        }
    return {
        "boxes": torch.stack(expanded_boxes).float(),
        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
    }


def decoded_predictions_to_metric(result, caption_to_id, device="cpu"):
    boxes = []
    labels = []
    scores = []
    for box, label_group, score_group in zip(result.get("boxes", []), result.get("labels", []), result.get("scores", [])):
        box_tensor = box.detach().to(device).float() if torch.is_tensor(box) else torch.tensor(box, dtype=torch.float32)
        for label, score in zip(label_group, score_group):
            if label not in caption_to_id:
                continue
            boxes.append(box_tensor)
            labels.append(caption_to_id[label])
            scores.append(float(score))
    if not boxes:
        return empty_prediction(device)
    return {
        "boxes": torch.stack(boxes).float(),
        "scores": torch.tensor(scores, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
    }


def metric_value(metric):
    result = metric.compute()
    value = result["map_50"]
    return float(value.detach().cpu().item() if hasattr(value, "detach") else value)


def base_runtime_config(args, variant, pipeline_mode):
    return {
        "mode": "offline",
        "pipeline_mode": pipeline_mode,
        "video_path": "unused.avi",
        "weights_path": args.weights,
        "actions_json": args.actions_json,
        "output_root": str(Path(args.output_root)),
        "output_video_name": "runtime_offline.mp4",
        "device": "cuda",
        "precision": variant["precision"],
        "autocast": variant["precision"] == "fp16",
        "backend_name": variant["backend_name"],
        "trt_engine_path": variant["trt_engine_path"],
        "optimization_label": f"jhmdb_{variant['backend_name']}_{variant['precision']}",
        "model_size": "b",
        "pretrain": None,
        "det_token_num": 20,
        "text_lora": True,
        "num_frames": args.frames,
        "buffer_max_len": args.buffer_max_len,
        "img_size": [args.height, args.width],
        "threshold": args.threshold,
        "human_confidence_threshold": args.human_confidence_threshold,
        "top_k_labels": top_k_value(args),
        "render_enabled": False,
        "sync_cuda_timing": True,
        "sia_target_fps": 0,
        "sia_min_new_frames": args.frames,
    }


def run_direct_sia(args, suite_dir, dataset, variant, captions):
    caption_to_id = {label: idx for idx, label in enumerate(captions)}
    raw_config = base_runtime_config(args, variant, "always_on")
    config = RuntimeConfig.from_dict(raw_config)
    core = SIARuntimeCore(config)
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", iou_thresholds=[0.5], backend="faster_coco_eval")
    pred_instances = 0
    gt_instances = 0
    prediction_rows = []
    start = time.perf_counter()

    for index in range(len(dataset)):
        clip, target = dataset[index]
        clip = clip.unsqueeze(0).to(core.device)
        if core.input_use_fp16:
            clip = clip.half()
        result = core.infer_clip(clip, frame_size=(args.height, args.width))
        pred = decoded_predictions_to_metric(result, caption_to_id)
        tgt = target_to_metric_target(target, caption_to_id, args.width, args.height)
        metric.update([pred], [tgt])
        pred_instances += int(pred["labels"].numel())
        gt_instances += int(tgt["labels"].numel())
        for box, label_id, score in zip(pred["boxes"].cpu().tolist(), pred["labels"].cpu().tolist(), pred["scores"].cpu().tolist()):
            prediction_rows.append(
                {
                    "mode": "direct_sia",
                    "backend_name": variant["backend_name"],
                    "precision": variant["precision"],
                    "video": target["video"],
                    "keyframe": target["keyframe"],
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "label": captions[label_id],
                    "score": score,
                }
            )

    run_dir = suite_dir / "runs" / "direct_sia" / f"{variant['backend_name']}_{variant['precision']}"
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", raw_config)
    write_csv(run_dir / "predictions.csv", prediction_rows, DIRECT_PREDICTION_FIELDS)
    return {
        "mode": "direct_sia",
        "backend_name": variant["backend_name"],
        "precision": variant["precision"],
        "samples": len(dataset),
        "gt_instances": gt_instances,
        "pred_instances": pred_instances,
        "f_map_50": round(metric_value(metric), 6),
        "runtime_frames_with_prediction": "",
        "runtime_missing_prediction_frames": "",
        "mean_effective_fps": round(len(dataset) / max(time.perf_counter() - start, 1e-9), 3),
        "mean_effective_active_fps": "",
        "mean_inference_ms": "",
        "run_dir": str(run_dir),
        "status": "ok",
        "error": "",
    }


def runtime_config_for_mode(args, mode, variant, video_path):
    config_path = Path(args.sia_config if mode == "runtime_sia" else args.full_config)
    raw = load_json(config_path)
    raw.update(base_runtime_config(args, variant, "always_on" if mode == "runtime_sia" else "motion_person_sia"))
    raw["video_path"] = str(video_path)
    raw["output_root"] = str(Path(args.output_root))
    return raw


def runtime_predictions_by_center(run_dir):
    grouped = {}
    for row in read_csv(Path(run_dir) / "predictions.csv"):
        center = int(row["center_frame_index"])
        grouped.setdefault(center, []).append(row)
    return grouped


def rows_to_metric_prediction(rows, caption_to_id):
    boxes = []
    labels = []
    scores = []
    for row in rows:
        label = row["predicted_label"]
        if label not in caption_to_id:
            continue
        boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
        labels.append(caption_to_id[label])
        scores.append(float(row["score"]))
    if not boxes:
        return empty_prediction()
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def nearest_prediction_rows(predictions_by_center, target_center, tolerance):
    if target_center in predictions_by_center:
        return predictions_by_center[target_center]
    best_center = None
    best_distance = None
    for center in predictions_by_center:
        distance = abs(center - target_center)
        if distance <= tolerance and (best_distance is None or distance < best_distance):
            best_center = center
            best_distance = distance
    return predictions_by_center.get(best_center, [])


def summarize_runtime_metrics(run_dirs):
    fps = []
    active_fps = []
    inference_ms = []
    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = load_json(metrics_path)
        if metrics.get("effective_fps") is not None:
            fps.append(float(metrics["effective_fps"]))
        if metrics.get("effective_active_fps") is not None:
            active_fps.append(float(metrics["effective_active_fps"]))
        timings = metrics.get("timings", {})
        timing = timings.get("inference") or timings.get("inference_s", {})
        if timing.get("mean_ms") is not None:
            inference_ms.append(float(timing["mean_ms"]))
        elif timing.get("mean") is not None:
            inference_ms.append(float(timing["mean"]) * 1000.0)
    mean = lambda values: round(sum(values) / len(values), 3) if values else ""
    return mean(fps), mean(active_fps), mean(inference_ms)


def mean_or_blank(values):
    return round(sum(values) / len(values), 3) if values else ""


def process_video_warm(pipeline, config, video_path, run_dir):
    ensure_dir(run_dir)
    pipeline.reset_sequence_state()
    config.video_path = str(video_path)
    pipeline.config.video_path = str(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    active_frames = 0
    output_ready_frames = 0
    prediction_rows = []
    inference_rows = []
    inference_ms = []
    active_loop_ms = []
    start = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = capture.read()
            if not ret:
                break
            frame_count += 1
            result = pipeline.process_frame(frame, (frame_height, frame_width))
            if result["output_ready"]:
                output_ready_frames += 1
            if result["active"]:
                active_frames += 1
                frame_meta = _center_frame_metadata(frame_count, pipeline.buffer)
                inference_row, active_prediction_rows = _prediction_artifact_rows(config, result, frame_meta)
                inference_rows.append(inference_row)
                prediction_rows.extend(active_prediction_rows)
                inference_ms.append(float(result["timings"].get("inference_s", 0.0)) * 1000.0)
                active_loop_ms.append((time.perf_counter() - loop_start) * 1000.0)
    finally:
        capture.release()

    elapsed_s = time.perf_counter() - start
    active_loop_total_s = sum(value / 1000.0 for value in active_loop_ms)
    metrics = {
        "video_path": str(video_path),
        "backend_name": config.backend_name,
        "precision": config.precision,
        "pipeline_mode": config.pipeline_mode,
        "frames_read": frame_count,
        "source_frames": source_frame_count,
        "active_frames": active_frames,
        "output_ready_frames": output_ready_frames,
        "prediction_rows": len(prediction_rows),
        "sia_inference_frame_rows": len(inference_rows),
        "sia_inference_iterations": pipeline.sia_inference_count,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else None,
        "effective_active_fps": round(active_frames / active_loop_total_s, 3) if active_loop_total_s > 0 else None,
        "timings": {
            "inference": {"mean_ms": mean_or_blank(inference_ms), "count": len(inference_ms)},
            "active_loop": {"mean_ms": mean_or_blank(active_loop_ms), "count": len(active_loop_ms)},
        },
    }
    write_csv(run_dir / "sia_inference_frames.csv", inference_rows, SIA_INFERENCE_FRAME_FIELDNAMES)
    write_csv(run_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDNAMES)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def run_runtime_mode_warm(args, suite_dir, dataset, mode, variant, captions):
    caption_to_id = {label: idx for idx, label in enumerate(captions)}
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", iou_thresholds=[0.5], backend="faster_coco_eval")
    pred_instances = 0
    gt_instances = 0
    matched_frames = 0
    missing_frames = 0
    run_dirs = []

    raw_config = runtime_config_for_mode(args, mode, variant, "unused.avi")
    raw_config["video_path"] = "unused.avi"
    config = RuntimeConfig.from_dict(raw_config)
    pipeline = AlwaysOnSIAPipeline(config)

    for index in range(len(dataset)):
        _, target = dataset[index]
        video_stem = Path(target["video"]).name
        video_path = Path(str(target["video"]) + ".avi")
        run_dir = suite_dir / "runs" / mode / f"{variant['backend_name']}_{variant['precision']}" / f"{index + 1:03d}_{video_stem}"
        run_dirs.append(run_dir)
        if not args.skip_runs:
            print(
                f"  {mode} {variant['backend_name']}_{variant['precision']} "
                f"[{index + 1}/{len(dataset)}] warm: {video_path}"
            )
            process_video_warm(pipeline, config, video_path, run_dir)
        predictions_by_center = runtime_predictions_by_center(run_dir)
        target_center = int(target["keyframe"]) - 1
        pred_rows = nearest_prediction_rows(predictions_by_center, target_center, args.frame_tolerance)
        if pred_rows:
            matched_frames += 1
        else:
            missing_frames += 1
        pred = rows_to_metric_prediction(pred_rows, caption_to_id)
        tgt = target_to_metric_target(target, caption_to_id, args.width, args.height)
        metric.update([pred], [tgt])
        pred_instances += int(pred["labels"].numel())
        gt_instances += int(tgt["labels"].numel())

    mean_fps, mean_active_fps, mean_inference_ms = summarize_runtime_metrics(run_dirs)
    return {
        "mode": mode,
        "backend_name": variant["backend_name"],
        "precision": variant["precision"],
        "samples": len(dataset),
        "gt_instances": gt_instances,
        "pred_instances": pred_instances,
        "f_map_50": round(metric_value(metric), 6),
        "runtime_frames_with_prediction": matched_frames,
        "runtime_missing_prediction_frames": missing_frames,
        "mean_effective_fps": mean_fps,
        "mean_effective_active_fps": mean_active_fps,
        "mean_inference_ms": mean_inference_ms,
        "run_dir": str(suite_dir / "runs" / mode / f"{variant['backend_name']}_{variant['precision']}"),
        "status": "ok",
        "error": "",
    }


def run_runtime_mode(args, suite_dir, dataset, mode, variant, captions, invoked_command):
    if not args.cold_runtime_per_video:
        return run_runtime_mode_warm(args, suite_dir, dataset, mode, variant, captions)

    caption_to_id = {label: idx for idx, label in enumerate(captions)}
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", iou_thresholds=[0.5], backend="faster_coco_eval")
    pred_instances = 0
    gt_instances = 0
    matched_frames = 0
    missing_frames = 0
    run_dirs = []

    for index in range(len(dataset)):
        _, target = dataset[index]
        video_stem = Path(target["video"]).name
        video_path = Path(str(target["video"]) + ".avi")
        run_dir = suite_dir / "runs" / mode / f"{variant['backend_name']}_{variant['precision']}" / f"{index + 1:03d}_{video_stem}"
        run_dirs.append(run_dir)
        if not args.skip_runs:
            print(f"  {mode} {variant['backend_name']}_{variant['precision']} [{index + 1}/{len(dataset)}]: {video_path}")
            raw_config = runtime_config_for_mode(args, mode, variant, video_path)
            run_offline_runtime(raw_config, invoked_command, run_name=video_stem, run_dir=run_dir)
        predictions_by_center = runtime_predictions_by_center(run_dir)
        target_center = int(target["keyframe"]) - 1
        pred_rows = nearest_prediction_rows(predictions_by_center, target_center, args.frame_tolerance)
        if pred_rows:
            matched_frames += 1
        else:
            missing_frames += 1
        pred = rows_to_metric_prediction(pred_rows, caption_to_id)
        tgt = target_to_metric_target(target, caption_to_id, args.width, args.height)
        metric.update([pred], [tgt])
        pred_instances += int(pred["labels"].numel())
        gt_instances += int(tgt["labels"].numel())

    mean_fps, mean_active_fps, mean_inference_ms = summarize_runtime_metrics(run_dirs)
    return {
        "mode": mode,
        "backend_name": variant["backend_name"],
        "precision": variant["precision"],
        "samples": len(dataset),
        "gt_instances": gt_instances,
        "pred_instances": pred_instances,
        "f_map_50": round(metric_value(metric), 6),
        "runtime_frames_with_prediction": matched_frames,
        "runtime_missing_prediction_frames": missing_frames,
        "mean_effective_fps": mean_fps,
        "mean_effective_active_fps": mean_active_fps,
        "mean_inference_ms": mean_inference_ms,
        "run_dir": str(suite_dir / "runs" / mode / f"{variant['backend_name']}_{variant['precision']}"),
        "status": "ok",
        "error": "",
    }


def main():
    args = parse_args()
    suite_dir = suite_output_dir(args)
    captions = labels_from_actions(args.actions_json)
    dataset = load_dataset(args)
    variants = backend_variants(args)
    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    summaries = []
    failures = []

    print(f"JHMDB samples: {len(dataset)}")
    print(f"Output: {suite_dir}")
    for mode in args.modes:
        for variant in variants:
            try:
                if mode == "direct_sia":
                    print(f"Running direct_sia {variant['backend_name']}_{variant['precision']}")
                    row = run_direct_sia(args, suite_dir, dataset, variant, captions)
                else:
                    row = run_runtime_mode(args, suite_dir, dataset, mode, variant, captions, invoked_command)
                summaries.append(row)
                print(
                    f"  {mode} {variant['backend_name']}_{variant['precision']}: "
                    f"f-mAP@0.5={row['f_map_50']}, preds={row['pred_instances']}, gt={row['gt_instances']}"
                )
            except Exception as exc:
                failure = {
                    "mode": mode,
                    "backend_name": variant["backend_name"],
                    "precision": variant["precision"],
                    "samples": len(dataset),
                    "gt_instances": "",
                    "pred_instances": "",
                    "f_map_50": "",
                    "runtime_frames_with_prediction": "",
                    "runtime_missing_prediction_frames": "",
                    "mean_effective_fps": "",
                    "mean_effective_active_fps": "",
                    "mean_inference_ms": "",
                    "run_dir": "",
                    "status": "error",
                    "error": str(exc),
                }
                summaries.append(failure)
                failures.append(failure)
                print(f"  ERROR {mode} {variant['backend_name']}_{variant['precision']}: {exc}")
                if not args.continue_on_error:
                    break

    write_csv(suite_dir / "jhmdb_accuracy_summary.csv", summaries, SUMMARY_FIELDS)
    write_json(
        suite_dir / "jhmdb_accuracy_summary.json",
        {
            "primary_metric": "f-mAP@0.5",
            "samples": len(dataset),
            "modes": args.modes,
            "variants": variants,
            "summary": to_builtin(summaries),
            "failures": to_builtin(failures),
        },
    )
    print(f"Summary CSV: {suite_dir / 'jhmdb_accuracy_summary.csv'}")
    print(f"Summary JSON: {suite_dir / 'jhmdb_accuracy_summary.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
