import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import json
from pathlib import Path
import shlex
import sys
import time
from datetime import datetime

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]


DEFAULT_VIDEO_DIR = REPO_ROOT / "meva-videos"
DEFAULT_ANNOTATION_ROOT = (
    REPO_ROOT.parent / "meva-data-repo" / "annotation" / "DIVA-phase-2" / "MEVA" / "kitware-meva-training"
)
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "meva_pilot_manifest.yml"
DEFAULT_LABEL_MAP = REPO_ROOT / "gpt" / "meva_to_sia_eval_map.yml"
DEFAULT_SIA_CONFIG = REPO_ROOT / "configs" / "runtime_offline_always_on.json"
DEFAULT_FULL_CONFIG = REPO_ROOT / "configs" / "runtime_offline_motion_person_sia.json"
DEFAULT_TRT_FP16_ENGINE = REPO_ROOT / "tmp" / "sia_vision_fp16.engine"
DEFAULT_TRT_INT8_ENGINE = REPO_ROOT / "results" / "tensorrt_vision" / "int8_check" / "sia_vision_int8.engine"

MODE_CONFIGS = {
    "sia_only": DEFAULT_SIA_CONFIG,
    "full_pipeline": DEFAULT_FULL_CONFIG,
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_structured(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_slug():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_name(value):
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "video"


def discover_videos(video_dir, pattern, recursive):
    root = Path(video_dir)
    if not root.is_dir():
        raise RuntimeError(f"Video directory '{video_dir}' does not exist or is not a directory.")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS:
            paths.append(path)
    return sorted(paths)

SUMMARY_FIELDNAMES = [
    "mode",
    "backend_name",
    "precision",
    "optimization_label",
    "trt_engine_path",
    "video_path",
    "annotation_clip",
    "run_dir",
    "frames_evaluated",
    "gt_instances",
    "pred_instances",
    "f_map_50",
    "sia_inference_iterations",
    "active_frames",
    "frames_read",
    "effective_fps",
    "effective_active_fps",
    "inference_mean_ms",
    "active_loop_mean_ms",
    "status",
    "error",
]

AGGREGATE_FIELDNAMES = [
    "mode",
    "backend_name",
    "precision",
    "frames_evaluated",
    "gt_instances",
    "pred_instances",
    "f_map_50",
    "sia_inference_iterations",
    "mean_effective_fps",
    "mean_effective_active_fps",
    "mean_inference_ms",
    "mean_active_loop_ms",
    "class_ap_50",
]

PREDICTION_EXPORT_FIELDNAMES = [
    "mode",
    "backend_name",
    "precision",
    "run_id",
    "video_path",
    "annotation_clip",
    "center_frame_index",
    "source_frame_index",
    "x1",
    "y1",
    "x2",
    "y2",
    "predicted_label",
    "score",
]

GT_EXPORT_FIELDNAMES = [
    "mode",
    "backend_name",
    "precision",
    "run_id",
    "video_path",
    "annotation_clip",
    "center_frame_index",
    "source_frame_index",
    "actor_id",
    "meva_label",
    "sia_label",
    "x1",
    "y1",
    "x2",
    "y2",
]

FIDELITY_FIELDNAMES = [
    "mode",
    "video_path",
    "reference_backend_name",
    "reference_precision",
    "candidate_backend_name",
    "candidate_precision",
    "candidate_run_dir",
    "common_frames",
    "reference_frames",
    "candidate_frames",
    "reference_pred_rows",
    "candidate_pred_rows",
    "label_jaccard_mean",
    "box_iou50_precision",
    "box_iou50_recall",
    "box_iou50_f1",
    "same_label_iou50_precision",
    "same_label_iou50_recall",
    "same_label_iou50_f1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the MEVA pilot videos through SiA-only and/or the full three-stage runtime, "
            "then score frame-level f-mAP@0.5 against MEVA annotations."
        )
    )
    parser.add_argument("--video", action="append", help="Run one specific video. May be provided more than once.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="Directory of MEVA pilot videos.")
    parser.add_argument("--glob", default="*", help="Filename pattern inside --video-dir. Default: '*'")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under --video-dir.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_CONFIGS),
        default=["sia_only", "full_pipeline"],
        help="Runtime modes to execute. Default: sia_only full_pipeline",
    )
    parser.add_argument("--sia-config", default=str(DEFAULT_SIA_CONFIG), help="Config for --modes sia_only.")
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG), help="Config for --modes full_pipeline.")
    parser.add_argument("--annotation-root", default=str(DEFAULT_ANNOTATION_ROOT), help="MEVA annotation root.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Clip-to-annotation manifest YAML/JSON.")
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP), help="MEVA-to-SiA label map YAML/JSON.")
    parser.add_argument("--output-dir", help="Explicit suite output directory.")
    parser.add_argument("--output-root", default="results/meva_pilot", help="Suite output root when --output-dir is not set.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running if one video/mode fails.")
    parser.add_argument("--progress-every", type=int, default=60, help="Print runtime progress every N frames.")
    parser.add_argument("--skip-runs", action="store_true", help="Skip runtime execution and evaluate existing run dirs.")

    parser.add_argument("--weights", help="Runtime weights override.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], help="Runtime precision override.")
    parser.add_argument("--backend-name", choices=["pytorch", "tensorrt"], help="Runtime backend override.")
    parser.add_argument("--trt-engine-path", help="Runtime TensorRT engine override.")
    parser.add_argument(
        "--backend-sweep",
        action="store_true",
        help="Run PyTorch FP32/FP16 plus available TensorRT FP16/INT8 variants.",
    )
    parser.add_argument(
        "--pytorch-precisions",
        default="fp32,fp16",
        help="Comma-separated PyTorch precisions for --backend-sweep. Default: fp32,fp16",
    )
    parser.add_argument(
        "--trt-fp16-engine-path",
        default=str(DEFAULT_TRT_FP16_ENGINE),
        help="TensorRT FP16 engine for --backend-sweep.",
    )
    parser.add_argument(
        "--trt-int8-engine-path",
        default=str(DEFAULT_TRT_INT8_ENGINE),
        help="TensorRT INT8 engine for --backend-sweep.",
    )
    parser.add_argument(
        "--skip-missing-trt",
        action="store_true",
        help="In --backend-sweep, skip missing TensorRT engines instead of failing.",
    )
    parser.add_argument("--sia-target-fps", type=float, help="Runtime SiA activation FPS cap. Use 0 to disable.")
    parser.add_argument("--adaptive-sia-target-fps", action="store_true", help="Enable runtime adaptive SiA FPS cap.")
    parser.add_argument("--adaptive-sia-warmup-frames", type=int, help="Runtime adaptive SiA warmup frames.")
    parser.add_argument("--adaptive-sia-utilization", type=float, help="Runtime adaptive SiA utilization.")
    parser.add_argument("--adaptive-sia-smoothing", type=float, help="Runtime adaptive SiA smoothing.")
    parser.add_argument("--adaptive-sia-min-fps", type=float, help="Runtime adaptive SiA minimum FPS.")
    parser.add_argument("--adaptive-sia-max-fps", type=float, help="Runtime adaptive SiA maximum FPS.")
    parser.add_argument("--motion-min-on-time", type=int, help="Runtime motion gate min-on frames.")
    parser.add_argument("--person-min-on-time", type=int, help="Runtime person gate min-on frames.")
    parser.add_argument("--threshold", type=float, help="Runtime SiA postprocess action threshold.")
    parser.add_argument("--top-k-labels", type=int, help="Runtime decoded action labels per detected box.")
    parser.add_argument("--max-frames", type=int, help="Runtime cap on frames read from each source video.")
    parser.add_argument("--no-render", action="store_true", help="Disable runtime output video writing.")
    parser.add_argument("--show-active-tiers", action="store_true", help="Overlay active-tier indicators if rendering.")
    return parser.parse_args()


def resolve_video_paths(args):
    if args.video:
        paths = [Path(video) for video in args.video]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Video path(s) do not exist: {', '.join(missing)}")
        return paths

    videos = discover_videos(args.video_dir, args.glob, args.recursive)
    if not videos:
        raise RuntimeError(f"No supported videos found under '{args.video_dir}' with pattern '{args.glob}'.")
    return videos


def suite_output_dir(args):
    if args.output_dir:
        path = Path(args.output_dir)
    else:
        path = Path(args.output_root) / f"{timestamp_slug()}_meva_pilot"
    ensure_dir(path)
    return path


def mode_config_path(args, mode):
    if mode == "sia_only":
        return Path(args.sia_config)
    if mode == "full_pipeline":
        return Path(args.full_config)
    raise RuntimeError(f"Unsupported mode: {mode}")


def parse_csv_arg(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def backend_variants(args):
    if not args.backend_sweep:
        backend_name = args.backend_name or "pytorch"
        precision = args.precision or "fp32"
        return [
            {
                "backend_name": backend_name,
                "precision": precision,
                "trt_engine_path": args.trt_engine_path if backend_name == "tensorrt" else None,
                "optimization_label": f"meva_{backend_name}_{precision}",
            }
        ]

    variants = []
    pytorch_precisions = parse_csv_arg(args.pytorch_precisions)
    invalid = sorted(set(pytorch_precisions) - {"fp32", "fp16"})
    if invalid:
        raise RuntimeError(f"Unsupported PyTorch precision(s) for --backend-sweep: {invalid}")
    for precision in pytorch_precisions:
        variants.append(
            {
                "backend_name": "pytorch",
                "precision": precision,
                "trt_engine_path": None,
                "optimization_label": f"meva_pytorch_{precision}",
            }
        )

    trt_specs = [
        ("fp16", Path(args.trt_fp16_engine_path)),
        ("int8", Path(args.trt_int8_engine_path)),
    ]
    for precision, engine_path in trt_specs:
        if not engine_path.is_file():
            if args.skip_missing_trt:
                print(f"Skipping TensorRT {precision}: engine not found at {engine_path}")
                continue
            raise RuntimeError(
                f"TensorRT {precision} engine not found at {engine_path}. "
                "Pass --skip-missing-trt to omit it."
            )
        variants.append(
            {
                "backend_name": "tensorrt",
                "precision": precision,
                "trt_engine_path": str(engine_path),
                "optimization_label": f"meva_tensorrt_{precision}",
            }
        )
    return variants


def apply_runtime_overrides(raw_config, args, video_path, variant):
    config = dict(raw_config)
    config["video_path"] = str(video_path)
    config["backend_name"] = variant["backend_name"]
    config["precision"] = variant["precision"]
    config["autocast"] = variant["precision"] == "fp16"
    config["optimization_label"] = variant["optimization_label"]
    if variant["backend_name"] == "tensorrt":
        config["trt_engine_path"] = variant["trt_engine_path"]
    if args.weights:
        config["weights_path"] = args.weights
    if args.sia_target_fps is not None:
        config["sia_target_fps"] = args.sia_target_fps
    if args.adaptive_sia_target_fps:
        config["adaptive_sia_target_fps"] = True
    if args.adaptive_sia_warmup_frames is not None:
        config["adaptive_sia_warmup_frames"] = args.adaptive_sia_warmup_frames
    if args.adaptive_sia_utilization is not None:
        config["adaptive_sia_utilization"] = args.adaptive_sia_utilization
    if args.adaptive_sia_smoothing is not None:
        config["adaptive_sia_smoothing"] = args.adaptive_sia_smoothing
    if args.adaptive_sia_min_fps is not None:
        config["adaptive_sia_min_fps"] = args.adaptive_sia_min_fps
    if args.adaptive_sia_max_fps is not None:
        config["adaptive_sia_max_fps"] = args.adaptive_sia_max_fps
    if args.motion_min_on_time is not None:
        config["motion_min_on_time"] = args.motion_min_on_time
    if args.person_min_on_time is not None:
        config["person_min_on_time"] = args.person_min_on_time
    if args.threshold is not None:
        config["threshold"] = args.threshold
    if args.top_k_labels is not None:
        config["top_k_labels"] = args.top_k_labels
    if args.max_frames is not None:
        config["max_frames"] = args.max_frames
    if args.no_render:
        config["render_enabled"] = False
    if args.show_active_tiers:
        config["show_active_tiers"] = True
    return config


def progress_label(frame_index, total_frames, max_frames):
    if max_frames:
        effective_total = min(total_frames, max_frames) if total_frames else max_frames
    else:
        effective_total = total_frames
    if effective_total:
        return f"{frame_index}/{effective_total}"
    return str(frame_index)


def make_progress_callback(args):
    last_progress_time = 0.0

    def progress_callback(payload):
        nonlocal last_progress_time
        event = payload["event"]
        if event == "start":
            total_hint = payload["frame_count_hint"]
            max_frames = payload["max_frames"]
            total_msg = f", source frames={total_hint}" if total_hint else ""
            max_msg = f", max_frames={max_frames}" if max_frames else ""
            print(f"    Run directory: {payload['run_dir']}{total_msg}{max_msg}")
            return
        if event == "frame":
            frame_index = payload["frame_index"]
            should_print = frame_index == 1 or frame_index % max(1, args.progress_every) == 0
            now = time.perf_counter()
            if not should_print and now - last_progress_time < 10.0:
                return
            last_progress_time = now
            label = progress_label(frame_index, payload["frame_count_hint"], payload["max_frames"])
            print(
                "    Progress: "
                f"frame {label}, active_frames={payload['active_frames']}, "
                f"clips_processed={payload['clips_processed']}"
            )
            return
        if event == "complete":
            print(
                "    Completed: "
                f"frames_read={payload['frames_read']}, active_frames={payload['active_frames']}, "
                f"clips_processed={payload['clips_processed']}, effective_fps={payload['effective_fps']}"
            )

    return progress_callback


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_yaml_list(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    if isinstance(data, list):
        return data
    return [data]


def strip_r13_suffix(name):
    if name.endswith(".r13.avi"):
        return name[: -len(".r13.avi")]
    if name.endswith(".r13.mp4"):
        return name[: -len(".r13.mp4")]
    return Path(name).stem


def infer_clip_mapping(video_path, manifest):
    video_path = Path(video_path)
    entry = manifest.get(video_path.name)
    if entry:
        return {
            "annotation_clip": entry["annotation_clip"],
            "source_start_frame": int(entry.get("source_start_frame", 0)),
        }
    return {
        "annotation_clip": strip_r13_suffix(video_path.name),
        "source_start_frame": 0,
    }


def annotation_paths(annotation_root, annotation_clip):
    parts = annotation_clip.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Cannot infer MEVA annotation directory for clip '{annotation_clip}'.")
    date = parts[0]
    hour = parts[1].split("-")[0]
    base = Path(annotation_root) / date / hour / annotation_clip
    return {
        "activities": Path(f"{base}.activities.yml"),
        "geom": Path(f"{base}.geom.yml"),
        "types": Path(f"{base}.types.yml"),
    }


def spans_contain(spans, frame_index):
    for start, end in spans:
        if int(start) <= frame_index <= int(end):
            return True
    return False


def parse_spans(span_entries):
    spans = []
    for entry in span_entries or []:
        if not isinstance(entry, dict):
            continue
        span = entry.get("tsr0")
        if span and len(span) == 2:
            spans.append((int(span[0]), int(span[1])))
    return spans


def parse_box(value):
    coords = [float(part) for part in str(value).split()]
    if len(coords) != 4:
        return None
    return coords


class MevaClipAnnotations:
    def __init__(self, annotation_root, annotation_clip, label_map, nearest_tolerance=15):
        self.annotation_root = Path(annotation_root)
        self.annotation_clip = annotation_clip
        self.label_map = label_map
        self.nearest_tolerance = int(nearest_tolerance)
        self.person_actor_ids = set()
        self.geom_by_actor = defaultdict(dict)
        self.sorted_geom_frames = {}
        self.activities = []
        self._load()

    def _load(self):
        paths = annotation_paths(self.annotation_root, self.annotation_clip)
        for item in load_yaml_list(paths["types"]):
            type_payload = item.get("types", {}) if isinstance(item, dict) else {}
            cset = type_payload.get("cset3", {}) or {}
            if "person" in cset:
                self.person_actor_ids.add(int(type_payload["id1"]))

        for item in load_yaml_list(paths["geom"]):
            geom = item.get("geom", {}) if isinstance(item, dict) else {}
            if "id1" not in geom or "ts0" not in geom or "g0" not in geom:
                continue
            actor_id = int(geom["id1"])
            box = parse_box(geom["g0"])
            if box is None:
                continue
            self.geom_by_actor[actor_id][int(geom["ts0"])] = box
        self.sorted_geom_frames = {
            actor_id: sorted(frames)
            for actor_id, frames in self.geom_by_actor.items()
        }

        for item in load_yaml_list(paths["activities"]):
            act = item.get("act", {}) if isinstance(item, dict) else {}
            label_scores = act.get("act2", {}) or {}
            meva_labels = [label for label in label_scores if label in self.label_map]
            if not meva_labels:
                continue
            activity_spans = parse_spans(act.get("timespan"))
            for actor in act.get("actors", []) or []:
                actor_id = int(actor.get("id1"))
                if actor_id not in self.person_actor_ids:
                    continue
                actor_spans = parse_spans(actor.get("timespan")) or activity_spans
                for meva_label in meva_labels:
                    self.activities.append(
                        {
                            "actor_id": actor_id,
                            "meva_label": meva_label,
                            "sia_label": self.label_map[meva_label],
                            "spans": actor_spans,
                        }
                    )

    def _nearest_box(self, actor_id, frame_index, spans):
        boxes = self.geom_by_actor.get(actor_id, {})
        if frame_index in boxes:
            return boxes[frame_index]
        frames = self.sorted_geom_frames.get(actor_id, [])
        if not frames:
            return None
        insert_at = bisect_left(frames, frame_index)
        candidates = []
        if insert_at < len(frames):
            candidates.append(frames[insert_at])
        if insert_at > 0:
            candidates.append(frames[insert_at - 1])
        valid = [
            candidate
            for candidate in candidates
            if abs(candidate - frame_index) <= self.nearest_tolerance and spans_contain(spans, candidate)
        ]
        if not valid:
            return None
        nearest = min(valid, key=lambda candidate: abs(candidate - frame_index))
        return boxes[nearest]

    def ground_truth_for_frame(self, frame_index):
        rows = []
        seen = set()
        for activity in self.activities:
            if not spans_contain(activity["spans"], frame_index):
                continue
            box = self._nearest_box(activity["actor_id"], frame_index, activity["spans"])
            if box is None:
                continue
            key = (
                activity["actor_id"],
                activity["meva_label"],
                tuple(round(value, 3) for value in box),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "actor_id": activity["actor_id"],
                    "meva_label": activity["meva_label"],
                    "sia_label": activity["sia_label"],
                    "box": box,
                }
            )
        return rows


def iou_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def f1_score(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def average_precision(recalls, precisions):
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def compute_f_map_50(frame_ids, gt_rows, pred_rows):
    labels_with_gt = sorted({row["sia_label"] for row in gt_rows})
    if not labels_with_gt:
        return {
            "f_map_50": None,
            "class_ap_50": {},
        }

    gt_by_label_image = defaultdict(lambda: defaultdict(list))
    for row in gt_rows:
        gt_by_label_image[row["sia_label"]][row["image_id"]].append(
            {
                "box": row["box"],
                "matched": False,
            }
        )

    preds_by_label = defaultdict(list)
    for row in pred_rows:
        preds_by_label[row["predicted_label"]].append(row)

    class_ap = {}
    for label in labels_with_gt:
        gt_for_label = gt_by_label_image[label]
        npos = sum(len(items) for items in gt_for_label.values())
        preds = sorted(preds_by_label.get(label, []), key=lambda row: row["score"], reverse=True)
        tp = []
        fp = []
        for pred in preds:
            image_gt = gt_for_label.get(pred["image_id"], [])
            best_iou = 0.0
            best_index = None
            for index, target in enumerate(image_gt):
                if target["matched"]:
                    continue
                overlap = iou_xyxy(pred["box"], target["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_iou >= 0.5 and best_index is not None:
                image_gt[best_index]["matched"] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        if not preds:
            class_ap[label] = 0.0
            continue

        cumulative_tp = []
        cumulative_fp = []
        running_tp = 0.0
        running_fp = 0.0
        for true_positive, false_positive in zip(tp, fp):
            running_tp += true_positive
            running_fp += false_positive
            cumulative_tp.append(running_tp)
            cumulative_fp.append(running_fp)
        recalls = [value / npos for value in cumulative_tp]
        precisions = [
            cumulative_tp[index] / max(cumulative_tp[index] + cumulative_fp[index], 1e-12)
            for index in range(len(cumulative_tp))
        ]
        class_ap[label] = average_precision(recalls, precisions)

    return {
        "f_map_50": sum(class_ap.values()) / len(class_ap),
        "class_ap_50": class_ap,
        "frames_evaluated": len(frame_ids),
    }


def mean_present(values):
    clean = [float(value) for value in values if value not in ("", None)]
    if not clean:
        return ""
    return round(sum(clean) / len(clean), 6)


def safe_int(value):
    if value in ("", None):
        return None
    return int(float(value))


def safe_float(value):
    if value in ("", None):
        return None
    return float(value)


def build_eval_payload(run, annotation_cache, label_values):
    mapping = run["mapping"]
    annotation_clip = mapping["annotation_clip"]
    offset = int(mapping["source_start_frame"])
    annotations = annotation_cache[annotation_clip]
    eval_frame_rows = read_csv_rows(Path(run["run_dir"]) / "sia_inference_frames.csv")
    prediction_csv_rows = read_csv_rows(Path(run["run_dir"]) / "predictions.csv")

    frame_records = []
    frame_lookup = {}
    for row in eval_frame_rows:
        center_frame = safe_int(row["center_frame_index"])
        if center_frame is None:
            continue
        source_frame = center_frame + offset
        image_id = f"{run['run_id']}:{source_frame}"
        record = {
            "image_id": image_id,
            "center_frame_index": center_frame,
            "source_frame_index": source_frame,
        }
        frame_records.append(record)
        frame_lookup[center_frame] = record

    gt_rows = []
    for record in frame_records:
        for gt in annotations.ground_truth_for_frame(record["source_frame_index"]):
            row = {
                "mode": run["mode"],
                "backend_name": run["backend_name"],
                "precision": run["precision"],
                "run_id": run["run_id"],
                "video_path": run["video_path"],
                "annotation_clip": annotation_clip,
                "center_frame_index": record["center_frame_index"],
                "source_frame_index": record["source_frame_index"],
                "actor_id": gt["actor_id"],
                "meva_label": gt["meva_label"],
                "sia_label": gt["sia_label"],
                "x1": gt["box"][0],
                "y1": gt["box"][1],
                "x2": gt["box"][2],
                "y2": gt["box"][3],
                "image_id": record["image_id"],
                "box": gt["box"],
            }
            gt_rows.append(row)

    pred_rows = []
    for row in prediction_csv_rows:
        center_frame = safe_int(row["center_frame_index"])
        if center_frame not in frame_lookup:
            continue
        label = row.get("predicted_label", "")
        if label not in label_values:
            continue
        record = frame_lookup[center_frame]
        box = [
            safe_float(row["x1"]),
            safe_float(row["y1"]),
            safe_float(row["x2"]),
            safe_float(row["y2"]),
        ]
        if any(value is None for value in box):
            continue
        pred_rows.append(
            {
                "mode": run["mode"],
                "backend_name": run["backend_name"],
                "precision": run["precision"],
                "run_id": run["run_id"],
                "video_path": run["video_path"],
                "annotation_clip": annotation_clip,
                "center_frame_index": center_frame,
                "source_frame_index": record["source_frame_index"],
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "predicted_label": label,
                "score": safe_float(row["score"]) or 0.0,
                "image_id": record["image_id"],
                "box": box,
            }
        )

    return {
        "frame_ids": {record["image_id"] for record in frame_records},
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
    }


def exportable_prediction_row(row):
    return {field: row.get(field, "") for field in PREDICTION_EXPORT_FIELDNAMES}


def exportable_gt_row(row):
    return {field: row.get(field, "") for field in GT_EXPORT_FIELDNAMES}


def load_inference_frame_set(run_dir):
    frames = set()
    for row in read_csv_rows(Path(run_dir) / "sia_inference_frames.csv"):
        frame_index = safe_int(row.get("center_frame_index"))
        if frame_index is not None:
            frames.add(frame_index)
    return frames


def load_prediction_rows_by_frame(run_dir):
    rows_by_frame = defaultdict(list)
    for row in read_csv_rows(Path(run_dir) / "predictions.csv"):
        frame_index = safe_int(row.get("center_frame_index"))
        if frame_index is None:
            continue
        box = [
            safe_float(row.get("x1")),
            safe_float(row.get("y1")),
            safe_float(row.get("x2")),
            safe_float(row.get("y2")),
        ]
        if any(value is None for value in box):
            continue
        rows_by_frame[frame_index].append(
            {
                "label": row.get("predicted_label", ""),
                "score": safe_float(row.get("score")) or 0.0,
                "box": box,
            }
        )
    return rows_by_frame


def match_prediction_rows(reference_rows, candidate_rows, same_label=False, iou_threshold=0.5):
    matched_candidate = set()
    matches = 0
    for ref in reference_rows:
        best_iou = 0.0
        best_index = None
        for index, candidate in enumerate(candidate_rows):
            if index in matched_candidate:
                continue
            if same_label and ref["label"] != candidate["label"]:
                continue
            overlap = iou_xyxy(ref["box"], candidate["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index is not None:
            matched_candidate.add(best_index)
            matches += 1
    return matches


def prediction_row_fidelity(reference_rows_by_frame, candidate_rows_by_frame, common_frames):
    label_jaccards = []
    ref_count = 0
    cand_count = 0
    box_matches = 0
    same_label_matches = 0
    for frame_index in common_frames:
        reference_rows = reference_rows_by_frame.get(frame_index, [])
        candidate_rows = candidate_rows_by_frame.get(frame_index, [])
        ref_count += len(reference_rows)
        cand_count += len(candidate_rows)
        reference_labels = {row["label"] for row in reference_rows}
        candidate_labels = {row["label"] for row in candidate_rows}
        union = reference_labels | candidate_labels
        label_jaccards.append(
            1.0 if not union else len(reference_labels & candidate_labels) / len(union)
        )
        box_matches += match_prediction_rows(reference_rows, candidate_rows, same_label=False)
        same_label_matches += match_prediction_rows(reference_rows, candidate_rows, same_label=True)

    box_precision = box_matches / cand_count if cand_count else (1.0 if ref_count == 0 else 0.0)
    box_recall = box_matches / ref_count if ref_count else (1.0 if cand_count == 0 else 0.0)
    same_label_precision = same_label_matches / cand_count if cand_count else (1.0 if ref_count == 0 else 0.0)
    same_label_recall = same_label_matches / ref_count if ref_count else (1.0 if cand_count == 0 else 0.0)
    return {
        "reference_pred_rows": ref_count,
        "candidate_pred_rows": cand_count,
        "label_jaccard_mean": mean_present(label_jaccards),
        "box_iou50_precision": round(box_precision, 6),
        "box_iou50_recall": round(box_recall, 6),
        "box_iou50_f1": round(f1_score(box_precision, box_recall), 6),
        "same_label_iou50_precision": round(same_label_precision, 6),
        "same_label_iou50_recall": round(same_label_recall, 6),
        "same_label_iou50_f1": round(f1_score(same_label_precision, same_label_recall), 6),
    }


def compute_fidelity_rows(runs):
    ok_runs = [run for run in runs if run["status"] == "ok"]
    reference_by_key = {
        (run["mode"], run["video_path"]): run
        for run in ok_runs
        if run["backend_name"] == "pytorch" and run["precision"] == "fp32"
    }
    frames_cache = {}
    preds_cache = {}
    rows = []
    for run in ok_runs:
        key = (run["mode"], run["video_path"])
        reference = reference_by_key.get(key)
        if reference is None:
            continue
        reference_dir = Path(reference["run_dir"])
        candidate_dir = Path(run["run_dir"])
        if reference_dir not in frames_cache:
            frames_cache[reference_dir] = load_inference_frame_set(reference_dir)
            preds_cache[reference_dir] = load_prediction_rows_by_frame(reference_dir)
        if candidate_dir not in frames_cache:
            frames_cache[candidate_dir] = load_inference_frame_set(candidate_dir)
            preds_cache[candidate_dir] = load_prediction_rows_by_frame(candidate_dir)

        reference_frames = frames_cache[reference_dir]
        candidate_frames = frames_cache[candidate_dir]
        common_frames = sorted(reference_frames & candidate_frames)
        metrics = prediction_row_fidelity(
            preds_cache[reference_dir],
            preds_cache[candidate_dir],
            common_frames,
        )
        rows.append(
            {
                "mode": run["mode"],
                "video_path": run["video_path"],
                "reference_backend_name": reference["backend_name"],
                "reference_precision": reference["precision"],
                "candidate_backend_name": run["backend_name"],
                "candidate_precision": run["precision"],
                "candidate_run_dir": run["run_dir"],
                "common_frames": len(common_frames),
                "reference_frames": len(reference_frames),
                "candidate_frames": len(candidate_frames),
                **metrics,
            }
        )
    return rows


def run_runtime_suite(args, suite_dir, video_paths, manifest):
    if not args.skip_runs:
        from tools.offline_runtime_demo import run_offline_runtime

    invoked_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    variants = backend_variants(args)
    runs = []
    failures = []
    total = len(video_paths) * len(args.modes) * len(variants)
    counter = 0
    for mode in args.modes:
        raw_mode_config = load_json(mode_config_path(args, mode))
        for variant in variants:
            variant_label = f"{variant['backend_name']}_{variant['precision']}"
            for video_index, video_path in enumerate(video_paths, start=1):
                counter += 1
                video_label = sanitize_name(video_path.stem)
                run_id = f"{mode}_{variant_label}_{video_index:03d}_{video_label}"
                run_dir = suite_dir / "runs" / mode / variant_label / f"{video_index:03d}_{video_label}"
                mapping = infer_clip_mapping(video_path, manifest)
                print(f"[{counter}/{total}] {mode} {variant_label}: {video_path}")
                try:
                    config = apply_runtime_overrides(raw_mode_config, args, video_path, variant)
                    if not args.skip_runs:
                        run_offline_runtime(
                            config,
                            invoked_command,
                            run_name=run_id,
                            run_dir=run_dir,
                            progress_callback=make_progress_callback(args),
                        )
                    runs.append(
                        {
                            "mode": mode,
                            "backend_name": variant["backend_name"],
                            "precision": variant["precision"],
                            "optimization_label": variant["optimization_label"],
                            "trt_engine_path": variant["trt_engine_path"] or "",
                            "run_id": run_id,
                            "run_dir": str(run_dir),
                            "video_path": str(video_path),
                            "annotation_clip": mapping["annotation_clip"],
                            "mapping": mapping,
                            "status": "ok",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "mode": mode,
                            "backend_name": variant["backend_name"],
                            "precision": variant["precision"],
                            "video_path": str(video_path),
                            "error": str(exc),
                        }
                    )
                    runs.append(
                        {
                            "mode": mode,
                            "backend_name": variant["backend_name"],
                            "precision": variant["precision"],
                            "optimization_label": variant["optimization_label"],
                            "trt_engine_path": variant["trt_engine_path"] or "",
                            "run_id": run_id,
                            "run_dir": str(run_dir),
                            "video_path": str(video_path),
                            "annotation_clip": mapping["annotation_clip"],
                            "mapping": mapping,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    if not args.continue_on_error:
                        return runs, failures
    return runs, failures


def evaluate_runs(args, suite_dir, runs, manifest, label_map):
    del manifest
    label_values = set(label_map.values())
    annotation_root = Path(args.annotation_root)
    annotation_cache = {}
    summary_rows = []
    all_gt_rows = []
    all_pred_rows = []
    aggregate_by_mode = defaultdict(lambda: {"frame_ids": set(), "gt_rows": [], "pred_rows": []})

    for run in runs:
        if run["status"] != "ok":
            summary_rows.append(
                {
                    "mode": run["mode"],
                    "backend_name": run["backend_name"],
                    "precision": run["precision"],
                    "optimization_label": run["optimization_label"],
                    "trt_engine_path": run["trt_engine_path"],
                    "video_path": run["video_path"],
                    "annotation_clip": run["annotation_clip"],
                    "run_dir": run["run_dir"],
                    "frames_evaluated": 0,
                    "gt_instances": 0,
                    "pred_instances": 0,
                    "f_map_50": "",
                    "sia_inference_iterations": "",
                    "active_frames": "",
                    "frames_read": "",
                    "effective_fps": "",
                    "effective_active_fps": "",
                    "inference_mean_ms": "",
                    "active_loop_mean_ms": "",
                    "status": "error",
                    "error": run["error"],
                }
            )
            continue

        annotation_clip = run["mapping"]["annotation_clip"]
        if annotation_clip not in annotation_cache:
            annotation_cache[annotation_clip] = MevaClipAnnotations(annotation_root, annotation_clip, label_map)

        payload = build_eval_payload(run, annotation_cache, label_values)
        metrics = compute_f_map_50(payload["frame_ids"], payload["gt_rows"], payload["pred_rows"])
        run_metrics = load_json(Path(run["run_dir"]) / "metrics.json")
        timings = run_metrics.get("timings", {})
        summary_rows.append(
            {
                "mode": run["mode"],
                "backend_name": run["backend_name"],
                "precision": run["precision"],
                "optimization_label": run["optimization_label"],
                "trt_engine_path": run["trt_engine_path"],
                "video_path": run["video_path"],
                "annotation_clip": annotation_clip,
                "run_dir": run["run_dir"],
                "frames_evaluated": len(payload["frame_ids"]),
                "gt_instances": len(payload["gt_rows"]),
                "pred_instances": len(payload["pred_rows"]),
                "f_map_50": "" if metrics["f_map_50"] is None else round(metrics["f_map_50"], 6),
                "sia_inference_iterations": run_metrics.get("sia_inference_iterations"),
                "active_frames": run_metrics.get("active_frames"),
                "frames_read": run_metrics.get("frames_read"),
                "effective_fps": run_metrics.get("effective_fps"),
                "effective_active_fps": run_metrics.get("effective_active_fps"),
                "inference_mean_ms": timings.get("inference", {}).get("mean_ms"),
                "active_loop_mean_ms": timings.get("active_loop", {}).get("mean_ms"),
                "status": "ok",
                "error": "",
            }
        )
        all_gt_rows.extend(payload["gt_rows"])
        all_pred_rows.extend(payload["pred_rows"])
        aggregate_key = (run["mode"], run["backend_name"], run["precision"])
        aggregate = aggregate_by_mode[aggregate_key]
        aggregate["frame_ids"].update(payload["frame_ids"])
        aggregate["gt_rows"].extend(payload["gt_rows"])
        aggregate["pred_rows"].extend(payload["pred_rows"])
        aggregate.setdefault("runtime_rows", []).append(run_metrics)

    aggregate_rows = []
    for (mode, backend_name, precision), payload in sorted(aggregate_by_mode.items()):
        metrics = compute_f_map_50(payload["frame_ids"], payload["gt_rows"], payload["pred_rows"])
        runtime_rows = payload.get("runtime_rows", [])
        aggregate_rows.append(
            {
                "mode": mode,
                "backend_name": backend_name,
                "precision": precision,
                "frames_evaluated": len(payload["frame_ids"]),
                "gt_instances": len(payload["gt_rows"]),
                "pred_instances": len(payload["pred_rows"]),
                "f_map_50": "" if metrics["f_map_50"] is None else round(metrics["f_map_50"], 6),
                "sia_inference_iterations": sum(
                    int(row.get("sia_inference_iterations") or 0)
                    for row in runtime_rows
                ),
                "mean_effective_fps": mean_present(row.get("effective_fps") for row in runtime_rows),
                "mean_effective_active_fps": mean_present(
                    row.get("effective_active_fps") for row in runtime_rows
                ),
                "mean_inference_ms": mean_present(
                    row.get("timings", {}).get("inference", {}).get("mean_ms")
                    for row in runtime_rows
                ),
                "mean_active_loop_ms": mean_present(
                    row.get("timings", {}).get("active_loop", {}).get("mean_ms")
                    for row in runtime_rows
                ),
                "class_ap_50": {
                    label: round(value, 6)
                    for label, value in metrics["class_ap_50"].items()
                },
            }
        )

    write_csv(suite_dir / "meva_pilot_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    write_csv(suite_dir / "meva_pilot_aggregate.csv", AGGREGATE_FIELDNAMES, aggregate_rows)
    write_csv(
        suite_dir / "all_predictions.csv",
        PREDICTION_EXPORT_FIELDNAMES,
        [exportable_prediction_row(row) for row in all_pred_rows],
    )
    write_csv(
        suite_dir / "ground_truth.csv",
        GT_EXPORT_FIELDNAMES,
        [exportable_gt_row(row) for row in all_gt_rows],
    )
    fidelity_rows = compute_fidelity_rows(runs)
    write_csv(suite_dir / "prediction_fidelity.csv", FIDELITY_FIELDNAMES, fidelity_rows)
    write_json(
        suite_dir / "meva_pilot_summary.json",
        {
            "suite_dir": str(suite_dir),
            "primary_metric": "f-mAP@0.5",
            "deployment_fidelity_metric": (
                "prediction_fidelity.csv compares each optimized variant to PyTorch FP32 "
                "on the same runtime center frames."
            ),
            "frame_alignment": "SiA predictions are scored on the center frame of the 9-frame runtime window.",
            "summary_rows": summary_rows,
            "aggregate_by_mode_backend_precision": aggregate_rows,
            "prediction_fidelity_rows": fidelity_rows,
        },
    )
    return summary_rows, aggregate_rows


def main():
    args = parse_args()
    video_paths = resolve_video_paths(args)
    suite_dir = suite_output_dir(args)
    manifest = load_structured(args.manifest) if Path(args.manifest).is_file() else {}
    label_map = load_structured(args.label_map)

    runs, failures = run_runtime_suite(args, suite_dir, video_paths, manifest)
    summary_rows, aggregate_rows = evaluate_runs(args, suite_dir, runs, manifest, label_map)

    print(f"MEVA pilot pipeline complete. Artifacts saved to: {suite_dir}")
    print(f"Run summary CSV: {suite_dir / 'meva_pilot_summary.csv'}")
    print(f"Aggregate CSV: {suite_dir / 'meva_pilot_aggregate.csv'}")
    print(f"Prediction fidelity CSV: {suite_dir / 'prediction_fidelity.csv'}")
    print(f"Aggregate JSON: {suite_dir / 'meva_pilot_summary.json'}")
    for row in aggregate_rows:
        print(
            f"  {row['mode']} {row['backend_name']} {row['precision']}: "
            f"f-mAP@0.5={row['f_map_50'] or 'n/a'}, "
            f"frames={row['frames_evaluated']}, gt={row['gt_instances']}, preds={row['pred_instances']}"
        )
    if failures and not args.continue_on_error:
        raise RuntimeError(f"Stopped after failure: {failures[0]}")
    if not summary_rows:
        raise RuntimeError("No runs were evaluated.")


if __name__ == "__main__":
    main()
