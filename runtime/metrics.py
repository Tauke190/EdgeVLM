from collections import defaultdict
from dataclasses import dataclass, field

import torch


STAGE_TIMING_FIELDNAMES = [
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
]


def maybe_cuda_synchronize(device, enabled=True):
    if not enabled:
        return
    if isinstance(device, str):
        should_sync = device.startswith("cuda")
    else:
        should_sync = getattr(device, "type", None) == "cuda"
    if should_sync and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_series(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }

    ordered = sorted(clean)
    count = len(ordered)
    p95_index = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
    return {
        "count": count,
        "mean_ms": round(sum(ordered) / count * 1000.0, 3),
        "median_ms": round(ordered[count // 2] * 1000.0, 3),
        "p95_ms": round(ordered[p95_index] * 1000.0, 3),
        "max_ms": round(max(ordered) * 1000.0, 3),
    }


@dataclass
class RuntimeMetricsCollector:
    stage_rows: list[dict] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record_frame(
        self,
        *,
        frame_index,
        active_iteration,
        capture_s,
        preprocess_s,
        inference_s,
        postprocess_s,
        postprocess_filter_s,
        postprocess_nms_s,
        postprocess_threshold_s,
        label_decode_s,
        render_s,
        loop_s,
        detections,
    ):
        self.series["capture"].append(capture_s)
        self.series["preprocess"].append(preprocess_s)
        self.series["loop"].append(loop_s)

        if active_iteration:
            self.series["inference"].append(inference_s)
            self.series["postprocess"].append(postprocess_s)
            self.series["postprocess_filter"].append(postprocess_filter_s)
            self.series["postprocess_nms"].append(postprocess_nms_s)
            self.series["postprocess_threshold"].append(postprocess_threshold_s)
            self.series["label_decode"].append(label_decode_s)
            self.series["render"].append(render_s)
            self.series["active_loop"].append(loop_s)

        self.stage_rows.append(
            {
                "frame_index": frame_index,
                "active_iteration": int(active_iteration),
                "capture_ms": round(capture_s * 1000.0, 3),
                "preprocess_ms": round(preprocess_s * 1000.0, 3),
                "inference_ms": round(inference_s * 1000.0, 3),
                "postprocess_ms": round(postprocess_s * 1000.0, 3),
                "postprocess_filter_ms": round(postprocess_filter_s * 1000.0, 3),
                "postprocess_nms_ms": round(postprocess_nms_s * 1000.0, 3),
                "postprocess_threshold_ms": round(postprocess_threshold_s * 1000.0, 3),
                "label_decode_ms": round(label_decode_s * 1000.0, 3),
                "render_ms": round(render_s * 1000.0, 3),
                "loop_ms": round(loop_s * 1000.0, 3),
                "detections": detections,
            }
        )

    def summarized_timings(self):
        return {
            "capture": summarize_series(self.series["capture"]),
            "preprocess": summarize_series(self.series["preprocess"]),
            "inference": summarize_series(self.series["inference"]),
            "postprocess": summarize_series(self.series["postprocess"]),
            "postprocess_filter": summarize_series(self.series["postprocess_filter"]),
            "postprocess_nms": summarize_series(self.series["postprocess_nms"]),
            "postprocess_threshold": summarize_series(self.series["postprocess_threshold"]),
            "label_decode": summarize_series(self.series["label_decode"]),
            "render": summarize_series(self.series["render"]),
            "loop": summarize_series(self.series["loop"]),
            "active_loop": summarize_series(self.series["active_loop"]),
        }
