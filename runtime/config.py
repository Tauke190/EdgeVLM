from dataclasses import dataclass
import json


@dataclass
class RuntimeConfig:
    mode: str
    weights_path: str
    actions_json: str
    pipeline_mode: str = "always_on"
    backend_name: str = "pytorch"
    trt_engine_path: str | None = None
    optimization_label: str | None = None
    device: str = "cuda"
    precision: str = "fp32"
    autocast: bool = False
    model_size: str = "b"
    pretrain: str | None = None
    det_token_num: int = 20
    text_lora: bool = True
    num_frames: int = 9
    buffer_max_len: int = 72
    img_height: int = 240
    img_width: int = 320
    threshold: float = 0.25
    top_k_labels: int | None = 3
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None
    color: str = "green"
    font_scale: float = 0.5
    line_thickness: int = 1
    output_root: str = "results/runtime"
    output_video_name: str = "runtime_output.mp4"
    output_fps: float = 25.0
    video_codec: str = "mp4v"
    sync_cuda_timing: bool = True
    system_metrics_interval_s: float = 1.0
    render_enabled: bool = True
    show_active_tiers: bool = False
    show_preview: bool = False
    video_path: str | None = None
    video_device: int = 0
    simulate_live: bool = False
    drop_frames: bool = False
    source_fps_override: float | None = None
    max_frames: int | None = None
    max_seconds: float | None = None
    motion_threshold_area: int = 1000
    motion_frames: int = 3
    motion_cooldown_frames: int = 60
    motion_blur_kernel: int = 5
    motion_learning_rate: float = 0.001
    person_detector: str = "yolov8n"
    person_weights: str = "weights/yolov8n.pt"
    person_threshold: float = 0.3
    person_precision: str = "fp32"
    person_stride: int = 3
    person_cooldown_frames: int = 6
    person_hit_threshold: float = 0.0
    person_scale: float = 1.05
    person_resize_width: int = 320
    person_min_box_area: int = 4096
    sia_target_fps: float = 9.0
    action_persist_ms: float = 15.0
    sia_min_new_frames: int = 9
    sia_retrigger_on_motion_edge: bool = True
    sia_retrigger_on_person_edge: bool = True

    @classmethod
    def from_dict(cls, payload):
        img_size = payload.get("img_size", [240, 320])
        normalize_mean = payload.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = payload.get("normalize_std", [0.229, 0.224, 0.225])
        pipeline_mode = payload.get("pipeline_mode", "always_on")
        valid_pipeline_modes = {"always_on", "motion_only", "person_only", "motion_person_sia"}
        if pipeline_mode not in valid_pipeline_modes:
            raise ValueError(
                f"Unsupported pipeline_mode '{pipeline_mode}'. Expected one of: {sorted(valid_pipeline_modes)}"
            )
        backend_name = payload.get("backend_name", "pytorch")
        valid_backends = {"pytorch", "tensorrt"}
        if backend_name not in valid_backends:
            raise ValueError(f"Unsupported backend_name '{backend_name}'. Expected one of: {sorted(valid_backends)}")
        return cls(
            mode=payload["mode"],
            weights_path=payload["weights_path"],
            actions_json=payload["actions_json"],
            pipeline_mode=pipeline_mode,
            backend_name=backend_name,
            trt_engine_path=payload.get("trt_engine_path"),
            optimization_label=payload.get("optimization_label"),
            device=payload.get("device", "cuda"),
            precision=payload.get("precision", "fp32"),
            autocast=bool(payload.get("autocast", payload.get("precision", "fp32") == "fp16")),
            model_size=payload.get("model_size", "b"),
            pretrain=payload.get("pretrain"),
            det_token_num=int(payload.get("det_token_num", 20)),
            text_lora=bool(payload.get("text_lora", True)),
            num_frames=int(payload.get("num_frames", 9)),
            buffer_max_len=int(payload.get("buffer_max_len", 72)),
            img_height=int(img_size[0]),
            img_width=int(img_size[1]),
            threshold=float(payload.get("threshold", 0.25)),
            top_k_labels=payload.get("top_k_labels"),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            color=payload.get("color", "green"),
            font_scale=float(payload.get("font_scale", 0.5)),
            line_thickness=int(payload.get("line_thickness", 1)),
            output_root=payload.get("output_root", "results/runtime"),
            output_video_name=payload.get("output_video_name", "runtime_output.mp4"),
            output_fps=float(payload.get("output_fps", 25.0)),
            video_codec=payload.get("video_codec", "mp4v"),
            sync_cuda_timing=bool(payload.get("sync_cuda_timing", True)),
            system_metrics_interval_s=float(payload.get("system_metrics_interval_s", 1.0)),
            render_enabled=bool(payload.get("render_enabled", True)),
            show_active_tiers=bool(payload.get("show_active_tiers", False)),
            show_preview=bool(payload.get("show_preview", False)),
            video_path=payload.get("video_path"),
            video_device=int(payload.get("video_device", 0)),
            simulate_live=bool(payload.get("simulate_live", False)),
            drop_frames=bool(payload.get("drop_frames", False)),
            source_fps_override=payload.get("source_fps_override"),
            max_frames=payload.get("max_frames"),
            max_seconds=payload.get("max_seconds"),
            motion_threshold_area=int(payload.get("motion_threshold_area", 1000)),
            motion_frames=int(payload.get("motion_frames", 3)),
            motion_cooldown_frames=int(payload.get("motion_cooldown_frames", 60)),
            motion_blur_kernel=int(payload.get("motion_blur_kernel", 5)),
            motion_learning_rate=float(payload.get("motion_learning_rate", 0.001)),
            person_detector=payload.get("person_detector", "yolov8n"),
            person_weights=payload.get("person_weights", "weights/yolov8n.pt"),
            person_threshold=float(payload.get("person_threshold", 0.3)),
            person_precision=payload.get("person_precision", "fp32"),
            person_stride=int(payload.get("person_stride", 3)),
            person_cooldown_frames=int(payload.get("person_cooldown_frames", 6)),
            person_hit_threshold=float(payload.get("person_hit_threshold", 0.0)),
            person_scale=float(payload.get("person_scale", 1.05)),
            person_resize_width=int(payload.get("person_resize_width", 320)),
            person_min_box_area=int(payload.get("person_min_box_area", 4096)),
            sia_target_fps=float(payload.get("sia_target_fps", 9.0)),
            action_persist_ms=float(payload.get("action_persist_ms", 15.0)),
            sia_min_new_frames=int(payload.get("sia_min_new_frames", 9)),
            sia_retrigger_on_motion_edge=bool(payload.get("sia_retrigger_on_motion_edge", True)),
            sia_retrigger_on_person_edge=bool(payload.get("sia_retrigger_on_person_edge", True)),
        )

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
