from dataclasses import dataclass
import json


@dataclass
class RuntimeConfig:
    mode: str
    weights_path: str
    actions_json: str
    pipeline_mode: str = "always_on"
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
    show_preview: bool = False
    video_path: str | None = None
    video_device: int = 0
    max_frames: int | None = None
    max_seconds: float | None = None
    motion_threshold_area: int = 1000
    motion_frames: int = 3
    motion_cooldown_frames: int = 60
    motion_blur_kernel: int = 5
    motion_learning_rate: float = 0.001

    @classmethod
    def from_dict(cls, payload):
        img_size = payload.get("img_size", [240, 320])
        normalize_mean = payload.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = payload.get("normalize_std", [0.229, 0.224, 0.225])
        pipeline_mode = payload.get("pipeline_mode", "always_on")
        valid_pipeline_modes = {"always_on", "motion_only"}
        if pipeline_mode not in valid_pipeline_modes:
            raise ValueError(
                f"Unsupported pipeline_mode '{pipeline_mode}'. Expected one of: {sorted(valid_pipeline_modes)}"
            )
        return cls(
            mode=payload["mode"],
            weights_path=payload["weights_path"],
            actions_json=payload["actions_json"],
            pipeline_mode=pipeline_mode,
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
            show_preview=bool(payload.get("show_preview", False)),
            video_path=payload.get("video_path"),
            video_device=int(payload.get("video_device", 0)),
            max_frames=payload.get("max_frames"),
            max_seconds=payload.get("max_seconds"),
            motion_threshold_area=int(payload.get("motion_threshold_area", 1000)),
            motion_frames=int(payload.get("motion_frames", 3)),
            motion_cooldown_frames=int(payload.get("motion_cooldown_frames", 60)),
            motion_blur_kernel=int(payload.get("motion_blur_kernel", 5)),
            motion_learning_rate=float(payload.get("motion_learning_rate", 0.001)),
        )

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
