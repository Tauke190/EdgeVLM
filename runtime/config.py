from dataclasses import dataclass
import json


@dataclass
class RuntimeConfig:
    mode: str
    weights_path: str
    actions_json: str
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
    render_enabled: bool = True
    show_preview: bool = False
    video_path: str | None = None
    video_device: int = 0
    max_frames: int | None = None
    max_seconds: float | None = None

    @classmethod
    def from_dict(cls, payload):
        img_size = payload.get("img_size", [240, 320])
        normalize_mean = payload.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = payload.get("normalize_std", [0.229, 0.224, 0.225])
        return cls(
            mode=payload["mode"],
            weights_path=payload["weights_path"],
            actions_json=payload["actions_json"],
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
            render_enabled=bool(payload.get("render_enabled", True)),
            show_preview=bool(payload.get("show_preview", False)),
            video_path=payload.get("video_path"),
            video_device=int(payload.get("video_device", 0)),
            max_frames=payload.get("max_frames"),
            max_seconds=payload.get("max_seconds"),
        )

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
