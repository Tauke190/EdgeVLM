import time

from runtime.buffer import SlidingWindowBuffer
from runtime.inference_core import SIARuntimeCore
from runtime.metrics import maybe_cuda_synchronize
from runtime.motion import MotionGate
from runtime.preprocess import build_clip_tensor, build_normalizer, resize_frame
from runtime.visualize import draw_predictions, resolve_color


class AlwaysOnSIAPipeline:
    def __init__(self, config):
        self.config = config
        self.core = SIARuntimeCore(config)
        self.buffer = SlidingWindowBuffer(config.buffer_max_len, config.num_frames)
        self.normalizer = build_normalizer(config.normalize_mean, config.normalize_std)
        self.color = resolve_color(config.color)
        self.motion_gate = None
        if config.pipeline_mode == "motion_only":
            self.motion_gate = MotionGate(
                threshold_area=config.motion_threshold_area,
                motion_frames=config.motion_frames,
                cooldown_frames=config.motion_cooldown_frames,
                blur_kernel=config.motion_blur_kernel,
                learning_rate=config.motion_learning_rate,
            )

    def process_frame(self, frame, frame_size):
        preprocess_start = time.perf_counter()
        original_chw = frame.transpose(2, 0, 1)
        resized = resize_frame(frame, self.config.img_width, self.config.img_height)
        resized_chw = resized.transpose(2, 0, 1)
        self.buffer.push(resized_chw, original_chw)
        preprocess_time = time.perf_counter() - preprocess_start
        gate_state = {
            "motion_detected": None,
            "motion_active": None,
            "motion_roi": None,
        }
        if self.motion_gate is not None:
            gate_state = self.motion_gate.update(frame)

        if not self.buffer.ready():
            return {
                "active": False,
                "rendered_frame": frame.copy(),
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    "preprocess_s": preprocess_time,
                    "inference_s": 0.0,
                    "postprocess_s": 0.0,
                    "postprocess_filter_s": 0.0,
                    "postprocess_nms_s": 0.0,
                    "postprocess_threshold_s": 0.0,
                    "label_decode_s": 0.0,
                    "render_s": 0.0,
                },
            }

        if self.config.pipeline_mode == "motion_only" and not gate_state["motion_active"]:
            return {
                "active": False,
                "rendered_frame": self.buffer.render_frame(),
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    "preprocess_s": preprocess_time,
                    "inference_s": 0.0,
                    "postprocess_s": 0.0,
                    "postprocess_filter_s": 0.0,
                    "postprocess_nms_s": 0.0,
                    "postprocess_threshold_s": 0.0,
                    "label_decode_s": 0.0,
                    "render_s": 0.0,
                },
            }

        clip_tensor = build_clip_tensor(
            self.buffer.sampled_clip(),
            self.normalizer,
            self.core.device,
            self.core.use_fp16 and not self.config.autocast,
        )
        inference_result = self.core.infer_clip(clip_tensor, frame_size)
        render_base = self.buffer.render_frame()

        maybe_cuda_synchronize(self.core.device, self.config.sync_cuda_timing)
        render_start = time.perf_counter()
        rendered_frame = draw_predictions(
            render_base,
            inference_result["boxes"],
            inference_result["labels"],
            inference_result["scores"],
            self.color,
            self.config.font_scale,
            self.config.line_thickness,
        )
        maybe_cuda_synchronize(self.core.device, self.config.sync_cuda_timing)
        render_time = time.perf_counter() - render_start
        return {
            "active": True,
            "rendered_frame": rendered_frame,
            "detections": inference_result["num_detections"],
            "boxes": inference_result["boxes"],
            "labels": inference_result["labels"],
            "scores": inference_result["scores"],
            "gate_state": gate_state,
            "timings": {
                "preprocess_s": preprocess_time,
                **inference_result["timings"],
                "render_s": render_time,
            },
        }
