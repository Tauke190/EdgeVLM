from runtime.buffer import SlidingWindowBuffer
from runtime.inference_core import SIARuntimeCore
from runtime.preprocess import build_clip_tensor, build_normalizer, resize_frame
from runtime.visualize import draw_predictions, resolve_color


class AlwaysOnSIAPipeline:
    def __init__(self, config):
        self.config = config
        self.core = SIARuntimeCore(config)
        self.buffer = SlidingWindowBuffer(config.buffer_max_len, config.num_frames)
        self.normalizer = build_normalizer(config.normalize_mean, config.normalize_std)
        self.color = resolve_color(config.color)

    def process_frame(self, frame, frame_size):
        original_chw = frame.transpose(2, 0, 1)
        resized = resize_frame(frame, self.config.img_width, self.config.img_height)
        resized_chw = resized.transpose(2, 0, 1)
        self.buffer.push(resized_chw, original_chw)

        if not self.buffer.ready():
            return {
                "active": False,
                "rendered_frame": frame.copy(),
                "detections": 0,
            }

        clip_tensor = build_clip_tensor(
            self.buffer.sampled_clip(),
            self.normalizer,
            self.core.device,
            self.core.use_fp16 and not self.config.autocast,
        )
        inference_result = self.core.infer_clip(clip_tensor, frame_size)
        render_base = self.buffer.render_frame()
        rendered_frame = draw_predictions(
            render_base,
            inference_result["boxes"],
            inference_result["labels"],
            inference_result["scores"],
            self.color,
            self.config.font_scale,
            self.config.line_thickness,
        )
        return {
            "active": True,
            "rendered_frame": rendered_frame,
            "detections": inference_result["num_detections"],
            "boxes": inference_result["boxes"],
            "labels": inference_result["labels"],
            "scores": inference_result["scores"],
        }
