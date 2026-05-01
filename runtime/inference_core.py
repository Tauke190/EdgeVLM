from contextlib import nullcontext
import json
import time

import torch
import torch.nn.functional as F

from sia import get_sia
from runtime.metrics import maybe_cuda_synchronize
from runtime.postprocess import RuntimePostProcessor


def load_actions(actions_json_path):
    with open(actions_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.keys())


def decode_text_labels(label_ids_per_box, score_values_per_box, captions, top_k_labels=None):
    decoded_labels = []
    decoded_scores = []
    for label_ids, score_values in zip(label_ids_per_box, score_values_per_box):
        if top_k_labels is not None:
            label_ids = label_ids[:top_k_labels]
            score_values = score_values[:top_k_labels]
        if torch.is_tensor(label_ids):
            label_indices = label_ids.tolist()
        else:
            label_indices = list(label_ids)
        decoded_labels.append([captions[index] for index in label_indices])
        if torch.is_tensor(score_values):
            decoded_scores.append(score_values.tolist())
        else:
            decoded_scores.append([float(score) for score in score_values])
    return decoded_labels, decoded_scores


class SIARuntimeCore:
    def __init__(self, config):
        self.config = config
        self.backend_name = config.backend_name
        self.device = config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
        self.device_is_cuda = str(self.device).startswith("cuda")
        self.precision_uses_cuda_fp16 = config.precision == "fp16" and self.device_is_cuda
        self.input_use_fp16 = self.backend_name == "pytorch" and self.precision_uses_cuda_fp16 and not config.autocast
        if config.precision == "fp16" and not self.precision_uses_cuda_fp16:
            raise RuntimeError("FP16 precision is only supported on CUDA in the runtime core.")
        if self.backend_name == "tensorrt" and not self.device_is_cuda:
            raise RuntimeError("TensorRT backend is only supported on CUDA in the runtime core.")
        if self.backend_name == "tensorrt" and not config.trt_engine_path:
            raise RuntimeError("TensorRT backend requires `trt_engine_path` in the runtime config.")

        self.model = get_sia(
            size=config.model_size,
            pretrain=config.pretrain,
            det_token_num=config.det_token_num,
            text_lora=config.text_lora,
            num_frames=config.num_frames,
        )["sia"]
        self.model.load_state_dict(
            torch.load(config.weights_path, map_location=self.device, weights_only=True),
            strict=False,
        )
        self.model.to(self.device)
        if self.input_use_fp16:
            self.model.half()
        self.model.eval()
        self.vision_backend = None
        if self.backend_name == "tensorrt":
            from runtime.tensorrt_backend import TensorRTVisionBackend

            self.vision_backend = TensorRTVisionBackend(config.trt_engine_path, self.device)

        self.captions = tuple(load_actions(config.actions_json))
        text_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.precision_uses_cuda_fp16 and config.autocast
            else nullcontext()
        )
        with torch.no_grad():
            with text_context:
                self.text_embeds = self.model.encode_text(self.captions)
        self.text_embeds = F.normalize(self.text_embeds, dim=-1)
        self.postprocess = RuntimePostProcessor(config.threshold, config.human_confidence_threshold)

    def infer_clip(self, clip_tensor, frame_size):
        timings = {
            "inference_s": 0.0,
            "postprocess_s": 0.0,
            "postprocess_filter_s": 0.0,
            "postprocess_nms_s": 0.0,
            "postprocess_threshold_s": 0.0,
            "label_decode_s": 0.0,
        }

        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        inference_start = time.perf_counter()
        with torch.no_grad():
            if self.backend_name == "tensorrt":
                outputs = self.vision_backend.infer(clip_tensor)
            else:
                inference_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.precision_uses_cuda_fp16 and self.config.autocast
                    else nullcontext()
                )
                with inference_context:
                    outputs = self.model.encode_vision(clip_tensor)
            similarity_text_embeds = self.text_embeds.to(dtype=outputs["pred_logits"].dtype)
            outputs["pred_logits"] = F.normalize(outputs["pred_logits"], dim=-1) @ similarity_text_embeds.T
            outputs = {
                key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                for key, value in outputs.items()
            }
        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        timings["inference_s"] = time.perf_counter() - inference_start

        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        postprocess_start = time.perf_counter()
        result = self.postprocess(outputs, frame_size, return_stage_timings=True)
        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        timings["postprocess_s"] = time.perf_counter() - postprocess_start
        if result["stage_timings"] is not None:
            timings["postprocess_filter_s"] = result["stage_timings"]["human_filter_s"]
            timings["postprocess_nms_s"] = result["stage_timings"]["nms_s"]
            timings["postprocess_threshold_s"] = result["stage_timings"]["threshold_s"]

        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        label_decode_start = time.perf_counter()
        labels, scores = decode_text_labels(
            result["label_ids"],
            result["scores"],
            self.captions,
            self.config.top_k_labels,
        )
        maybe_cuda_synchronize(self.device, self.config.sync_cuda_timing)
        timings["label_decode_s"] = time.perf_counter() - label_decode_start
        return {
            "boxes": result["boxes"],
            "labels": labels,
            "scores": scores,
            "num_detections": len(result["boxes"]),
            "timings": timings,
        }
