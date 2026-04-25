from contextlib import nullcontext
import json

import torch
import torch.nn.functional as F

from sia import get_sia
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
        decoded_labels.append([captions[index] for index in label_ids])
        decoded_scores.append([float(score) for score in score_values])
    return decoded_labels, decoded_scores


class SIARuntimeCore:
    def __init__(self, config):
        self.config = config
        self.device = config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
        self.use_fp16 = config.precision == "fp16" and str(self.device).startswith("cuda")
        if config.precision == "fp16" and not self.use_fp16:
            raise RuntimeError("FP16 precision is only supported on CUDA in the runtime core.")

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
        if self.use_fp16 and not config.autocast:
            self.model.half()
        self.model.eval()

        self.captions = load_actions(config.actions_json)
        text_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16 and config.autocast
            else nullcontext()
        )
        with torch.no_grad():
            with text_context:
                self.text_embeds = self.model.encode_text(self.captions)
        self.text_embeds = F.normalize(self.text_embeds, dim=-1)
        self.postprocess = RuntimePostProcessor(config.threshold)

    def infer_clip(self, clip_tensor, frame_size):
        with torch.no_grad():
            inference_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_fp16 and self.config.autocast
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

        result = self.postprocess(outputs, frame_size)
        labels, scores = decode_text_labels(
            result["label_ids"],
            result["scores"],
            self.captions,
            self.config.top_k_labels,
        )
        return {
            "boxes": result["boxes"],
            "labels": labels,
            "scores": scores,
            "num_detections": len(result["boxes"]),
        }
