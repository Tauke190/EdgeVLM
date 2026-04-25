import cv2
import torch
from torchvision.transforms import v2


def build_normalizer(mean, std):
    return v2.Normalize(mean, std)


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)


def build_clip_tensor(sampled_clip, normalizer, device, use_fp16):
    clip_tensor = torch.from_numpy(sampled_clip).float() / 255.0
    clip_tensor = normalizer(clip_tensor).unsqueeze(0).to(device)
    if use_fp16:
        clip_tensor = clip_tensor.half()
    return clip_tensor
