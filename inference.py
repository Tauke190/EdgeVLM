"""
inference.py

Offline single-tier action detection inference using the SIA (Spatio-temporal
Interactive Action) vision-language model.

This script runs SIA on every frame of a video without any hierarchical
gating (no motion or person pre-filtering). It is intended for:
  - Benchmarking the raw SIA model accuracy on a labelled dataset.
  - Quick evaluation where computational efficiency is not the priority.

The model encodes a 9-frame temporal window with a sliding buffer. Text
embeddings for the target actions are pre-computed once and matched against
vision features via cosine similarity to produce per-person action labels.

Usage:
    python inference.py -F <video_path> --act-file <actions.json/txt> [options]
"""
import os
import json
import numpy as np
import cv2
import time
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from sia import get_sia, PostProcessViz

parser = argparse.ArgumentParser(description="Offline Inference with SIA")
parser.add_argument("-F", type=str, required=True, help="file path")
parser.add_argument("-thresh", type=float, default=0.25, help="cosine threshold")
parser.add_argument(
    "--act-file",
    type=str,
    required=True,
    help="Path to a text file with one action description per line",
)
parser.add_argument(
    "--debug-frame-no",
    type=int,
    help="Process only the first N frames, then stop early for quick debugging",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug information, including loaded action descriptions",
)
parser.add_argument("-color", type=str, default='green', help="color to plot predictions")
parser.add_argument("-font", type=float, default=0.5, help="font size")
parser.add_argument("-line", type=int, default=1, help="line thickness")
args = parser.parse_args()


COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

color = COLORS[args.color]
font = args.font
thickness = args.line

print(f"Loading video: {args.F}")
cap = cv2.VideoCapture(args.F)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")
outsize = (frame_height, frame_width)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(
    torch.load('weights/avak_b16_11.pt', map_location=device, weights_only=True),
    strict=False,
)
model.to(device)
model.eval()
print("Model loaded")

tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def load_action_descriptions(action_file):
    """Load action descriptions from either JSON or text file.
    
    JSON format: {"action_name": "description", ...}
    Returns: (captions, action_names) where captions are descriptions and action_names are keys
    """
    if action_file.endswith('.json'):
        with open(action_file, "r", encoding="utf-8") as handle:
            action_dict = json.load(handle)
        if not action_dict:
            raise ValueError(f"No actions found in {action_file}")
        action_names = list(action_dict.keys())
        captions = list(action_dict.values())
    else:
        with open(action_file, "r", encoding="utf-8") as handle:
            captions = [line.strip() for line in handle if line.strip()]
        action_names = captions.copy()
        if not captions:
            raise ValueError(f"No action descriptions found in {action_file}")
    
    return captions, action_names


captions, action_names = load_action_descriptions(args.act_file)

print(f"Actions to detect: {len(captions)}")
if args.debug:
    print("Loaded action descriptions:")
    for idx, (name, desc) in enumerate(zip(action_names, captions), start=1):
        print(f"  {idx:02d}. {name}: {desc}")
text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

# Measure FLOPS for a single forward pass
try:
    from thop import profile
    dummy_clip = torch.randn(1, 3, 9, 240, 320).to(device)
    flops, params = profile(model.vision_encoder, inputs=(dummy_clip,), verbose=False)
    print(f"FLOPs per forward pass: {flops / 1e9:.2f} GFLOPs")
except ImportError:
    print("thop not installed, skipping FLOPS measurement")

imgsize = (240, 320)  # Resize resolution fed into the SIA vision encoder
output_path = 'pred_' + args.F.split('.')[0] + '.mp4'
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

# Sliding window buffer: holds the last `buffer_max_len` resized frames.
# Every inference call samples 9 evenly-spaced frames from this buffer.
buffer_max_len = 72
mididx = buffer_max_len // 2  # Mid-index used for box temporal alignment
buffer = []       # Resized frames fed to the model
plotbuffer = []   # Original-resolution frames for visualisation overlay

# Store last prediction so labels persist on-screen between inference calls
last_boxes = None
last_labels = None
last_scores = None

postprocess = PostProcessViz()  # Converts raw model outputs to boxes + labels
init = 0
ret = True
frame_count = 0
forward_passes = 0  # Counts actual SIA forward passes for throughput reporting

print("Starting inference...")
while ret:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    raw_image = frame
    out_frame = raw_image.copy()
    
    plotbuffer.append(raw_image.transpose(2, 0, 1))
    raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    color_image = raw_image.transpose(2, 0, 1)
    buffer.append(color_image)
    
    if len(buffer) > buffer_max_len:
        forward_passes += 1
        _ = buffer.pop(0)
        _ = plotbuffer.pop(0)
        clip_torch = torch.tensor(np.array(buffer)[0:buffer_max_len:buffer_max_len//9]) / 255
        clip_torch = tfs(clip_torch)

        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.unsqueeze(0).to(device))
            outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
            result = postprocess(outputs, outsize, human_conf=0.9, thresh=args.thresh)[0]
            result['text_labels'] = [[action_names[e] for e in ele] for ele in result['labels']]
            last_boxes = result['boxes']
            last_labels = result['text_labels']
            last_scores = result['scores']

        # Display on latest frame for real-time feedback
        # (predictions computed from 72-frame context centered ~36 frames ago)
        out_frame = plotbuffer[-1].transpose(1, 2, 0).astype(np.uint8)
        for j in range(len(last_boxes)):
            box = last_boxes[j]
            if torch.is_tensor(box):
                box = box.cpu().detach().numpy()
            else:
                box = np.array(box)
            
            label = last_labels[j]
            score = last_scores[j]
            
            # Convert to integers
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            start_point = (x1, y1)
            end_point = (x2, y2)
            
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, int(thickness))
            # Only show the action with highest probability
            if len(label) > 0:
                max_idx = torch.argmax(score).item()
                act = label[max_idx]
                sco = score[max_idx]
                text = act + ' ' + str(round(sco.item(), 2))
                print(f"Frame {frame_count}: {text}")
                out_frame = cv2.putText(out_frame, text, (int(box[0])-5, int(box[1])+20), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
    else:
        # Before buffer fills, display latest frame without predictions
        out_frame = plotbuffer[-1].transpose(1, 2, 0).astype(np.uint8)
    
    # Write to video (after buffer warm-up period)
    if len(buffer) >= buffer_max_len:
        writer.write(out_frame)
    
    end = time.time()
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")
    if args.debug_frame_no is not None and frame_count >= args.debug_frame_no:
        print(f"Debug frame limit reached at frame {frame_count}, stopping early.")
        break

# Process remaining frames in buffer after video ends (drain buffer)
# Pad with last frame so inference always runs, giving fresh predictions per frame
while len(plotbuffer) > 0:
    if buffer:
        buffer.pop(0)
    
    if len(buffer) > 0:
        # Pad buffer to full size using last frame so 9-frame sampling always works
        buf_padded = list(buffer)
        while len(buf_padded) < buffer_max_len:
            buf_padded.append(buf_padded[-1])
        
        forward_passes += 1
        clip_torch = torch.tensor(np.array(buf_padded)[0:buffer_max_len:buffer_max_len//9]) / 255
        clip_torch = tfs(clip_torch)
        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.unsqueeze(0).to(device))
            outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
            result = postprocess(outputs, outsize, human_conf=0.9, thresh=args.thresh)[0]
            result['text_labels'] = [[action_names[e] for e in ele] for ele in result['labels']]
            last_boxes = result['boxes']
            last_labels = result['text_labels']
            last_scores = result['scores']
    
    plot_frame = plotbuffer.pop(0)
    out_frame = plot_frame.transpose(1, 2, 0).astype(np.uint8)
    if last_boxes is not None:
        for j in range(len(last_boxes)):
            box = last_boxes[j]
            if torch.is_tensor(box):
                box = box.cpu().detach().numpy()
            else:
                box = np.array(box)
            label = last_labels[j]
            score = last_scores[j]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            out_frame = cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, int(thickness))
            if len(label) > 0:
                max_idx = torch.argmax(score).item()
                text = label[max_idx] + ' ' + str(round(score[max_idx].item(), 2))
                print(f"Frame {frame_count}: {text}")
                out_frame = cv2.putText(out_frame, text, (x1-5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, font, color, int(thickness), cv2.LINE_AA)
    writer.write(out_frame)

cap.release()
writer.release()

print(f"\n✓ Inference complete!")
print(f"  Output: {output_path}")
print(f"\nStats:")
print(f"  Total frames: {frame_count}")
print(f"  Forward passes: {forward_passes}")
