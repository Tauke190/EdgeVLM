# SiA — Open Vocabulary Action Detection

PyTorch implementation of **SiA** (Simple Architecture for Open Vocabulary Action Detection).  
SiA uses learnable `[DET]` tokens inside a ViCLIP vision-language encoder to perform single-stage spatio-temporal action detection without a decoder.

> **EZACT: Efficient Open Vocabulary Action Detection**  
> Z.H Sia and Y.S Rawat

---

## Setup

```bash
conda create -n ezact python=3.10
conda activate ezact
pip install -r requirements.txt
```

Pretrained weights go in `weights/`:
- `weights/avak_b16_11.pt` — SIA action model (ViT-B/16, fine-tuned on AVA-Kinetics)
- `weights/yolov8n.pt` — YOLOv8n person detector

---

## Multi-Tier Inference (Edge Deployment)

`multi-tier_inference_avinash.py` runs a hierarchical 3-tier pipeline optimised for edge devices:

| Tier | Model | Trigger |
|------|-------|---------|
| 1 — Motion | MOG2 background subtraction (CPU) | Every frame |
| 2 — Person | YOLOv8n / MobileNet-SSD (GPU) | Motion detected |
| 3 — Action | SIA vision-language model (GPU) | Person detected |

### Basic usage

```bash
python multi-tier_inference_avinash.py -F <video.mp4>
```

Output video is saved as `multitier_<video>.mp4` in the current directory.

### Common options

```bash
# Custom action mapping and output folder
python multi-tier_inference_avinash.py \
  -F video.mp4 \
  --act-map-file gpt/MEVA_to_GPT_AVA.json \
  --output-dir output/

# FP16 inference for faster edge GPU throughput
python multi-tier_inference_avinash.py \
  -F video.mp4 \
  --sia-precision fp16 \
  --person-precision fp16

# TensorRT person detector (Jetson / NVIDIA edge devices)
python multi-tier_inference_avinash.py \
  -F video.mp4 \
  --person-weights weights/yolov8n.engine

# Skip action detection — motion + person only
python multi-tier_inference_avinash.py \
  -F video.mp4 \
  --skip-tier3

# Quick debug: process only first 200 frames
python multi-tier_inference_avinash.py \
  -F video.mp4 \
  --debug-frame-no 200 --debug
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `-F` | — | **Required.** Input video path |
| `-thresh` | `0.3` | Cosine similarity threshold for action classification |
| `--act-map-file` | `gpt/MEVA_to_GPT_AVA.json` | JSON mapping domain labels → AVA action keys |
| `--output-dir` | `.` | Directory for the output video |
| `--motion-thresh` | `1000` | Min contour area (px²) to trigger motion |
| `--motion-frames` | `3` | Consecutive motion frames needed to activate Tier 2 |
| `--cooldown` | `60` | Frames to stay active after motion/person disappears |
| `--person-model` | `yolov8n` | Person detector: `yolov8n` or `mobilenet-ssd` |
| `--person-weights` | `weights/yolov8n.pt` | Path to person detector weights (`.pt` or `.engine`) |
| `--person-thresh` | `0.3` | Person detection confidence threshold |
| `--person-precision` | `fp32` | YOLO precision: `fp32` or `fp16` |
| `--person-stride` | `1` | Run person detection every N eligible frames |
| `--sia-weights` | `weights/avak_b16_11.pt` | Path to SIA model weights |
| `--sia-precision` | `fp32` | SIA precision: `fp32` or `fp16` |
| `--action-stride` | `1` | Run action detection every N eligible frames |
| `--skip-tier3` | `False` | Disable action detection (Tiers 1+2 only) |
| `--debug` | `False` | Overlay motion mask and tier state on output |
| `--debug-frame-no` | — | Stop after N frames (debugging) |
| `--debug-start-frame` | — | Start from frame N (1-based, debugging) |
| `-color` | `green` | Label text colour |
| `-font` | `0.5` | Label font scale |
| `-line` | `2` | Bounding box line thickness |

---

## Offline Single-Tier Inference

Runs SIA on every frame without hierarchical gating. Use for accuracy benchmarking.

```bash
python inference.py -F video.mp4 --act-file gpt/GPT_AVA.json
```

---

## Training

```bash
# Train on AVA-Kinetics (ViT-B/16, batch 100, 6 epochs)
python train_avak.py \
  -SIZE b16 -FRAMES 9 -BS 100 -EPOCH 6 -LR 1e-5 \
  -TRAIN AVAK \
  -AVA <ava_video_dir> \
  -KINETICS <k700_video_dir> \
  --TXTAUG --TXTLORA \
  -JSON stats.json
```

---

## Evaluation

```bash
# Evaluate on UCF24
python val_avak.py \
  -SIZE b16 -VAL UCF24 \
  -UCF24 <ucf24_video_dir> \
  -ANNOUCF24 datasets/anno/UCF101v2-GT.pkl \
  -JSON val_results.json

# Evaluate on AVA
python val_avak.py \
  -SIZE b16 -VAL AVA \
  -AVA <ava_video_dir> \
  -JSON val_results.json
```

---

## Project Structure

```
weights/          # Model weights (avak_b16_11.pt, yolov8n.pt)
gpt/              # GPT-generated action prompt banks (JSON)
datasets/         # Dataset loaders: AVA, UCF24, HMDB21, Kinetics, MultiSports
sia/              # SIA model: vision encoder, text encoder, detection heads
utils/            # Config, optimizer, scheduler, logging utilities
util/             # Box operations and misc helpers
configs/          # Example runtime configs (baseline, fp16)
scripts/          # Bash scripts for training/eval/inference
```
