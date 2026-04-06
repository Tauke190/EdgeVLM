#!/bin/bash

# Set output directory (default to ./results if not provided)
OUTPUT_DIR="${1:-.}/results"

python3 multi-tier_inference_v2.py \
  -F "sample_videos/bus.avi" \
  -thresh 0.3 \
  --act-map-file "gpt/MEVA_to_GPT_AVA.json" \
  --debug-frame-no 1000 \
  --debug-start-frame 1 \
  -color "green" \
  -font 0.5 \
  -line 2 \
  --motion-thresh 1000 \
  --motion-frames 3 \
  --cooldown 60 \
  --person-model "yolov8n" \
  --person-weights "weights/yolov8n.pt" \
  --person-thresh 0.3 \
  --person-precision "fp32" \
  --person-stride 1 \
  --sia-weights "weights/avak_b16_11.pt" \
  --sia-precision "fp16" \
  --action-stride 4 \
  --output-dir "$OUTPUT_DIR" \
  --debug