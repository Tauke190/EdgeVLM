#!/bin/bash

cd "$(dirname "$0")/.."

conda activate edgevlm

python multi-tier_inference.py -F bus.avi \
    --cooldown 30 \
    --act-map-file gpt/MEVA_to_GPT_AVA.json  \
    --person-weights weights/yolov8n.pt \
    --person-precision fp32 \
    --action-weights weights/avak_b16_11.pt \
    --action-precision fp16 \
    --debug-start-frame 1 \
    --person-stride 2 \
    --action-stride 4 \
    --debug 

# hallway.avi
# --debug-frame-no 1000 \