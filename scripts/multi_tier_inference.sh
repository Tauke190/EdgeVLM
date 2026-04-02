#!/bin/bash

cd "$(dirname "$0")/.."

conda activate edgevlm

python multi-tier_inference.py \
  -F SurvellienceFootage.mp4 \
  --precision fp16 \
  --simulate-live \
  --drop-frames \
  --show-preview \
  --debug
