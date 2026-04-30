#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

exec ./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --video-device 0 \
  --backend-name tensorrt \
  --trt-engine-path results/tensorrt_vision/fp16_check/sia_vision_fp16.engine \
  --motion-min-on-time 30 \
  --person-min-on-time 30 \
  --show-active-tiers \
  --show-preview
