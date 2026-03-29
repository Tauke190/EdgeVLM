#!/bin/bash

cd "$(dirname "$0")/.."

conda activate edgevlm

python multi-tier_inference.py -F falling.mp4 --debug