#!/bin/bash

cd "$(dirname "$0")/.."

conda activate edgevlm

python3 inference.py -F video.mp4 -thresh 0.25 -color green