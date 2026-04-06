#!/bin/bash

cd "$(dirname "$0")/.."

conda activate edgevlm

python3 inference.py  \
    -F bedgetout.mp4 \
    -thresh 0.3 \
    -color green \
    --act-file gpt/PATIENT_ACTIONS.json