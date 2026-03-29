#!/bin/bash

# UCF24 Evaluation Script

python -u val_avak.py -JSON ucf24_stats_flt_b16_txtaug_txtlora.json \
                      -PRETRAINED weights/ \
                      -HEIGHT 240 \
                      -WIDTH 320 \
                      -BS 32 \
                      -WORKERS 4 \
                      -UCF24 /mnt/SSD2/UCF101_v2/rgb-images \
                      -RATEUCF24 7 \
                      -ANNOUCF24 /mnt/SSD2/UCF101_v2/UCF101v2-GT.pkl \
                      -DET 20 \
                      -FRAMES 9 \
                      --TXTLORA \
                      -VAL UCF24
