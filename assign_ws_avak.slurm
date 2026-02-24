#!/bin/bash
#SBATCH -c 10
#SBATCH --mem=80G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user zh265998@ucf.edu

python -u assign_ws.py -JSON avak_stats_flt_b16_txtaug_txtlora.json \
                        -PRETRAINED weights/avak_stats_flt_b16_txtaug_txtlora/avak_b16_5.pt \
                        -HEIGHT 240 \
                        -WIDTH 320 \
                        -WORKERS 4 \
                        -KINETICS /datasets/kinetics-k700-2020 \
                        -RATEKINETICS 8 \
                        -DET 20 \
                        -FRAMES 9 \
                        --TXTAUG \
                        --TXTLORA \
