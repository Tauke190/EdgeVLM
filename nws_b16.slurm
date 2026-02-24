#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=100G
#SBATCH --time=336:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user zh265998@ucf.edu

python -u train_avak.py -JSON avak_nws_stats_flt_b16_txtaug_txtlora.json \
                        -SIZE b16 \
                        -HEIGHT 240 \
                        -WIDTH 320 \
                        -BS 72 \
                        -WORKERS 4 \
                        -EPOCH 6 \
                        -AVA /home/zsia/AVA/videos_15min_30fps \
                        -RATEAVA 8 \
                        -KINETICS /datasets/kinetics-k700-2020 \
                        -RATEKINETICS 8 \
                        -DET 20 \
                        -FRAMES 9 \
                        --TXTAUG \
                        --TXTLORA \
                        --SAVE \
