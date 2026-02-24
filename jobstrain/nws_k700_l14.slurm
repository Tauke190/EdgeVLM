#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=100G
#SBATCH --time=336:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user zh265998@ucf.edu

python -u train_avak.py -JSON k700_stats_flt_l14_txtaug_txtlora.json \
                        -SIZE l14 \
                        -HEIGHT 240 \
                        -WIDTH 320 \
                        -BS 72 \
                        -WORKERS 4 \
                        -EPOCH 12 \
                        -AVA /home/zsia/AVA/videos_15min_30fps \
                        -RATEAVA 8 \
                        -KINETICS /datasets/kinetics-k700-2020 \
                        -RATEKINETICS 8 \
                        -DET 20 \
                        -FRAMES 9 \
                        --TXTLORA \
                        --SAVE \
                        -TRAIN K700 \
