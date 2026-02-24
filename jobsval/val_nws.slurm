#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user zh265998@ucf.edu

python -u val_avak.py -JSON avak_stats_flt_b16_txtaug_txtlora.json \
                       -PRETRAINED weights/avak_stats_flt_b16_txtaug_txtlora \
                       -HEIGHT 240 \
                       -WIDTH 320 \
                       -BS 400 \
                       -WORKERS 4 \
                       -AVA /home/zsia/AVA/videos_15min_30fps \
                       -RATEAVA 8 \
                       -KINETICS /datasets/kinetics-k700-2020 \
                       -RATEKINETICS 8 \
                       -UCF24 /home/zsia/UCF-101 \
                       -RATEUCF24 7 \
                       -ANNOUCF24 anno/UCF101v2-GT.pkl \
                       -HMDB21 /home/zsia/HMDB21 \
                       -ANNOHMDB21 anno/JHMDB-GT.pkl \
                       -DET 20 \
                       -FRAMES 9 \
                       --TXTAUG \
		               --TXTLORA \
