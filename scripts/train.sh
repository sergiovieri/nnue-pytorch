#!/bin/bash

#python train.py ../Stockfish/src/d2_1000000000.binpack ../data/d10_128000_6293.binpack --lambda 1.0 --gpus 1 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20 --smart-fen-skipping
python train.py ../data/d5_2b.binpack ../data/d10_128000_6293.binpack --lambda 1.0 --gpus 1 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20 --smart-fen-skipping --resume_from_checkpoint logs/default/version_36/checkpoints/last.ckpt
