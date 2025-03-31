#!/bin/bash

CONFIG_PATH="configs/gemma2-27b-config.yaml"

echo "Starting inference..."
python inference.py --val_fold 0 --config ${CONFIG_PATH} --checkpoint "./checkpoints/gemma2-27b_0_20250330/checkpoint-1146" --result_dir "submissions/fold0"
python inference.py --val_fold 1 --config ${CONFIG_PATH} --checkpoint "./checkpoints/gemma2-27b_1_20250330/checkpoint-1146" --result_dir "submissions/fold1"
python inference.py --val_fold 2 --config ${CONFIG_PATH} --checkpoint "./checkpoints/gemma2-27b_2_20250330/checkpoint-1146" --result_dir "submissions/fold2"
python inference.py --val_fold 3 --config ${CONFIG_PATH} --checkpoint "./checkpoints/gemma2-27b_3_20250330/checkpoint-1146" --result_dir "submissions/fold3"
python inference.py --val_fold 4 --config ${CONFIG_PATH} --checkpoint "./checkpoints/gemma2-27b_4_20250331/checkpoint-1146" --result_dir "submissions/fold4"
echo "Completed inference"