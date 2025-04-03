#!/bin/bash

#CONFIG_PATH="configs/gemma2-27b-config.yaml"
CONFIG_PATH="configs/mdeberta.yaml"

echo "Starting inference..."
python inference.py --val_fold 0 --config ${CONFIG_PATH} --checkpoint "./checkpoints/mdeberta_0_20250331/checkpoint-573" --result_dir "submissions_mdeberta/fold0"
python inference.py --val_fold 1 --config ${CONFIG_PATH} --checkpoint "./checkpoints/mdeberta_1_20250331/checkpoint-573" --result_dir "submissions_mdeberta/fold1"
python inference.py --val_fold 2 --config ${CONFIG_PATH} --checkpoint "./checkpoints/mdeberta_2_20250331/checkpoint-573" --result_dir "submissions_mdeberta/fold2"
python inference.py --val_fold 3 --config ${CONFIG_PATH} --checkpoint "./checkpoints/mdeberta_3_20250331/checkpoint-573" --result_dir "submissions_mdeberta/fold3"
python inference.py --val_fold 4 --config ${CONFIG_PATH} --checkpoint "./checkpoints/mdeberta_4_20250331/checkpoint-573" --result_dir "submissions_mdeberta/fold4"
echo "Completed inference"