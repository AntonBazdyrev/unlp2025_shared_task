#!/bin/bash

CONFIG_PATH="configs/gemma2-27b-config.yaml"

for fold in {0..4}
do
    echo "Starting training for fold ${fold}..."
    python train.py --val_fold ${fold} --config ${CONFIG_PATH}
    echo "Completed training for fold ${fold}."
done
