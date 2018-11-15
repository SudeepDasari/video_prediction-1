#!/bin/bash
VMPC_DATA_DIR=/home/angelina/deeprl_project/video_prediction-1/pretrained_models/sawyer_newenv/
 #$VMPC_DATA_DIR/towel_pick/ \

CUDA_VISIBLE_DEVICES=1 python scripts/train.py --train_batch_sizes 16 \
 --input_dir \
 /raid/sudeep/towel_pick_30k/train/ \
 --dataset sawyer --model ensemble_savp \
  --conf pretrained_models/ensemble_tests/towel_pick/ \
 --logs_dir pretrained_models/ensemble_tests/towel_pick/view0 \
 --dataset_hparams image_view=[0]
