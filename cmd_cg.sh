#!/bin/bash
VMPC_DATA_DIR=/home/angelina/deeprl_project/video_prediction-1/pretrained_models/sawyer_newenv/
 #$VMPC_DATA_DIR/towel_pick/ \

CUDA_VISIBLE_DEVICES=$1 python scripts/train.py --train_batch_sizes 16 \
 --input_dir \
 /media/3tb/sudeep/cartgripper_xz_grasp/pick_place_records/good/train \
 --dataset sawyer --model savp \
  --conf pretrained_models/ensemble_tests/baseline_train_cartgripper/ \
 --logs_dir pretrained_models/ensemble_tests/baseline_train_cartgripper/$2 \
 --dataset_hparams image_view=[0]
