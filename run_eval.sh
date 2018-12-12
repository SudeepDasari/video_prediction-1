#!/bin/bash

CUDA_VISIBLE_DEVICES=0 
#python scripts/evaluate.py --input_dir /media/3tb/annie/train --output_dir log --checkpoint pretrained_ensembles/model.ensemble_savp.None --dataset sawyer --batch_size 16 --ensemble
python scripts/evaluate.py --input_dir /raid/sudeep/cartgripper_xz_grasp/vanilla_env_rand_actions/bad/train --output_dir log --checkpoint /home/angelina/deeprl_project/video_prediction-1/pretrained_models/ensemble_tests/train_cartgripper/view0/model.ensemble_savp.None --dataset sawyer --batch_size 16 --ensemble
