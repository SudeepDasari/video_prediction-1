#!/bin/bash

CUDA_VISIBLE_DEVICES=0 
python scripts/evaluate.py --input_dir /raid/sudeep/cartgripper_xz_grasp/vanilla_env_rand_actions/bad/train --output_dir log_1 --checkpoint /home/sudeep/Desktop/video_prediction-1/pretrained_models/ensemble_tests/baseline_train_cartgripper/ens1/model.savp.None --dataset sawyer --batch_size 16
python scripts/evaluate.py --input_dir /raid/sudeep/cartgripper_xz_grasp/vanilla_env_rand_actions/bad/train --output_dir log_2 --checkpoint /home/sudeep/Desktop/video_prediction-1/pretrained_models/ensemble_tests/baseline_train_cartgripper/ens2/model.savp.None --dataset sawyer --batch_size 16
#python scripts/evaluate.py --input_dir /raid/sudeep/cartgripper_xz_grasp/vanilla_env_rand_actions/bad/train --output_dir log_3 --checkpoint /home/sudeep/Desktop/video_prediction-1/pretrained_models/ensemble_tests/baseline_train_cartgripper/ens3/model.savp.None --dataset sawyer --batch_size 16
#python scripts/evaluate.py --input_dir /raid/sudeep/cartgripper_xz_grasp/vanilla_env_rand_actions/bad/train --output_dir log_4 --checkpoint /home/sudeep/Desktop/video_prediction-1/pretrained_models/ensemble_tests/baseline_train_cartgripper/ens4/model.savp.None --dataset sawyer --batch_size 16
