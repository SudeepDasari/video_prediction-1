#!/bin/bash

CUDA_VISIBLE_DEVICES=0
#python scripts/evaluate.py --input_dir /media/3tb/annie/train --output_dir log --checkpoint pretrained_ensembles/model.ensemble_savp.None --dataset sawyer --batch_size 16 --ensemble
python scripts/evaluate.py --input_dir /raid/sudeep/experiments/grasp_reflex_cubes/records/val --output_dir log_multi --checkpoint /home/sudeep/Desktop/video_prediction-1/pretrained_models/ensemble_tests/train_cartgripper --dataset sawyer --batch_size 16 --ensemble
