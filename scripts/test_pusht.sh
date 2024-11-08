#!/bin/bash
set -e

CONFIG_FOLDER="diffusion_policy/config"

# Vanilla PushT CNN
echo  "Running PushT with CNN (End2End)"
python train.py --config-dir=$CONFIG_FOLDER --config-name=train_pusht.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# PushT Using ResNet-ImageNet
echo  "Running PushT with CNN (Imagenet)"
python train.py --config-dir=$CONFIG_FOLDER --config-name=train_pusht_pretrained_imagenet.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# PushT Using ResNet-R3M
echo  "Running PushT with CNN (R3M)"
python train.py --config-dir=$CONFIG_FOLDER --config-name=train_pusht_pretrained_r3m.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
