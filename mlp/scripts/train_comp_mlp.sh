#!/bin/bash

#SBATCH --job-name=train_comp_mlp
#SBATCH --account=girasole1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=02-00:00:00
#SBATCH --output=../logs/test_angle.log

DEPTH=4
ACTIVATION="GELU"

INIT="orthogonal"
INIT_SCALE=1e-1

DATA_TYPE="fashion_mnist"
NUM_CLASSES=10
TRAIN_SAMPLES_PER_CLASS=6000
VAL_SAMPLES_PER_CLASS=1000
DATA_DIM=784

SUBSPACE_ANGLE=45

NUM_TRIALS=5
EPOCHS=1500
LR=1e-5 # ReLU, GELU, SiLU: LR = 1e-5, ELU: LR = 5e-6
LOSS="MSELoss"
OPTIMIZER="SGD"

PARENT_SAVE_DIR="../results/"

python3 ../train_mlp.py --data_type $DATA_TYPE --data_dim $DATA_DIM \
 --num_classes $NUM_CLASSES --train_samples_per_class $TRAIN_SAMPLES_PER_CLASS --val_samples_per_class $VAL_SAMPLES_PER_CLASS \
 --hidden_dim $DATA_DIM --depth $DEPTH --activation $ACTIVATION --compress --subspace_type 2 --subspace_angle $SUBSPACE_ANGLE \
 --init_method $INIT --init_scale $INIT_SCALE \
 --num_trials $NUM_TRIALS --epochs $EPOCHS --loss $LOSS --lr $LR --optimizer $OPTIMIZER \
 --parent_save_dir $PARENT_SAVE_DIR


