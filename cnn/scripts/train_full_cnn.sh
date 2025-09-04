#!/bin/bash

#SBATCH --job-name=train_full_cnn
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=07-00:00:00
#SBATCH --output=../logs/relu/full_vgg_relu_cifar10.log

ACTIVATION="ReLU"

INIT="orthogonal"
INIT_SCALE=1e-1

DATA_TYPE="cifar10"
NUM_CLASSES=10
TRAIN_SAMPLES_PER_CLASS=5000
VAL_SAMPLES_PER_CLASS=1000

NUM_TRIALS=5
EPOCHS=250
LR=1e-2
LOSS="CrossEntropyLoss"
OPTIMIZER="SGD"
BATCH_SIZE=128

PARENT_SAVE_DIR="../results/"

python3 ../train_cnn.py --data_type $DATA_TYPE \
 --num_classes $NUM_CLASSES --train_samples_per_class $TRAIN_SAMPLES_PER_CLASS --val_samples_per_class $VAL_SAMPLES_PER_CLASS --one_hot \
 --activation $ACTIVATION --pt_conv \
 --init_method $INIT --init_scale $INIT_SCALE \
 --num_trials $NUM_TRIALS --epochs $EPOCHS --loss $LOSS --lr $LR --optimizer $OPTIMIZER --batch_size $BATCH_SIZE \
 --parent_save_dir $PARENT_SAVE_DIR


