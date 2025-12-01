#!/bin/bash

#SBATCH --job-name=train_compressed_roberta_classifier
#SBATCH --account=girasole1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=07-00:00:00
#SBATCH --output=../logs/sst2/comp_roberta_classifier_cola_rand_init.log


GLUE_TASK_NAME="sst2"

NUM_LAYERS=2
SUBSPACE_TYPE="random"

INIT_SCALE=1e-1

NUM_TRIALS=5
BATCH_SIZE=32
EPOCHS=100
LR=1e-3
ACCUM_STEPS=1

PARENT_SAVE_DIR="../results/"

python3 ../train_roberta_classifier.py --glue_task_name $GLUE_TASK_NAME \
 --num_classifier_layers $NUM_LAYERS --init_scale $INIT_SCALE --compress --subspace_type $SUBSPACE_TYPE \
 --num_trials $NUM_TRIALS --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --accum_steps $ACCUM_STEPS \
 --parent_save_dir $PARENT_SAVE_DIR


