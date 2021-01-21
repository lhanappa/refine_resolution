#!/bin/bash

rm slurm-train-*.out
rm slurm-tfboard-*.out

echo training
# len_size e.g. 128
# max_distance e.g 2560000
sbatch ./submit_training_model.sh ${1} ${2}

echo logging tensorboard
sbatch ./submit_tfb.sh
