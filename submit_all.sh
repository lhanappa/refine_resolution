#!/bin/bash

rm slurm-*.out
rm tfboard-*.log

echo training
sbatch ./submit_training_model.sh

echo logging tensorboard
sbatch ./submit_tfb.sh
