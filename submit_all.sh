#!/bin/bash

rm slurm-*.out
rm tfboard-*.out

echo training
sbatch ./submit_training_model.sh

echo logging tensorboard
sbatch ./submit_tfb.sh
