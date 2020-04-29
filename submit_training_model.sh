#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=1-10:15:00     # 1 day and 15 minutes
#SBATCH --job-name="just_a_test"
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
#SBATCH --gres=gpu:k80:1

# Print current date
date

# Load samtools
#source activate tf_base
source activate tf_gpu_base
# run
python train.py

# Print name of node
hostname
