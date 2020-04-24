#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --time=1-10:15:00     # 1 day and 15 minutes
#SBATCH --job-name="just_a_test"
#SBATCH -p wmalab # This is the default partition, you can use any of the following; intel, batch, highmem, gpu

# Print current date
date

# Load samtools
source activate tf_base

# run
python train.py

# Print name of node
hostname
