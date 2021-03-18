#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=1-10:15:00     # 1 day and 10 hours 15 minutes
#SBATCH --job-name="data"
#SBATCH -p wmalab
# #SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
# #SBATCH --gres=gpu:k80:1
#SBATCH --output=slurm-data-%J.out

# Print current date
date
# Print name of node
hostname

METHOD=${1}
SCALE=${2}
source activate env_${METHOD}
echo python test_preprocessing_${METHOD}_seq.py ${SCALE}
python test_preprocessing_${METHOD}_seq.py ${SCALE}