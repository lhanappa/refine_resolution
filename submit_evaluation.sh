#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=10-10:15:00     # 10 day and 10 hours 15 minutes
#SBATCH --job-name="evaluation"
#SBATCH -p wmalab # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
###SBATCH --gres=gpu:k80:1
#SBATCH --output=slurm-evaluation-%J.out

# Print current date
date
# Print name of node
hostname

METHOD=${1}
CHR=${2}
source activate tf_base
echo python test_evaluation.py ${METHOD} ${CHR}
python test_evaluation.py ${METHOD} ${CHR}