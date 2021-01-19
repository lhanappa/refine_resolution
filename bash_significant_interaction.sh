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

CHR=${1} # 22
source activate env_fr_test
# echo python significant_interactions_gather_data.py ${CHR}
# python significant_interactions_gather_data.py ${CHR}
echo python test_significant_interaction.py ${CHR}
python test_significant_interaction.py ${CHR}