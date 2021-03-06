#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=10-10:15:00     # 10 day and 10 hours 15 minutes
#SBATCH --job-name="gather data"
#SBATCH -p wmalab 
#SBATCH --output=slurm-gather-%J.out

# Print current date
date
# Print name of node
hostname


source activate env_fr_test

SCALE=${1} # 0 1
CHR=${2}
echo python evaluation_gather_data_seq.py ${SCALE} ${CHR}
python evaluation_gather_data_seq.py ${SCALE} ${CHR}