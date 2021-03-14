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

ID=${1} # 22
source activate env_fr_test
echo python evaluation_gather_data.py ${ID}
python evaluation_gather_data.py ${ID}