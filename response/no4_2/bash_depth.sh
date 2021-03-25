#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=10-10:15:00     # 10 day and 10 hours 15 minutes
#SBATCH --job-name="responcse"
#SBATCH -p wmalab 
#SBATCH --output=slurm-gather-%J.out

# Print current date
date
# Print name of node
hostname

CHR=${1}

source activate env_fr_test
echo python prepare_seq_lr.py ${CHR}
python prepare_seq_lr.py ${CHR}

source activate 3dchromatin_replicate_qc
echo python test_3dchromatin_qc.py ${CHR}
python test_3dchromatin_qc.py ${CHR}