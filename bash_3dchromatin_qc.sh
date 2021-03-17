#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=2-10:15:00     # 10 day and 10 hours 15 minutes
#SBATCH --job-name="qc"
#SBATCH -p wmalab # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
#SBATCH --output=slurm-qc-%J.out

# Print current date
date
# Print name of node
hostname

# source activate tf_base
# python gather_data.py

ID=${1}
CHROMOSOME=${2}

source activate 3dchromatin_replicate_qc
python test_3dchromatin_qc.py ${ID} ${CHROMOSOME}