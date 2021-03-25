#!/bin/sh
chr=('22' '21' '20' '19' '18' '17' '16' '15' 'X' '14' '13' '12' '11' '10' '9' '8' '7' '6' '5' '4' '3' '2' '1') # '15' '16' '17' '18' '19' '20' '21' '22' 

#rm slurm-data-*.out
for c in "${chr[@]}"; do
    echo sbatch bash_depth.sh ${c}
    sbatch bash_depth.sh ${c}
done