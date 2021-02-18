#!/bin/sh
chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X')
# method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')
method=('ours_400' 'deephic_40' 'hicsr_40' 'low')

#rm slurm-data-*.out
#for c in "${chr[@]}"; do
#    echo sbatch bash_gather_tad_boundary.sh ${c}
#    sbatch bash_gather_tad_boundary.sh ${c}
#done

for c in "${chr[@]}"; do
    echo sbatch bash_tad_boundary.sh ${c}
    sbatch bash_tad_boundary.sh ${c}
done