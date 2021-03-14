#!/bin/sh
chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X')
# method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')
ID=(1 2 3 4 5 6 7)

#rm slurm-data-*.out
for c in "${chr[@]}"; do
    for i in "${ID[@]}"; do
        echo sbatch bash_3dchromatin.sh ${i} ${c}
        sbatch bash_3dchromatin.sh ${i} ${c}
    done
done

chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' 'X')
for c in "${chr[@]}"; do
    echo sbatch bash_3dchromatin.sh 0 ${c}
    sbatch bash_3dchromatin.sh 0 ${c}
done

chr=('19' '20' '21' '22' 'X'))
for c in "${chr[@]}"; do
    echo sbatch bash_3dchromatin.sh 8 ${c}
    sbatch bash_3dchromatin.sh 8 ${c}
done