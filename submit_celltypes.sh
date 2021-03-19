#!/bin/sh

# method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')
chr=('19' '18' '17' '16' '15' 'X' '14' '13' '12' '11' '10' '9' '8' '7' '6' '5' '4' '3' '2' '1') # '22' '21' '20' 
ID=(0 1)

#rm slurm-data-*.out
for c in "${chr[@]}"; do
    for i in "${ID[@]}"; do
        echo sbatch bash_3dchromatin_qc.sh ${i} ${c}
        sbatch bash_3dchromatin_qc.sh ${i} ${c}
    done
done


# chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' 'X') # '15' '16' '17' '18' '19' 
# for c in "${chr[@]}"; do
#    echo sbatch bash_3dchromatin_qc.sh 0 ${c}
#    sbatch bash_3dchromatin_qc.sh 0 ${c}
# done

# chr=('19' '20' '21' '22' 'X')
# for c in "${chr[@]}"; do
#     echo sbatch bash_3dchromatin_qc.sh 8 ${c}
#    sbatch bash_3dchromatin_qc.sh 8 ${c}
# done