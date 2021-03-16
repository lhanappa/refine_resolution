#!/bin/sh
chr=('22' '21') # '20' '19' '18' '17' '16' '15' 'X' '14' '13' '12' '11' '10' '9' '8' '7' '6' '5' '4' '3' '2' '1') # '15' '16' '17' '18' '19' '20' '21' '22' 
# method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')
ID=(4) # 8 16 32 48 64)

#rm slurm-data-*.out
for c in "${chr[@]}"; do
    for i in "${ID[@]}"; do
        echo sbatch bash_3dchromatin_qc_seq.sh ${i} ${c}
        sbatch bash_3dchromatin_qc.sh ${i} ${c}
    done
done
